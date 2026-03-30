# 混合粒度仿真技术报告

## 1. 概述

大规模数据中心网络仿真面临严峻的**计算效率矛盾**：

- **包级仿真**：精度高，能捕捉每一个数据包的行为，但对于大流量、长流（如大模型训练中的 GB 级 AllReduce）会产生海量事件，仿真时间极长；
- **流级仿真**：速度快，直接用解析模型估算流完成时间，但丢失了拥塞细节和排队动态。

ZTE-Simulator 实现了一种**混合粒度（Mixed-Granularity）仿真机制**，在同一仿真运行中动态切换精度：系统在流量稳定（稳态）期间"跳跃"时间轴，跳过大量重复的包级事件，仅在流量变化（瞬态）期间精确模拟每个数据包。

---

## 2. 核心概念：稳态跃迁（Steady-State Jump）

### 2.1 稳态定义

当一条流的发送速率在连续若干测量窗口内保持稳定（速率波动 ≤ `2.22045e-16`，即双精度浮点机器 epsilon），则认为该流进入**稳态**。当系统中所有活跃流均进入稳态时，系统整体进入稳态。

```cpp
// qbb-net-device.cc: calculateRate()
if (std::fabs(*maxIt - *minIt) <= 2.22045e-16) {
    stat.steadyStateReached = true;   // 单流进入稳态
} else {
    allSteadyStateReached = false;    // 任何一流不稳，系统不稳
}
```

### 2.2 跃迁触发条件

```cpp
if (allSteadyStateReached) {
    calculateMintime();               // 计算最短剩余完成时间
    transition_delay.push_back(transMinTime);  // 记录跃迁量
    transition_cnt++;                 // 全局跃迁计数器 +1
}
```

当系统进入稳态后：
1. 调用 `calculateMintime()` 扫描所有节点上的所有 QP（RDMA Queue Pair），找出按当前速率完成所需时间最短的流；
2. 将该最短完成时间作为**跃迁延迟**（`transMinTime`）压入 `transition_delay` 向量；
3. 递增全局跃迁计数器 `transition_cnt`。

---

## 3. 全局状态变量

### 3.1 mixed-granularity.h

```cpp
// src/core/model/mixed-granularity.h
extern int transition_cnt;                  // 已发生的跃迁次数
extern std::vector<uint64_t> transition_delay;  // 每次跃迁的时间跨度（ns）
```

这两个全局变量是混合粒度机制的"共享状态"，由 `qbb-net-device.cc`（检测稳态、写入）和 `default-simulator-impl.cc`（读取、修改事件时间戳）共同使用。

### 3.2 最短流完成时间计算

```cpp
// qbb-net-device.cc: calculateMintime()
void QbbNetDevice::calculateMintime() {
    for (NodeList::Iterator it = NodeList::Begin(); ...) {
        for (uint32_t i = 0; i < node->GetNDevices(); ++i) {
            // 遍历每个 QbbNetDevice 上的 RDMA 事件队列
            for (auto& qp : rdmaHw->m_rdmaEQ) {
                double avg = mean(flowStats[flowid].rate);   // 当前平均速率
                double trans_time = (qp->GetBytesLeft() - 1000) 
                                    / qp->m_rate.GetBitRate() * 1e9 * 8;
                if (trans_time < flowMinTime) {
                    flowMinTime  = trans_time;
                    transMinTime = trans_time;   // 记录最小值
                }
            }
        }
    }
}
```

`GetBytesLeft() - 1000` 排除了最后一个包（FIN/控制包），避免将即将完成的流计入主要传输时间。

---

## 4. 事件时间戳修改（DefaultSimulatorImpl）

混合粒度的"跳跃"通过在 **事件出队时修改其时间戳** 来实现，对 NS-3 事件调度器的修改最小化。

### 4.1 事件新建时的标记

每个新创建的事件记录当前跃迁阶段：

```cpp
// default-simulator-impl.cc: Schedule()
ev.key.transition_stage = transition_cnt;  // 记录创建时的跃迁计数
```

### 4.2 出队时的时间戳补偿

```cpp
// default-simulator-impl.cc: ProcessOneEvent()
Scheduler::Event next = m_events->RemoveNext();

bool flag = false;
while (next.key.transition_stage < transition_cnt) {
    // 事件创建后发生了新的跃迁，补加跃迁延迟
    next.key.m_ts += transition_delay[next.key.transition_stage];
    next.key.transition_stage++;
    flag = true;
}
if (flag) {
    // 时间戳已被修改，需要重新插入并对队列中所有其他事件做同样补偿
    std::deque<Scheduler::Event> l;
    while (!m_events->IsEmpty()) {
        Scheduler::Event evt = m_events->RemoveNext();
        while (evt.key.transition_stage < transition_cnt) {
            evt.key.m_ts += transition_delay[evt.key.transition_stage];
            evt.key.transition_stage++;
        }
        l.push_back(evt);
    }
    // 将所有事件重新放回队列
    for (auto& e : l) m_events->Insert(e);
}
```

**关键设计**：事件不是在创建时加上跃迁偏移量，而是在**出队时**才补偿。这样避免了对调度器插入逻辑的侵入式修改，同时保证了因果一致性。

---

## 5. 稳态退出（exitSteadyState）

跃迁发生后，系统需要更新各流的发送状态以与新的时间基准同步：

```cpp
// qbb-net-device.cc: exitSteadyState()
void QbbNetDevice::exitSteadyState() {
    for (NodeList::Iterator it = NodeList::Begin(); ...) {
        // 遍历所有 RDMA QP
        // 1. 修正排队延迟：用队列速率差 × 跃迁时间估算排队积压
        Time latency_fix = bps.CalculateBytesTxTime(
            transition_delay[transition_cnt] * queue_rate.GetBitRate());
        node_latency_fix[node->GetId()] = latency_fix;
        
        // 2. 更新已发送序号：snd_nxt += rate × 跃迁时间（ns）
        qp->snd_nxt += qp->m_rate.GetBitRate() / 8e9 
                       * transition_delay[transition_cnt - 1];
        
        // 3. 清除该流的速率统计，准备进入下一阶段测量
        flowStats.erase(flowid);
    }
}
```

`snd_nxt` 的更新使得 RDMA 协议层"认为"在跃迁期间已经发送了对应字节量，从而与实际跳过的时间保持一致。

---

## 6. 整体工作流程

```
┌─────────────────────────────────────────────────────────┐
│                   包级仿真阶段（精确模式）                  │
│  每个数据包独立调度，处理拥塞控制（QBB/DCQCN）、排队等      │
└──────────────┬──────────────────────────────────────────┘
               │ 检测到所有流速率稳定
               ↓
┌─────────────────────────────────────────────────────────┐
│              稳态检测（calculateRate）                    │
│  - 所有流 steadyStateReached == true                    │
│  - calculateMintime() → transMinTime（最短剩余时间）      │
│  - transition_delay.push_back(transMinTime)             │
│  - transition_cnt++                                     │
└──────────────┬──────────────────────────────────────────┘
               │ Simulator::ScheduleNow(exitSteadyState)
               ↓
┌─────────────────────────────────────────────────────────┐
│              时间轴跳跃（ProcessOneEvent 补偿）            │
│  - 出队事件：m_ts += transition_delay[stage]            │
│  - 队列中所有事件同步补偿                                 │
│  - 有效"快进"仿真时钟 transMinTime 纳秒                   │
└──────────────┬──────────────────────────────────────────┘
               │ 跳跃完成
               ↓
┌─────────────────────────────────────────────────────────┐
│              状态更新（exitSteadyState）                  │
│  - snd_nxt += rate × 跃迁时间（补偿已发送字节）           │
│  - 清除流速率统计，重置稳态标志                            │
└──────────────┬──────────────────────────────────────────┘
               │ 继续包级仿真
               ↓
         （循环至所有流完成）
```

---

## 7. 与 REVERIE 缓存管理的关系

混合粒度跃迁机制专为 REVERIE（SIGCOMM 2023）缓存感知流量调度设计。REVERIE 的核心目标是让大规模 AllReduce 流量在网络缓存中平稳通过，不产生突发拥塞。

在稳态期间，网络处于"匀速传输"状态，此时跃迁跳过的正是 REVERIE 希望保持的稳定传输阶段；跃迁后系统重新进入瞬态，恢复包级精度来捕捉流量切换时的拥塞动态，实现了**仿真精度与速度的最优权衡**。

---

## 8. 关键参数

| 变量 | 类型 | 说明 |
|------|------|------|
| `transition_cnt` | `int` | 已发生的跃迁次数，新事件的 `transition_stage` 初始值 |
| `transition_delay[i]` | `uint64_t (ns)` | 第 i 次跃迁的时间跨度 |
| `transMinTime` | `double` | 当前稳态下最短流完成时间（ns） |
| `flowMinTime` | `double` | 全局最小流完成时间缓冲变量 |
| `next_trans_avail` | `uint64_t` | 下一次可触发跃迁的最早时间 |
| `allSteadyStateReached` | `bool` | 系统级稳态标志 |

---

## 9. 设计优势

| 特性 | 说明 |
|------|------|
| **非侵入式** | 仅修改 `ProcessOneEvent()` 出队逻辑，不改变调度器接口 |
| **自适应** | 跃迁时机由运行时速率检测决定，无需用户手动配置 |
| **精度保证** | 瞬态（流启动/结束/拥塞）仍为包级精确模拟 |
| **加速比** | 对长流（GB 级 AllReduce）可将稳态中数百万包的仿真压缩为单次时间戳修改 |
