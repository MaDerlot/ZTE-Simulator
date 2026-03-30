# 时空切分并行仿真技术报告

## 1. 概述

ZTE-Simulator 的时空切分并行仿真，是基于大模型训练流量的**拓扑局部性**特征，将 NS-3 单进程仿真扩展为多 MPI 进程并行执行的实现方案。

**核心洞察**：大模型训练流量天然具有"空间聚集性"——
- **TP/EP 流量**（Tensor/Expert Parallelism）在同一 Pod 内的 Server 之间通信，不跨 Pod；
- **DP 流量**（Data Parallelism）跨 Pod，对应 MPI 进程间边界；
- **Spine 链路**（2μs 延迟）是跨 Pod 的唯一通道，其传播延迟天然提供保守并行仿真所需的 Lookahead 约束。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         MPI 并行仿真架构                          │
│                                                                   │
│  Rank 0 (Pod 0)          Rank 1 (Pod 1)      Rank N-1 (Pod N-1) │
│  ┌─────────────┐         ┌─────────────┐      ┌─────────────┐   │
│  │  ToR0       │         │  ToR1       │      │  TorN-1     │   │
│  │  Server 0-K │         │  Server K-2K│      │  Server ... │   │
│  └──────┬──────┘         └──────┬──────┘      └──────┬──────┘   │
│         │  QbbRemoteChannel     │  QbbRemoteChannel   │          │
│         └──────────────── Spine（Ghost，所有 Rank 创建）──────────┘
│                                                                   │
│  进程间通信：NullMessage MPI（保守并行离散事件仿真）                │
└─────────────────────────────────────────────────────────────────┘
```

**Ghost Node 设计**：Spine 交换机在每个 MPI 进程中都创建，但其 `SystemId` 统一设为 0（由 Rank 0 拥有）。这样路由表可在所有进程中正确计算，但跨进程数据包仍通过 `QbbRemoteChannel` 路由到正确目标进程。

---

## 3. 空间分区算法（partition.py）

### 3.1 功能

读取 Leaf-Spine 拓扑文件，按 Pod 归属关系将节点分配到 MPI rank，输出 `node_partition.txt`。

### 3.2 拓扑文件格式

```
total_nodes  switch_count  tor_count  link_count  leaf_bw  spine_bw
<switch_id_0> <switch_id_1> ... <switch_id_N>
src dst data_rate delay error_rate
...
```

前 `tor_count` 个 switch 为 ToR（接入层），其余为 Spine（汇聚层）。

### 3.3 分区规则

```python
def partition_topology(topo_file, num_ranks, output_file):
    # 1. 解析 link 列表，建立 server -> ToR 映射
    for (src, dst, ...) in links:
        if src not in switch_ids and dst in tor_ids:
            server_to_tor[src] = dst     # src 是 server

    # 2. ToR 轮询分配到各 rank
    for idx, tor_id in enumerate(tor_ids):
        tor_to_rank[tor_id] = idx % num_ranks

    # 3. 节点 rank 分配
    for node_id in range(total_nodes):
        if node_id in spine_ids:
            node_rank[node_id] = -1          # Ghost：所有 rank 都创建
        elif node_id in tor_ids:
            node_rank[node_id] = tor_to_rank[node_id]
        else:
            node_rank[node_id] = tor_to_rank[server_to_tor[node_id]]
```

### 3.4 输出格式

```
# node_id,rank  (rank=-1 means ghost/all-ranks Spine node)
0,0
1,0
8,-1    # Spine，ghost
16,1
...
```

---

## 4. QBB RemoteChannel 支持（qbb-helper.cc）

### 4.1 跨 rank 链路检测

`QbbHelper::Install()` 在创建链路时检查两端节点的 `SystemId`，若不同则改用 `QbbRemoteChannel`：

```cpp
// src/point-to-point/helper/qbb-helper.cc
bool useNormalChannel = true;
#ifdef NS3_MPI
if (MpiInterface::IsEnabled()) {
    uint32_t n1SystemId = a->GetSystemId();
    uint32_t n2SystemId = b->GetSystemId();
    uint32_t currSystemId = MpiInterface::GetSystemId();
    if (n1SystemId != currSystemId || n2SystemId != currSystemId) {
        useNormalChannel = false;   // 跨 rank，使用 RemoteChannel
    }
}
#endif
```

### 4.2 RemoteChannel 创建

```cpp
#ifdef NS3_MPI
else {
    // 创建 QbbRemoteChannel（继承自 QbbChannel）
    channel = m_remoteChannelFactory.Create<QbbRemoteChannel>();
    
    // 为两端设备分别聚合 MpiReceiver
    Ptr<MpiReceiver> mpiRecA = CreateObject<MpiReceiver>();
    Ptr<MpiReceiver> mpiRecB = CreateObject<MpiReceiver>();
    mpiRecA->SetReceiveCallback(MakeCallback(&QbbNetDevice::Receive, devA));
    mpiRecB->SetReceiveCallback(MakeCallback(&QbbNetDevice::Receive, devB));
    devA->AggregateObject(mpiRecA);
    devB->AggregateObject(mpiRecB);
}
#endif
```

**继承关系**：`QbbRemoteChannel : public QbbChannel`，因此现有代码中所有 `DynamicCast<QbbChannel>` 调用（用于获取链路延迟）对 RemoteChannel 同样有效，无需修改。

### 4.3 MpiReceiver 的作用

`MpiReceiver` 是 NS-3 MPI 框架提供的接收器，通过 `AggregateObject` 挂载到 `NetDevice`。当 MPI 消息从远程进程到达时，NS-3 的 MPI 层通过 `MpiReceiver::Receive()` 将数据包注入本地设备队列，对上层透明。

---

## 5. 仿真主程序的 MPI 集成

### 5.1 MPI 初始化

```cpp
// examples/Reverie/reverie-evaluation-sigcomm2023.cc: main()
#ifdef NS3_MPI
    MpiInterface::Enable(&argc, &argv);
    systemId = MpiInterface::GetSystemId();   // 本进程 rank
    systemNum = MpiInterface::GetSize();       // 总进程数
    if (systemNum > 1) {
        // 启用保守 NullMessage 并行仿真器
        GlobalValue::Bind("SimulatorImplementationType",
                          StringValue("ns3::NullMessageSimulatorImpl"));
    }
#endif
```

### 5.2 分区映射加载

```cpp
// 全局映射：node_id -> mpi_rank（-1 = ghost/spine）
std::map<uint32_t, int32_t> nodeToRank;

void loadPartition(const std::string& partFile) {
    std::ifstream f(partFile);
    std::string line;
    while (std::getline(f, line)) {
        size_t comma = line.find(',');
        uint32_t node = std::stoul(line.substr(0, comma));
        int32_t  rank = std::stol(line.substr(comma + 1));
        nodeToRank[node] = rank;
    }
}
```

通过命令行参数 `--partitionFile=<path>` 指定分区文件，为空则退化为单进程模式。

### 5.3 节点 SystemId 赋值

```cpp
// 拓扑创建后，按分区映射设置每个节点的 SystemId
#ifdef NS3_MPI
if (!nodeToRank.empty()) {
    for (uint32_t i = 0; i < node_num; i++) {
        auto it = nodeToRank.find(i);
        if (it != nodeToRank.end()) {
            // Spine（rank=-1）统一赋给 rank 0
            uint32_t assignedRank = (it->second == -1) ? 0 : (uint32_t)it->second;
            n.Get(i)->SetAttribute("SystemId", UintegerValue(assignedRank));
        }
    }
}
#endif
```

Spine 节点设为 rank 0 的原因：QbbHelper 在安装链路时，若任一端不属于当前 rank，则使用 RemoteChannel。Spine 作为"ghost node"，与各 Pod 的连接均应为 RemoteChannel（除 rank 0 外），这样实现最自然。

### 5.4 流量过滤（sendPhase / workload_rdma）

每个进程只负责发送源节点属于本 rank 的流：

```cpp
void sendPhase(uint16_t par) {
    for (FlowInfo flow : flowInfos[par][phaseCur[par]]) {
        if (!nodeToRank.empty()) {
            auto it = nodeToRank.find(flow.src_node);
            // 跳过不属于本 rank 的流（ghost/spine 节点不主动发流）
            if (it == nodeToRank.end() ||
                (it->second != -1 && (uint32_t)it->second != systemId))
                continue;
        }
        flowSend(flow);
    }
}
```

流量接收（QP 完成回调 `flowinput_cb`）无需过滤——每个进程只持有本 rank 节点的 RDMA 驱动，自然只会收到属于本进程的完成事件。

### 5.5 MPI 销毁

```cpp
Simulator::Destroy();
#ifdef NS3_MPI
    MpiInterface::Disable();    // 释放 MPI 资源
#endif
```

---

## 6. NullMessage 保守并行仿真

### 6.1 算法原理

`NullMessageSimulatorImpl` 实现了**保守 NullMessage 并行离散事件仿真**（PDES）算法：

- 每个进程维护本地事件队列；
- 进程间通过 `RemoteChannelBundle` 交换"安全时间"（Safe Time）；
- 进程只处理时间戳 ≤ Safe Time 的事件，保证因果一致性。

### 6.2 Lookahead 自动检测

```cpp
// null-message-simulator-impl.cc: CalculateLookAhead()
void NullMessageSimulatorImpl::CalculateLookAhead() {
    for (NodeContainer::Iterator iter = c.Begin(); ...) {
        // 枚举所有跨 rank 的 RemoteChannel
        for (uint32_t i = 0; i < (*iter)->GetNDevices(); ++i) {
            TimeValue delay;
            channel->GetAttribute("Delay", delay);
            // 将 RemoteChannel 的延迟加入对应 RemoteChannelBundle
            remoteChannelBundle->AddChannel(channel, delay.Get());
        }
    }
}
```

`CalculateLookAhead()` 自动扫描所有 `QbbRemoteChannel`，提取其 `Delay` 属性作为 Lookahead 窗口。Leaf-Spine 拓扑中 Spine 链路延迟为 **2μs**，这是跨 Pod 通信的最短传播路径，自动成为 Lookahead 约束——无需手动配置。

### 6.3 NullMessage 发送

```cpp
// null-message-simulator-impl.cc: NullMessageEventHandler()
void NullMessageSimulatorImpl::NullMessageEventHandler(RemoteChannelBundle* bundle) {
    // 告知对方：本进程当前安全时间为 Next() + bundle->GetDelay()
    Time time = Min(Next(), GetSafeTime()) + bundle->GetDelay();
    NullMessageMpiInterface::SendNullMessage(time, bundle);
    ScheduleNullMessageEvent(bundle);  // 定期重发
}
```

Null Message 携带的时间戳让接收方知道发送方至少会在该时间之后才会发送数据包，从而可安全推进本地事件队列。

### 6.4 安全时间更新

```cpp
void NullMessageSimulatorImpl::CalculateSafeTime() {
    m_safeTime = RemoteChannelBundleManager::GetSafeTime();
}

void NullMessageSimulatorImpl::Run() {
    CalculateLookAhead();
    RemoteChannelBundleManager::InitializeNullMessageEvents();
    while (!IsFinished()) {
        Time nextTime = Next();
        if (nextTime <= GetSafeTime()) {
            ProcessOneEvent();      // 因果安全，可以执行
        } else {
            // 等待其他进程的 Null Message 更新 SafeTime
        }
    }
}
```

---

## 7. 并行启动脚本（kira-mpi-parallel.sh）

```bash
#!/bin/bash
# 使用方式: ./kira-mpi-parallel.sh <num_ranks> <task_index>
NUM_RANKS=${1:-4}
TASKINDEX=${2:-1}

# 步骤1: 编译
cd ../..
./ns3 build reverie-evaluation-sigcomm2023
cd examples/Reverie

# 步骤2: 生成空间分区文件
python3 partition.py \
    --topo leaf-spine.txt \
    --ranks $NUM_RANKS \
    --output task${TASKINDEX}/node_partition.txt

# 步骤3: 启动 MPI 并行仿真
mpirun -np $NUM_RANKS $SIM_BIN \
    --TASKINDEX=$TASKINDEX \
    --partitionFile=examples/Reverie/task${TASKINDEX}/node_partition.txt \
    --START_TIME=0 \
    --END_TIME=10000 \
    ... > $DUMPFILE 2>&1
```

---

## 8. 构建配置（CMakeLists.txt）

MPI 支持通过 `ENABLE_MPI` 开关条件编译，保证在无 MPI 环境下仍可正常构建：

```cmake
# examples/Reverie/CMakeLists.txt
set(reverie_mpi_libs)
if(${ENABLE_MPI})
  set(reverie_mpi_libs ${libmpi})
endif()

build_example(
  NAME reverie-evaluation-sigcomm2023
  LIBRARIES_TO_LINK
    ${libpoint-to-point}
    ${libinternet}
    ${libapplications}
    ${reverie_mpi_libs}     # MPI 未启用时为空
)
```

同样地，`qbb-helper.cc` 和 `reverie-evaluation-sigcomm2023.cc` 中所有 MPI 相关代码均被 `#ifdef NS3_MPI` 保护。

---

## 9. 时空切分的"时间"维度

本方案的"时间切分"体现在**与混合粒度仿真的协同**：

- **空间切分**：按 Pod 将拓扑划分给不同 MPI 进程；
- **时间切分**：每个进程内部使用混合粒度跃迁，在稳态期间跳过时间轴；
- **协同效果**：空间切分减少了单进程需要仿真的节点数量（1/N），时间切分减少了每个进程内部的事件数量，两者相乘带来接近 N 倍的仿真加速。

---

## 10. 性能预期与约束

| 指标 | 说明 |
|------|------|
| **理论加速比** | 接近 N（进程数），受 Spine 链路通信开销限制 |
| **Lookahead 约束** | 2μs Spine 延迟，决定进程间同步频率（越大越好） |
| **负载均衡** | 每 rank 持有 1/N 的 Pod，TP/EP 流完全本地，DP 流跨 rank |
| **内存节省** | 每进程只维护本 rank 节点的完整状态，Spine 作 ghost 略有冗余 |
| **可扩展性** | rank 数量 = Pod 数量（或其因子），最大并行度由 ToR 数量决定 |

---

## 11. 启用方式

```bash
# 1. 安装 OpenMPI（Ubuntu）
sudo apt install libopenmpi-dev openmpi-bin

# 2. 重新配置 NS-3 以启用 MPI
cd /home/lb/ZTE-Simulator
./ns3 configure --enable-mpi --enable-examples

# 3. 构建
./ns3 build reverie-evaluation-sigcomm2023

# 4. 运行（4 进程并行）
cd examples/Reverie
./kira-mpi-parallel.sh 4 1
```

不安装 MPI 时，所有 `#ifdef NS3_MPI` 块被跳过，仿真器以原有单进程模式正常运行。
