# 图驱动的集合通信流量技术报告

## 1. 概述

ZTE-Simulator 中的图驱动集合通信流量系统，以**有向无环图（DAG）**为核心数据结构，对大模型训练过程中的算子级通信依赖进行建模，并据此生成精准的 RDMA 流量。整个系统由三个层次构成：

```
deepseek.txt / qwen.txt          ← 模型描述输入
        ↓
  trafficGen.py (DAG 构建 + 流量生成)
        ↓
  rdma_result/<节点>/ rdma_operateX.txt    ← 每算子的 RDMA 流文件
        ↓
  merge_dependency.sh + TreeCut.py         ← 依赖关系提取与合并
        ↓
  dependence.txt                           ← 算子间依赖配置
        ↓
  reverie-evaluation-sigcomm2023.cc        ← 仿真主程序驱动执行
```

---

## 2. 模型输入格式

输入文件（`deepseek.txt` / `qwen.txt`）的格式如下：

```
deepseek 4          # 模型名称 + 设备数量
# layer  mode  parent_node  [--参数 值 ...]
1  DP    Root  --host_num 128 --msg_len 32*1024*1024
2  TP    DP1   --num_nodes 8  --num_phases 7
3  EP    TP1   --host_num 128 --device 0
4  PP    TP1   --dp 2
```

**支持的并行模式：**

| 模式 | 含义 | 通信原语 |
|------|------|----------|
| TP   | Tensor Parallelism | Ring AllReduce（send_tensor_parallel） |
| EP   | Expert Parallelism | All-to-All |
| PP   | Pipeline Parallelism | 点对点传递 |
| DP   | Data Parallelism | Hypercube AllReduce |

---

## 3. DAG 构建（trafficGen.py）

### 3.1 核心数据结构

使用 `networkx.DiGraph` 作为 DAG 容器。每个节点携带属性：

```python
G.add_node(node_name,
    name=node_name,
    type='TP',         # 通信类型
    layer=2,           # 层编号
    nodes_passed=['A','B','C'],  # 经过的设备标识
    children=[],       # 子节点列表
    args={             # 通信参数
        'host_num': 128,
        'num_nodes': 8,
        'dp': 1,
        'msg_len': 33554432,
        'num_phases': 7,
        'num_iterations': 1,
        'device': 0,
        'forward': 1
    }
)
```

### 3.2 构建流程

`constructTree()` 函数逐行解析输入文件，将每条通信算子添加为 DAG 节点，并通过 `parent_node` 字段建立有向边（依赖关系）：

```
Root → DP1 → TP1 → EP1
              ↓
             PP1 → TP2
```

特殊处理：TP 节点支持多父节点（通过 `/` 分隔），使用 `tp_multi_parent_nodes` 字典去重，避免重复创建节点。

### 3.3 拓扑遍历顺序

`generate_traffic_from_tree()` 使用 `nx.topological_sort()` 遍历 DAG，保证父节点的流量总在子节点之前生成，体现了训练过程中的算子依赖关系。

---

## 4. 流量生成策略

每个 DAG 节点根据其通信类型调用不同的流量生成函数：

### 4.1 DP — Hypercube AllReduce

```python
def set_hypercube(host_list, msg_len, port, file_name):
    iter_times = log2(host_num)
    for iter in range(1, iter_times + 1):
        msg = msg_len // (2 ** iter)   # 每轮消息量减半
        for idx, host in enumerate(host_list):
            nei_idx = get_neighbor(idx, host_num, iter)
            # 生成 src → neighbor 的 RDMA 流
    result += result[::-1]             # Reduce-Scatter + AllGather 两轮
```

每个 phase 的消息量按 `msg_len / 2^iter` 递减，模拟 Butterfly / Hypercube AllReduce 的分阶段规约过程。

### 4.2 TP — Ring AllReduce

```python
def set_tensor_parallel(m, num_nodes, msg_len, num_phases, device, port, forward):
    for phase_idx in range(num_phases):
        start = floor((m + 0.5*device) * 2 * num_nodes)
        end   = floor(((m + 0.5*device)*2 + 1) * num_nodes)
        for i in range(start, end):
            dst_node = i + 1 if i < end-1 else start  # Ring 拓扑
```

按 device 偏移计算 Ring 的起始节点，`num_phases` 决定 Ring 传递轮数。

### 4.3 EP — All-to-All

```python
def set_all2all(host_list, msg_len, port, forward):
    for step in range(1, host_num):
        for idx, host_id_a in enumerate(host_list):
            idy = (step + idx) % host_num
            # src=host_id_a → dst=host_list[idy]
```

每步偏移产生一个 phase，共 `host_num - 1` 个 phase，覆盖所有点对通信。

### 4.4 PP — Pipeline Parallel

```python
def set_pipeline_parallel(node_pairs, msg_len, port, forward):
    for src_node, dst_node in node_pairs:
        # 单 phase，所有流并行启动
```

流水线并行为点对点传输，一个 phase 内所有阶段传递同时发出。

### 4.5 输出格式

每个 DAG 节点生成独立文件 `rdma_result/<node_name>/rdma_operateX.txt`：

```
stat rdma operate:
phase:3000
Type rdma_send src_node 0 src_port 1000 dst_node 8 dst_port 1000 priority 0 msg_len 4194304
Type rdma_send src_node 1 src_port 1000 dst_node 9 dst_port 1000 priority 0 msg_len 4194304
phase:3000
...
```

---

## 5. 依赖关系提取（TreeCut.py）

`TreeCut.py` 对 DAG 进行**树切割**，提取算子间的因果依赖：

### 5.1 毛刺节点识别与处理

"毛刺"节点（Burr Node）定义为叶节点中有兄弟节点的那些节点。`findBurr()` 识别后，`cutBurrNode()` 将其从主树切割，并记录到 `out_dependence_list`：

```python
def cutBurrNode(Tree, node):
    parents = get_parent(Tree, node)
    out_dependence_list.append((node, parents))  # 记录依赖边
    Tree.remove_node(node)
    # 将节点封装为独立子树加入 Tree_list
```

### 5.2 冲突检测

`oneNodeTreatment()` 检查目标节点的 `nodes_passed` 集合是否与同层其他节点有交集（即是否共享物理设备）。若有交集（冲突），则沿父节点路径合并，不可独立并行；若无交集，则安全切割为独立子任务。

### 5.3 输出

切割结果写入 `dependence.txt`，格式为：

```
depend A-B on C-D     # A到B号算子依赖C到D号算子
```

---

## 6. 仿真驱动执行

### 6.1 数据结构

```cpp
vector<vector<vector<FlowInfo>>> flowInfos;  // [operate][phase][flow]
vector<uint16_t> phaseCur;                   // 每个 operate 当前执行到第几个 phase
vector<vector<int>> opDependence;            // 算子依赖关系图
unordered_map<FlowKey, uint16_t> flowToPar;  // (src,dst,sport,dport) → operate_id
```

### 6.2 依赖驱动调度

```cpp
void checkDpd() {
    for (int i = 0; i < operateNum; i++) {
        if (opStart[i]) continue;
        bool flag = true;
        for (int j : opDependence[i]) {
            if (phaseCur[j] < flowInfos[j].size()) {
                flag = false; break;   // 前置算子未完成
            }
        }
        if (flag) {
            opStart[i] = true;
            sendPhase(i);              // 依赖满足，启动算子
        }
    }
}
```

### 6.3 Phase 推进

当一个 phase 内所有流完成时（`flowinput_cb` 回调统计），推进到下一个 phase，或触发 `checkDpd()` 检查后续算子是否可以解锁：

```cpp
void flowinput_cb(Ptr<OutputStreamWrapper> fout, Ptr<RdmaQueuePair> q) {
    // 找到完成流所属的 operate
    phaseCur[par]++;
    if (phaseCur[par] >= flowInfos[par].size()) {
        Simulator::Schedule(Seconds(0), checkDpd);   // operate 完成，检查依赖
    } else {
        Simulator::Schedule(Seconds(0), sendPhase, par);  // phase 推进
    }
}
```

---

## 7. 整体数据流

```
deepseek.txt
    ↓  constructTree()
  DAG (nx.DiGraph)
    ↓  topological_sort() + generate_traffic_from_tree()
rdma_result/DP1_0/rdma_operate.txt   (Hypercube AllReduce, N phases)
rdma_result/TP1_1/rdma_operate.txt   (Ring AllReduce, 7 phases)
rdma_result/EP1_2/rdma_operate.txt   (All-to-All, N-1 phases)
    ↓  TreeCut.py
dependence.txt   (算子依赖关系)
    ↓  reverie-evaluation-sigcomm2023.cc
仿真执行：依赖驱动 → Phase 推进 → 流完成回调 → 下一算子解锁
```

---

## 8. 关键设计优势

| 特性 | 说明 |
|------|------|
| **算子级精度** | 每个 DP/TP/EP/PP 算子独立建模，精确还原训练通信时序 |
| **依赖因果性** | DAG 拓扑序驱动，确保前置算子完成后才启动后续算子 |
| **多模型支持** | DeepSeek（MoE + EP）、Qwen（Dense + TP/DP）均支持，EP 在 DeepSeek 下自动映射为 TP 通信模式 |
| **Phase 内并发** | 同一 phase 内所有流并发发送，真实反映集合通信的批量特性 |
| **可扩展性** | 通过修改输入文件即可描述任意模型并行策略，无需修改仿真代码 |
