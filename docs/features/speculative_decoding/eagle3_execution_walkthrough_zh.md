# vLLM Eagle3 执行逻辑详解

本文基于 vLLM V1 Engine 的 Eagle3 实现，使用一个完整的单请求案例说明：

- Target model 与 Eagle3 draft model 的关系；
- prefill 时 Drafter 的输入、输出和 token shift；
- decode 时 Target 如何验证 draft token；
- 全部接受、部分接受和全部拒绝后的下一轮 drafting；
- Target KV cache 与 Draft KV cache 的管理策略。

本文主要覆盖标准自回归 Eagle3。Parallel drafting、Mamba/GDN recurrent
state 等特殊路径只在相关章节简要说明。

## 1. 总体结构

Eagle3 包含两个逻辑模型：

1. **Target model**：完整模型，负责产生权威概率分布并验证候选 token。
2. **Draft model**：轻量 Eagle3 head，根据 Target 的中间层 feature
   快速预测未来 token。

一次生成循环可以概括为：

```text
Target forward
  ├─ 最终 hidden -> Target logits -> 正式采样 token
  └─ 多个中间层 hidden
       -> concat
       -> FC 投影
       -> Eagle3 Drafter
       -> K 个 draft tokens

下一轮 Target forward
  -> 并行验证 K 个 draft tokens
  -> 接受最长前缀
  -> 首次拒绝处产生 recovered token
  -> 或全部接受后产生 bonus token
  -> 使用本轮真实 Target hidden 生成下一批 draft
```

Eagle3 与 Eagle1 的核心差别是 Drafter 的 feature 输入：

```text
Eagle1:
  Target 最后一层 hidden

Eagle3:
  多个 Target 中间层 hidden
    -> concat
    -> norm/FC
    -> Draft hidden
```

验证、接受/拒绝和调度逻辑与其他 speculative decoding 方法共用。

## 2. 调度器中的关键状态

vLLM Scheduler 没有严格分离的 prefill/decode 状态机，而是维护：

```text
num_tokens_with_spec
  = prompt tokens + confirmed output tokens + speculative tokens

num_computed_tokens
  = 当前已有有效 Target KV 的 token 数
```

调度器不断让 `num_computed_tokens` 追上 `num_tokens_with_spec`。

必须区分：

```text
token 已经采样出来
≠
token 已经经过 Target forward 并拥有有效 KV
```

每轮最后产生的 recovered token 或 bonus token，通常要到下一轮才进入
Target model。

## 3. 完整案例与符号

假设 prompt token 为：

```text
token:    10  20  30  40  50
position:  0   1   2   3   4
```

配置：

```text
num_speculative_tokens = 2
```

令：

```text
H10 = Target 处理 token 10 后提供给 Eagle3 的融合 feature
H20 = Target 处理 token 20 后提供给 Eagle3 的融合 feature
...
H50 = Target 处理 token 50 后提供给 Eagle3 的融合 feature
```

这些 `H` 不是简单的最终层 hidden。典型 Eagle3 会从 Target 的多个层提取：

```text
A50 = concat(
    aux_layer_low(50),
    aux_layer_middle(50),
    aux_layer_high(50),
)

H50 = FC(A50)
```

如果 Target hidden size 为 4096，选取 3 个辅助层，则 FC 前后的形状可能是：

```text
concat auxiliary hidden: [num_tokens, 12288]
projected Eagle3 hidden:  [num_tokens, 4096]
```

默认辅助层通常为：

```text
(2, num_layers // 2, num_layers - 3)
```

模型配置也可以通过 `eagle_aux_hidden_state_layer_ids` 指定其他层。

## 4. Target prefill

Target 首先处理完整 prompt：

```text
input_ids: [10,20,30,40,50]
positions: [0, 1, 2, 3, 4]
```

Target 写入：

```text
Target KV position 0: token 10
Target KV position 1: token 20
Target KV position 2: token 30
Target KV position 3: token 40
Target KV position 4: token 50
```

最后位置的 Target logits 采样出第一个正式输出 token：

```text
Target logits after token 50 -> token 60
```

此时逻辑序列为：

```text
[10,20,30,40,50,60]
```

但是 `60` 尚未经过 Target forward：

```text
num_tokens = 6
num_computed_tokens = 5
Target 有效 KV positions = 0...4
```

## 5. Prefill 后如何构造 Draft 输入

Eagle3 对 `input_ids` 做一位 shift：

```text
原 Target token: [10, 20, 30, 40, 50]
Draft input_ids: [20, 30, 40, 50, 60]
Target hidden:   [H10,H20,H30,H40,H50]
positions:       [0,  1,  2,  3,  4]
```

这里只 shift `input_ids`：

- Target hidden 不 shift；
- positions 不 shift；
- 最后一个位置插入 Target 刚采样的 token `60`。

因此每个 Draft slot 的输入关系是：

| Draft slot | Target feature | Token embedding |
| --- | --- | --- |
| 0 | H10 | E20 |
| 1 | H20 | E30 |
| 2 | H30 | E40 |
| 3 | H40 | E50 |
| 4 | H50 | E60 |

这是 EAGLE feature autoregression 的核心设计，而不是 off-by-one 错误。

## 6. Shift 的精确语义

普通 Target 模型满足：

```text
token 10 -> Target feature H10 -> logits 预测 token 20
token 20 -> Target feature H20 -> logits 预测 token 30
```

EAGLE 先预测未来 feature：

```text
当前 feature H10
+ 已知的下一个 token embedding E20
    ↓
Draft Transformer
    ↓
预测下一个 feature Ĥ20
    ↓
Draft LM Head
    ↓
预测 token 30
```

可以写成：

```text
Ĥ(i+1) = DraftTransformer(H(i), Embedding(token(i+1)))

draft_logits(i+1) = DraftLMHead(Ĥ(i+1))
```

所以每个位置的输出语义是：

| Draft slot | 输入 | 预测 feature | Draft logits 预测 |
| --- | --- | --- | --- |
| 0 | H10 + E20 | Ĥ20 | token 30 |
| 1 | H20 + E30 | Ĥ30 | token 40 |
| 2 | H30 + E40 | Ĥ40 | token 50 |
| 3 | H40 + E50 | Ĥ50 | token 60 |
| 4 | H50 + E60 | Ĥ60 | 第一个 draft token |

关键点是：

```text
H50 + E60 不是用来预测 60。

60 已经由 Target 采样，是 Drafter 的已知条件。
H50 + E60 用于预测 Ĥ60，再由 Ĥ60 预测 60 后面的 token。
```

标准自回归 Eagle3 只在每个请求最后一个有效 slot 采样。前面的 slot
仍然参与 forward，用于填充 Draft KV cache 和建立完整 causal context。

## 7. Draft 模型内部

以 Llama Eagle3 为例，Draft model 首先查 embedding：

```text
E60 = DraftEmbedding(60)
```

第一层分别归一化 embedding 和 Target feature：

```text
e = RMSNorm(E60)
h = RMSNorm(H50)
```

然后拼接：

```text
Z4 = concat(e, h)
```

如果 hidden size 为 4096：

```text
E60: [4096]
H50: [4096]
Z4:  [8192]
```

因此第一层 QKV projection 的输入维度为 `2 * hidden_size`。后续 Draft
层只处理普通 `hidden_size`。

经过 Draft attention、MLP 和最终 norm，得到预测 feature：

```text
Ĥ60
```

再经过 Draft LM Head：

```text
draft_logits_0 = DraftLMHead(Ĥ60)
d0 = sample(draft_logits_0)
```

假设：

```text
d0 = 70
```

完整关系是：

```text
H50 + E60
  -> Ĥ60
  -> Draft LM Head
  -> token 70
```

## 8. Draft forward 的两个输出

Eagle3 Llama forward 返回：

```text
(last_hidden_states, aux_output)
```

二者用途不同。

### 8.1 `last_hidden_states`

这是最终 norm 后的 hidden，用于：

```text
LM Head -> draft logits -> draft token
```

例如：

```text
Ĥ60_postnorm -> logits -> token 70
```

### 8.2 `aux_output`

默认是最终 norm 前的 hidden。它保留下来，作为下一次 Draft
autoregressive forward 的 feature 输入：

```text
Ĥ60 + Embedding(70)
  -> 下一次 Draft forward
```

## 9. 生成第二个 Draft token

第一个 draft token 为：

```text
d0 = 70
```

为了生成第二个 draft token，Drafter 再运行一步：

```text
input_id:     70
hidden_state: Ĥ60
position:     5
```

计算：

```text
Ĥ60 + E70
  -> Ĥ70
  -> Draft LM Head
  -> d1
```

假设：

```text
d1 = 80
```

完整生成链：

```text
Target:
H50 -> Target LM Head -> 60

Draft step 0:
H50 + E60 -> Ĥ60 -> Draft LM Head -> 70

Draft step 1:
Ĥ60 + E70 -> Ĥ70 -> Draft LM Head -> 80
```

最终返回：

```text
draft_token_ids = [70,80]
```

## 10. Prefill 后的 Target KV 与 Draft KV

Target 与 Drafter 的 KV 物理 tensor 不共享：

```text
Target layer 0 KV
Target layer 1 KV
...
Draft layer 0 KV
```

在典型 uniform KV group 配置中，它们使用协调一致的：

```text
block IDs
block table
position -> slot_mapping
```

同一个逻辑 slot 会分别索引 Target 层和 Draft 层自己的 KV tensor。

例如：

```text
block_id = 5
block_size = 16
position = 4

slot = 5 * 16 + 4 = 84
```

那么：

```text
TargetKV[layer_i][slot=84]
DraftKV[layer_0][slot=84]
```

是两个不同物理 KV，只是逻辑 slot 编号相同。

### 10.1 Target KV

| Position | Target KV 内容 |
| --- | --- |
| 0 | token 10 在 Target attention 层产生的 K/V |
| 1 | token 20 的 Target K/V |
| 2 | token 30 的 Target K/V |
| 3 | token 40 的 Target K/V |
| 4 | token 50 的 Target K/V |

### 10.2 Draft KV

| Position | Draft KV 内容 |
| --- | --- |
| 0 | `H10 + E20` 产生的 Draft K/V |
| 1 | `H20 + E30` 产生的 Draft K/V |
| 2 | `H30 + E40` 产生的 Draft K/V |
| 3 | `H40 + E50` 产生的 Draft K/V |
| 4 | `H50 + E60` 产生的 Draft K/V |
| 5 | `Ĥ60 + E70` 产生的 Draft K/V |

注意：

```text
Target KV position 4:
  token 50 的 Target K/V

Draft KV position 4:
  H50 + E60 的 Draft K/V
```

位置相同，但内容和语义完全不同。

`80` 此时只是采样结果，还没有作为 Draft 输入，因此没有对应的新
Draft KV slot。

## 11. 第一次 decode：Target 验证 70 和 80

下一轮 Target 输入：

```text
Target input: [60,70,80]
position:     [5, 6, 7]
```

Target 一次 forward 写入：

```text
position 5: token 60 的 Target KV
position 6: token 70 的 Target KV
position 7: token 80 的 Target KV
```

并产生真实 Target feature：

```text
[H60,H70,H80]
```

三个位置的 logits 分别用于：

```text
position 5 logits -> 验证 draft 70
position 6 logits -> 验证 draft 80
position 7 logits -> 全部接受时产生 bonus token
```

## 12. Draft token 的接受比对逻辑

Target 验证输入为：

```text
input:    [60,70,80]
position: [5, 6, 7]
draft:       [70,80]
```

Target logits 与 draft 的对齐关系为：

```text
Target 在 input 60 后的 logits  <-> draft[0] = 70
Target 在 input 70 后的 logits  <-> draft[1] = 80
Target 在 input 80 后的 logits  <-> bonus token
```

因此不是用 position 6 的 logits 验证 position 6 的 token `70`，而是用
position 5 在处理完 `60` 后产生的 next-token 分布验证 `70`。

验证必须从第一个 draft token开始顺序执行，并且只能接受一个连续前缀：

```text
draft[0] 接受 -> 继续验证 draft[1]
draft[0] 拒绝 -> draft[1] 无条件失效
draft[1] 拒绝 -> 保留 draft[0]，停止验证后续 draft
```

### 12.1 Greedy 模式

Greedy 模式下，对每个位置计算 Target argmax：

```text
target_token = argmax(target_logits)
accepted = draft_token == target_token
```

#### 示例一：全部接受

```text
draft tokens:       [70,80]
Target argmax:      [70,80]
Target bonus token: 90
```

逐位置比较：

```text
draft[0] = 70, Target argmax = 70 -> 接受
draft[1] = 80, Target argmax = 80 -> 接受
```

两个 draft 全部接受，追加 Target 在最后位置采样的 bonus：

```text
RejectionSampler output = [70,80,90]
```

#### 示例二：接受一个、拒绝一个

```text
draft tokens:  [70,80]
Target argmax: [70,81]
```

逐位置比较：

```text
draft[0] = 70, Target argmax = 70 -> 接受
draft[1] = 80, Target argmax = 81 -> 拒绝
```

Greedy 模式下，首次不匹配位置的 Target argmax `81` 成为 recovered token：

```text
RejectionSampler tensor = [70,81,-1]
最终有效输出            = [70,81]
```

`-1` 是 placeholder，表示该位置之后的 draft 不再有效。

#### 示例三：第一个 token 就拒绝

```text
draft tokens:  [70,80]
Target argmax: [71,...]
```

因为第一个 token 不匹配：

```text
draft[0] = 70, Target argmax = 71 -> 拒绝
draft[1] = 80                    -> 不再验证
```

输出：

```text
RejectionSampler tensor = [71,-1,-1]
最终有效输出            = [71]
```

### 12.2 随机采样模式

随机模式不是简单比较 token 是否等于 Target argmax。

令当前 draft token 为 `d`：

```text
q(d) = Draft model 分配给 d 的概率
p(d) = Target model 分配给 d 的概率
u    = [0,1) 上的均匀随机数
```

接受条件：

```text
q(d) > 0
并且
p(d) / q(d) >= u
```

等价的接受概率为：

```text
P(accept d) = min(1, p(d) / q(d))
```

这意味着：

- `p(d) >= q(d)` 时，该 draft token 总是接受；
- `p(d) < q(d)` 时，以 `p(d) / q(d)` 的概率接受；
- 随机模式下，即使 draft token 不是 Target argmax，也可能被接受。

#### 随机接受的数值案例

仍然使用：

```text
draft tokens = [70,80]
```

验证第一个 draft `70`：

```text
q(70) = 0.50
p(70) = 0.40
u0    = 0.60

p(70) / q(70) = 0.40 / 0.50 = 0.80
0.80 >= 0.60 -> 接受 70
```

验证第二个 draft `80`：

```text
q(80) = 0.40
p(80) = 0.10
u1    = 0.70

p(80) / q(80) = 0.10 / 0.40 = 0.25
0.25 < 0.70 -> 拒绝 80
```

最终接受前缀为：

```text
[70]
```

### 12.3 拒绝后的 recovered token

随机模式拒绝 draft token 后，不能直接从原始 Target 分布 `p(x)` 重新采样，
否则会破坏 speculative decoding 与 Target 原始采样分布的一致性。

vLLM 从修正分布采样：

```text
p_recovered(x)
  = max(p(x) - q(x), 0)
    / sum_y max(p(y) - q(y), 0)
```

以上面拒绝 `80` 的位置为例，简化词表只考虑：

```text
token:       80    81    82
Draft q(x): 0.40  0.20  0.40
Target p(x):0.10  0.60  0.30
```

先计算：

```text
max(p-q, 0):

token 80: max(0.10 - 0.40, 0) = 0
token 81: max(0.60 - 0.20, 0) = 0.40
token 82: max(0.30 - 0.40, 0) = 0
```

归一化后：

```text
p_recovered(81) = 1
```

所以 recovered token 为：

```text
81
```

RejectionSampler 输出：

```text
[70,81,-1]
```

最终有效输出：

```text
[70,81]
```

### 12.4 全部接受后的 bonus token

只有全部 draft token 都接受时，才使用 bonus token：

```text
accepted drafts = [70,80]
bonus token      = Target 在 input 80 后采样的 token 90

output = [70,80,90]
```

如果任意 draft 被拒绝，预先计算的 bonus token 会被丢弃，首次拒绝位置改用
recovered token。

### 12.5 接受结果如何驱动下一轮

令：

```text
K = draft token 数
A = accepted draft token 数
```

则：

```text
num_rejected = K - A
num_computed_tokens -= num_rejected
```

下一轮 Drafter 使用 RejectionSampler 输出中的最后一个有效 token：

```text
全部接受:
  output = [70,80,90]
  next_token = 90
  可用真实 hidden = [H60,H70,H80]

接受一个:
  output = [70,81]
  next_token = 81
  可用真实 hidden = [H60,H70]

全部拒绝:
  output = [71]
  next_token = 71
  可用真实 hidden = [H60]
```

对应实现主要位于：

```text
vllm/v1/sample/rejection_sampler.py
vllm/v1/spec_decode/utils.py
vllm/v1/core/sched/scheduler.py
```

## 13. 情况一：全部接受

假设 Target 结果：

```text
position 5 argmax = 70
position 6 argmax = 80
position 7 sample = 90
```

验证结果：

```text
70 accepted
80 accepted
bonus = 90
```

RejectionSampler 输出：

```text
[70,80,90]
```

计数：

```text
num_draft_tokens = 2
valid_sampled_tokens_count = 3
num_accepted = 2
num_rejected = 0
```

### 13.1 Target KV 状态

Target KV 中：

```text
position 5: 60，有效
position 6: 70，有效
position 7: 80，有效
```

`90` 是 bonus token，尚未经过 Target：

```text
正式序列:
[10,20,30,40,50,60,70,80,90]

num_tokens = 9
num_computed_tokens = 8
Target 有效 KV positions = 0...7
```

### 13.2 下一轮 Draft 输入

Target 本轮提供：

```text
target_token_ids:     [60,70,80]
target_hidden_states: [H60,H70,H80]
next_token:           90
positions:            [5,6,7]
```

Shift 后：

```text
原 Target token: [60,70,80]
Draft input_ids: [70,80,90]
Target hidden:   [H60,H70,H80]
positions:       [5, 6, 7]
```

对应关系：

| Draft position | 输入 | 预测 feature | logits 预测 |
| --- | --- | --- | --- |
| 5 | H60 + E70 | Ĥ70 | token 80 |
| 6 | H70 + E80 | Ĥ80 | token 90 |
| 7 | H80 + E90 | Ĥ90 | 新 draft `d0'` |

只在最后位置采样。假设：

```text
H80 + E90 -> Ĥ90 -> d0'=100
```

第二个新 draft：

```text
Ĥ90 + E100 -> Ĥ100 -> d1'=110
```

得到：

```text
new drafts = [100,110]
```

下一轮 Target 验证输入：

```text
[90,100,110]
positions = [8,9,10]
```

### 13.3 全接受时 Draft KV 刷新

Prefill 后 Draft position 5 原本使用预测 feature：

```text
position 5:
  Ĥ60 + E70
```

Target 验证后已经得到真实 `H60`，所以这一位置被刷新为：

```text
position 5:
  H60 + E70
```

后续为：

```text
position 6:
  H70 + E80

position 7:
  H80 + E90

position 8:
  Ĥ90 + E100
```

规律是：

```text
Target 已计算过的位置 -> 使用真实 H 刷新 Draft KV
Target 尚未计算的位置 -> 使用 Drafter 预测的 Ĥ
```

## 14. 情况二：接受 70，拒绝 80

假设：

```text
position 5 argmax = 70
position 6 argmax = 81
```

验证：

```text
70 == 70 -> 接受
80 != 81 -> 拒绝
```

首次拒绝后不再继续接受，拒绝位置产生：

```text
recovered = 81
```

RejectionSampler 的逻辑输出：

```text
[70,81,-1]
```

过滤 placeholder 后的正式输出：

```text
[70,81]
```

计数：

```text
num_draft_tokens = 2
valid_sampled_tokens_count = 2
num_accepted = 1
num_rejected = 1
next_token = 81
```

### 14.1 Target KV 状态

Target 已经物理写入：

```text
position 5: token 60
position 6: token 70
position 7: token 80
```

但 position 7 的 `80` 已被拒绝：

```text
position 5: 有效
position 6: 有效
position 7: stale/逻辑无效
```

正式序列：

```text
[10,20,30,40,50,60,70,81]
```

`81` 尚未经过 Target：

```text
num_tokens = 8
num_computed_tokens = 7
```

下一轮 Target 从 position 7 开始处理 `81`，覆盖原来被拒绝的 `80` KV。

### 14.2 下一轮 Draft 输入

本轮 Target 虽然计算了：

```text
Target token:  [60,70,80]
Target hidden: [H60,H70,H80]
```

但有效前缀只包含：

```text
[60,70]
```

因此逻辑上的 Draft 输入是：

```text
有效 Target token: [60,70]
Draft input_ids:    [70,81]
Target hidden:      [H60,H70]
positions:          [5,6]
```

对应：

| Draft position | 输入 | 预测 feature | logits 预测 |
| --- | --- | --- | --- |
| 5 | H60 + E70 | Ĥ70 | token 81 |
| 6 | H70 + E81 | Ĥ81 | 新 draft `d0'` |

假设：

```text
H70 + E81 -> Ĥ81 -> d0'=82
```

继续自回归：

```text
Ĥ81 + E82 -> Ĥ82 -> d1'=83
```

得到：

```text
new drafts = [82,83]
```

下一轮 Target 输入：

```text
[81,82,83]
positions = [7,8,9]
```

### 14.3 部分接受后的 Draft KV

逻辑有效的 Draft KV 更新为：

```text
position 5:
  H60 + E70

position 6:
  H70 + E81

position 7:
  Ĥ81 + E82
```

原来与 rejected token `80` 相关的 speculative 状态不再属于有效上下文。

## 15. 情况三：第一个 Draft 即被拒绝

假设 Target 在 position 5 产生：

```text
recovered = 71
```

输出：

```text
[71,-1,-1]
```

计数：

```text
num_accepted = 0
num_rejected = 2
next_token = 71
```

有效的 Target hidden 只有：

```text
H60
```

新的 Draft 输入：

```text
Draft input_id: [71]
Target hidden:  [H60]
position:       [5]
```

生成：

```text
H60 + E71 -> Ĥ71 -> d0'
Ĥ71 + E(d0') -> Ĥ(d0') -> d1'
```

下一轮 Target 输入：

```text
[71,d0',d1']
```

并从 position 6 开始覆盖原来 draft `70`、`80` 对应的无效 Target KV。

## 16. 三种结果对照

| 验证结果 | 正式输出 | 下一轮可用的真实 Target hidden | 下一轮条件 token |
| --- | --- | --- | --- |
| 全接受 | `[70,80,90]` | `[H60,H70,H80]` | `90` |
| 接受 70、拒绝 80 | `[70,81]` | `[H60,H70]` | `81` |
| 70 即拒绝 | `[71]` | `[H60]` | `71` |

Shift 后分别是：

```text
全接受:
[60,70,80] + next=90
  -> [70,80,90] + [H60,H70,H80]

接受一个:
[60,70] + next=81
  -> [70,81] + [H60,H70]

全部拒绝:
[60] + next=71
  -> [71] + [H60]
```

## 17. Target KV 的回滚策略

Target 在验证时会乐观地为全部 query token 写 KV。拒绝后不会：

- 清零 KV tensor；
- 删除单个 KV entry；
- 立即收缩 block table。

Scheduler 只回退逻辑计数：

```text
num_computed_tokens -= num_rejected
```

例如接受 70、拒绝 80：

```text
验证前 num_computed_tokens = 5
乐观处理 [60,70,80] 后     = 8
拒绝 1 个 draft 后         = 7
```

于是 position 7 上的 `80` KV 变成 stale 数据。下一轮 `81` 使用同一逻辑
position 写入时会覆盖它。

因此 Target KV 策略是：

```text
物理 KV：不回滚
逻辑长度：回滚
stale slot：后续覆盖
```

Prefix caching 只提交 finalized token，不会把可被拒绝的 draft token
作为可复用 prefix。

## 18. Draft KV 的拒绝处理

Draft 输入准备有两种主要路径。

### 18.1 Non-padded 路径

直接裁掉 rejected 尾部。

例如：

```text
验证输入 [60,70,80]
接受 70、拒绝 80

裁剪为:
[60,70]

再 shift:
[70,81]
```

同时缩短：

```text
query length
seq length
slot mapping
```

### 18.2 Padded 路径

为了维持 CUDA Graph 和 batch shape，张量中可能仍保留 rejected 对应位置，
但：

- `token_indices_to_sample` 回退到最后一个有效位置；
- rejected 尾部不会成为采样位置；
- 后续 autoregressive drafting 前执行：

```text
seq_lens -= num_rejected_tokens
```

因此 stale Draft KV 不会进入后续有效 attention 上下文，之后会被覆盖。

额外扩展槽和 parallel drafting 路径还会使用：

```text
PADDING_SLOT_ID = -1
```

阻止无效 slot 写入 KV。不能简单认为标准 padded Eagle 路径中的所有
rejected slot 都必然设置为 `-1`。

## 19. Chunked prefill

如果当前只完成了中间 prefill chunk：

- Target 正常处理该 chunk；
- Drafter 仍运行 first pass，同步自己的 KV；
- 当前产生的 draft token 不进入下一次 Target 验证；
- Scheduler 会在 `request.is_prefill_chunk=True` 时忽略这些 draft。

只有完成最后一个 prefill chunk 后，生成的 draft token 才被保留并进入
首次 decode 验证。

## 20. Draft K 步的控制边界与调用链

### 20.1 结论先行

标准自回归 Eagle3 的 K 步循环不由 Scheduler 或 ModelRunner 逐步驱动。

ModelRunner 在一个 engine step 中只调用一次：

```python
self.drafter.propose(...)
```

这一次调用会在 Proposer 内部同步生成完整的：

```text
[batch_size, K]
```

中间的 `K-1` 个自回归步骤不会返回 Scheduler，也不会产生新的调度决策。

更准确地说，`EagleProposer` 是一个很薄的包装类，实际 K 步实现位于其父类
`SpecDecodeBaseProposer.propose()`：

```text
vllm/v1/spec_decode/eagle.py
  -> EagleProposer

vllm/v1/spec_decode/llm_base_proposer.py
  -> SpecDecodeBaseProposer.propose()
  -> for token_index in range(K - 1)
```

因此，“K 步循环在 EagleProposer 内”描述的是逻辑归属；具体 Python
循环代码在 `llm_base_proposer.py` 的基类实现中。

### 20.2 必须区分的两层循环

推测解码有两层不同粒度的循环：

| 层次 | 循环内容 | 控制者 | 一次迭代 |
| --- | --- | --- | --- |
| 外层 | 验证上轮 draft，再生成下轮 draft | Scheduler + ModelRunner | 一个 engine step |
| 内层 | 连续生成 K 个 draft token | Proposer | 一次 Draft forward |

这里“内层一次迭代等于一次 Draft forward”是指：

```text
Step 0:
  first pass -> 产生 d0

后续每次 for 迭代:
  输入上一个 draft token
  -> 一次单 token Draft forward
  -> 产生下一个 draft token
```

生成 K 个 token 的总调用数量是：

```text
1 次 first pass + (K - 1) 次单 token Draft forward
```

### 20.3 完整外层调用链

一个 engine step 的主链路可以表示为：

```text
EngineCore.step()
  │
  ├─ scheduler.schedule()
  │    ├─ 取出 request.spec_token_ids
  │    ├─ 形成 scheduled_spec_decode_tokens
  │    ├─ 分配 Target 本轮 query 所需 KV slots
  │    └─ 为 Eagle 保留 lookahead 空间
  │
  ├─ model_executor.execute_model()
  │    └─ ModelRunner.execute_model()
  │         ├─ 准备 [anchor, draft_0, ..., draft_K-1]
  │         ├─ Target 一次 forward 并行验证
  │         └─ 保存 logits、hidden 和 auxiliary hidden
  │
  ├─ model_executor.sample_tokens()
  │    └─ ModelRunner.sample_tokens()
  │         ├─ RejectionSampler 接受/拒绝
  │         ├─ 整理有效 token 和 rejected 数量
  │         └─ propose_draft_token_ids()
  │              └─ self.drafter.propose(...)  # 每个分支只调用一次
  │                   ├─ first pass -> d0
  │                   ├─ for token_index in range(K - 1)
  │                   └─ 返回 [batch_size, K]
  │
  ├─ scheduler.update_from_output()
  │    ├─ 追加 accepted/recovered/bonus token
  │    └─ 按 rejected 数量回退 num_computed_tokens
  │
  └─ EngineCore.post_step()
       ├─ take_draft_token_ids()
       └─ scheduler.update_draft_token_ids()
            └─ request.spec_token_ids = 新一轮 draft
```

新产生的 draft 在当前 engine step 中不会再交给 Target。它们被保存在
`request.spec_token_ids`，供下一个 engine step 验证。

### 20.4 Scheduler 实际控制什么

Scheduler 不执行 Draft 的每一个自回归步骤，但提供外层框架。

#### 选择 K

固定配置下：

```text
K = speculative_config.num_speculative_tokens
```

如果配置 dynamic speculative decoding，则 Scheduler 根据当前 scheduled
batch 的请求数量，通过 lookup table 选择本轮 K：

```text
K = dynamic_sd_lookup[num_scheduled_requests]
```

这个 K 通过：

```text
SchedulerOutput.num_spec_tokens_to_schedule
```

传给 ModelRunner，再传入一次 `drafter.propose()`。

#### 预留 KV 空间

Eagle 初始化时：

```text
num_lookahead_tokens = num_spec_tokens
```

Scheduler 在 KV block 分配时预留 lookahead slots，使 Drafter 的内部
自回归步骤能够安全计算后续 position 和 slot mapping。

#### 组织下一轮 Target 验证

Drafter 返回：

```text
[d0,d1,...,d(K-1)]
```

Scheduler 将其保存在：

```text
request.spec_token_ids
```

下一轮构造：

```text
scheduled_spec_decode_tokens
```

Target 才会一次性验证这些 token。

#### 处理接受/拒绝记账

Target 验证后，Scheduler 计算：

```text
num_rejected = num_draft_tokens - num_accepted
num_computed_tokens -= num_rejected
```

这是 request/engine-step 粒度的状态更新，不会介入 Proposer 内部 K 步。

### 20.5 ModelRunner 实际控制什么

ModelRunner 是一个 engine step 的执行器，负责：

1. 根据 SchedulerOutput 组织 Target 输入；
2. 运行一次 Target forward；
3. 调用 RejectionSampler；
4. 准备 Drafter 的 shifted token、Target hidden 和 attention metadata；
5. 调用一次 `self.drafter.propose(...)`；
6. 接收完整 `[batch_size, K]` 结果。

ModelRunner 看不到 Proposer 中间生成的：

```text
d0
d1
...
d(K-2)
```

它只能在 `propose()` 返回后拿到完整结果。

### 20.6 Proposer 内部如何生成 K 个 token

以 K=4 为例，prefill 后已有：

```text
Target 输出: 60
最后真实 feature: H50
```

#### First pass

```text
H50 + E60
  -> Ĥ60
  -> d0=70
```

此时：

```text
draft_token_ids_list = [70]
```

#### `for` 第 1 次迭代

```text
input_id = 70
hidden   = Ĥ60
position = 5

Ĥ60 + E70
  -> Ĥ70
  -> d1=80
```

结果：

```text
[70,80]
```

#### `for` 第 2 次迭代

```text
input_id = 80
hidden   = Ĥ70
position = 6

Ĥ70 + E80
  -> Ĥ80
  -> d2=90
```

结果：

```text
[70,80,90]
```

#### `for` 第 3 次迭代

```text
input_id = 90
hidden   = Ĥ80
position = 7

Ĥ80 + E90
  -> Ĥ90
  -> d3=100
```

最终：

```text
draft_token_ids = [70,80,90,100]
shape = [batch_size, 4]
```

### 20.7 每个内部步骤更新哪些状态

每次 `for` 迭代由 Proposer 自己更新：

```text
input_ids      = 上一步 draft token
hidden_states  = 上一步 Draft 输出 feature
positions      = positions + 1
seq_lens       = seq_lens + 1
slot_mapping   = 从 block table 计算新 slot
draft_index    = token_index + 1
```

随后：

```text
构建当前 Draft attention metadata
-> Draft model forward
-> 采样 token
-> 追加到 draft_token_ids_list
```

位置和 slot mapping 的增量更新使用：

```text
eagle_step_update_slot_mapping_and_metadata()
```

如果上一轮存在 rejected token，进入 K 步循环前还会执行：

```text
seq_lens -= num_rejected_tokens
```

使自回归从当前请求的有效边界继续。

### 20.8 Batch 内是并行的

虽然 K 维度是串行的，但每一个内部 Draft step 会同时处理 batch 中的所有
请求。

假设：

```text
batch_size = 8
K = 4
```

执行方式是：

```text
first pass:
  8 个请求一起产生 d0

for iteration 0:
  8 个请求一起输入各自 d0，产生各自 d1

for iteration 1:
  8 个请求一起输入各自 d1，产生各自 d2

for iteration 2:
  8 个请求一起输入各自 d2，产生各自 d3
```

不是：

```text
先给请求 0 跑完 K 步
再给请求 1 跑完 K 步
```

所以：

```text
K 维度串行
batch 维度并行
```

### 20.9 K=0、K=1 和 parallel drafting

#### K=0

Dynamic speculative decoding 可能令 K=0。Drafter first pass 仍然执行，以
保持 Draft KV 与 Target 进度同步，但返回：

```text
shape = [batch_size, 0]
```

#### K=1

First pass 产生 `d0` 后直接返回，不进入 `for` 循环。

#### Parallel drafting

`parallel_drafting=True` 时也不会进入标准 `K-1` 循环。它通过扩展后的
输入和 mask slots，在一次 Draft forward 中生成多个候选。

因此：

```text
标准 Eagle3:
  1 次 first pass + K-1 次单 token Draft forward

Parallel Eagle3:
  1 次扩展后的 Draft forward 产生 K 个候选
```

### 20.10 常见误区

错误理解：

```text
K 个 draft token
= K 个 engine step
= Scheduler 调度 K 次
```

正确理解：

```text
一个 engine step:
  1 次 Target verify
  + Proposer 内部同步生成下一轮 K 个 draft

下一个 engine step:
  Target 才验证这 K 个 draft
```

传统 decode 生成 N 个 token 通常需要 N 个 Target/engine step。推测解码
则可能在一次 Target verify 后推进最多 `K+1` 个 token：

```text
K 个 accepted draft + 1 个 bonus
```

节省的是昂贵的 Target forward 和 engine step 数量；Drafter 内部仍要
承担 K 步 feature autoregression，只是 Draft 模型远小于 Target。

一句话总结：

```text
外层“猜测 -> 验证”跨 step 循环由 Scheduler + ModelRunner 驱动；
内层连续生成 K 个 token 的循环由 Proposer 在一次 propose() 中跑完。
Scheduler 只提供 K、KV 空间、下一轮 verify 组织和接受/拒绝记账。
```

## 21. Target 与 Draft 的关系总结

| 项目 | Target model | Eagle3 Draft model |
| --- | --- | --- |
| 作用 | 产生权威概率并验证 | 快速产生候选 token |
| 输入 | token IDs | shifted token IDs + Target feature |
| feature | 真实 Target hidden | 预测的未来 feature |
| logits | 决定最终输出 | 只产生候选 |
| KV tensor | Target 各 attention 层独立 KV | Draft 各 attention 层独立 KV |
| block layout | 与 Draft 协调的逻辑布局 | 与 Target position 对齐 |
| 拒绝后 | 回退逻辑长度，stale KV 后续覆盖 | trim/pad/seq_len 回退 |

最核心的闭环是：

```text
Target 提供真实历史 feature H
  ↓
Drafter 使用 H + 下一 token embedding
预测未来 feature Ĥ 和 draft token
  ↓
Target 批量验证 draft
  ↓
验证得到真实的新 H
  ↓
使用真实 H 刷新 Draft KV
```

## 22. 特殊情况

上述“不物理回滚”主要描述标准 Transformer attention KV。

Mamba、GDN 等 recurrent state 不能简单依靠 attention seq_len 忽略尾部，
因此可能在接受/拒绝后执行 state block 拷贝或移位。这属于 recurrent
state 修正，不是普通 attention KV entry 删除。

## 23. 关键源码索引

| 主题 | 文件 |
| --- | --- |
| Scheduler speculative token 调度与回退 | `vllm/v1/core/sched/scheduler.py` |
| Target forward、hidden 收集和 Drafter 入口 | `vllm/v1/worker/gpu_model_runner.py` |
| Eagle shift、first pass 和多步 drafting | `vllm/v1/spec_decode/llm_base_proposer.py` |
| Eagle3 Llama draft 模型 | `vllm/model_executor/models/llama_eagle3.py` |
| Target Eagle3 auxiliary hidden 协议 | `vllm/model_executor/models/interfaces.py` |
| 接受、拒绝、recovered 和 bonus | `vllm/v1/sample/rejection_sampler.py` |
| rejected 数量与 slot mapping 工具 | `vllm/v1/spec_decode/utils.py` |
| KV block 分配和 prefix cache 提交 | `vllm/v1/core/kv_cache_manager.py` |

建议按照以下顺序阅读：

```text
scheduler.py
  -> gpu_model_runner.py
  -> llm_base_proposer.py
  -> llama_eagle3.py
  -> rejection_sampler.py
  -> utils.py
  -> kv_cache_manager.py
```
