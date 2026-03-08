# Block Diffusion for Text-to-Shape 架构文档（原始 GPT / 当前实现 / 论文实现）

本文档只保留三个模型：

1. 原始 Text-to-Shape GPT（AR）
2. 当前代码实现（已按 Block Diffusion 原论文关键机制改造）
3. Block Diffusion 原论文实现（BD3-LM）

并明确说明三者关系、复用点、差异点。

---

## 1. 原始 Text-to-Shape GPT（AR）

### 1.1 输入输出

- 输入：文本 prompt -> CLIP hidden -> `text_proj`（可加 bbox）得到条件 `cond`
- 输出：长度 `L=1024` 的 shape token 序列 `shape_ids`

### 1.2 模型结构

- 主干：`DualStreamRoformer`（decoder-only）
- 条件与形状序列等价于 `[cond | shape]` 拼接后做注意力
- 注意力掩码：严格 token-causal（下三角）
- 推理：逐 token 采样（AR）
- 训练目标：NTP（next-token CE）

代码：

- `cube3d/model/gpt/dual_stream_roformer.py`
- `cube3d/inference/engine.py`

---

## 2. 当前实现（Block Diffusion 对齐版）

这部分是本次实际落地后的代码行为，不是旧版“简化置信度解 mask”。

### 2.1 主干复用与新增模块

- 复用：
  - `DualStreamRoformer` 的参数与大部分计算图
  - 文本编码与 shape tokenizer 管线
- 新增：
  - `[MASK]` token（词表 +1，`wte/lm_head` 同步扩容）
  - Block Diffusion 两种掩码：
    1. `block_causal`（采样）
    2. `bd_training`（训练时 `x_t ⊕ x_0`）

代码：

- `cube3d/model/gpt/block_diffusion_roformer.py`
- `cube3d/model/gpt/dual_stream_roformer.py`（新增可注入 mask/is_causal 的 forward 参数）

### 2.2 训练路径（对齐论文核心）

当前训练采用：

1. 按 block 采样 `t`（loglinear，默认 `t ~ U[eps_min, eps_max]`，支持 antithetic）
2. 生成 `x_t`：每个 token 以概率 `t` 变成 `[MASK]`
3. 构造模型输入 `x_input = [x_t | x_0]`（长度 `2L`）
4. 用 `bd_training` 掩码前向
5. 只取前半段 logits（对应 `x_t` 位置），限制到 `[codebook tokens + MASK]`
6. 用 `subs` 参数化（未 mask 位置强制 one-hot，不产生损失）
7. 损失：`NLL * (1/t)`（loglinear 对齐）
8. 支持 clipped schedule 搜索（`var_min` + `clip_search_widths`）

代码：

- `cube3d/train/runners/train_block_diffusion_t2s.py`
- `cube3d/train/noise/bd3_schedule.py`

### 2.3 采样路径（替换旧版置信度解 mask）

当前推理改为论文式更新：

1. 全序列初始化为 `[MASK]`
2. 外层按 block 从左到右
3. 内层多步 `t -> s`：
   - 计算 `mask_prob = move_chance(s)/move_chance(t)`（loglinear 下约为 `s/t`）
   - 前向得到 `p(x_0 | x_t, prefix)`
   - 对 masked 位置做反向更新：
     - 以 `mask_prob` 保持 MASK
     - 以 `1-mask_prob` 采样真实 token
   - carry-over unmasking：已解开的 token 不再回到 MASK
4. 支持 `first_hitting` 开关

注意：旧版“按置信度 top-k 逐轮填充”已移除。

代码：

- `cube3d/inference/engine_block_diffusion.py`

### 2.4 当前实现流程图（中文）

```mermaid
flowchart TD
  A[文本条件 cond] --> B[采样 t 并构造 x_t]
  B --> C[拼接输入 x_t | x_0]
  C --> D[BlockDiffusionRoformer 前向\nbd_training 掩码]
  D --> E[前半段 logits + subs 参数化]
  E --> F[加权 NLL: 1/t]
  F --> G[反向传播]

  H[推理全 MASK] --> I[按 block 从左到右]
  I --> J[block 内 t->s 反向更新]
  J --> K[mask_prob 更新 + carry-over]
  K --> I
```

---

## 3. Block Diffusion 原论文实现（BD3-LM）

论文核心点：

1. Block 级 AR：`p(x)=sum_b p(x^b|x^{<b})`
2. Block 内离散 diffusion（吸收态 `[MASK]`）
3. `x_t ⊕ x_0` 训练与专用 attention mask
4. `subs` 参数化
5. loglinear 下 `1/t` 级别的损失加权
6. clipped schedules + 方差最小化搜索
7. 采样时按反向转移（非置信度启发式）

---

## 4. 当前实现 vs 论文实现：对齐程度

### 4.1 已对齐

- `x_t ⊕ x_0` 训练输入
- BD3-LM 训练掩码规则（block diagonal + offset block causal + block causal）
- `subs` 参数化
- loglinear 语义下的 `1/t` 加权
- 采样反向更新 `mask_prob = p(s)/p(t)` + carry-over unmasking
- block-causal 采样掩码
- clipped schedule 搜索入口（`var_min`）

### 4.2 仍有工程差异（与官方仓库不完全同构）

- **骨架（backbone）不同**：论文官方仓库默认是**单流 DiT**（`models/dit.py::DIT`），而本项目保留你原来的**双流 DualStreamRoformer**（`cube3d/model/gpt/dual_stream_roformer.py::DualStreamRoformer`）。
- **注意力后端不同**：官方支持 `flash_attn/flex` 并为 BD3-LM 生成专用 mask；本项目当前以 `sdpa + bool mask` 为主，优先保证“能跑通 + 易集成”。
- **采样加速路径不同**：官方有更完整的 `kv_cache` / sliding-window（semi-AR）路径；本项目目前采样每步会重复计算较多（能用，但更慢）。
- **训练框架不同**：官方是 Lightning + metrics/方差搜索等配套；本项目是你现有 runner 体系（因此有些统计与搜索实现方式更简化）。

结论：本项目已经把**Block Diffusion 的关键算法机制**（训练输入、mask、`subs`、loglinear 加权、反向更新采样）对齐到可训练可采样；但与官方“逐行同构”的差别主要来自**骨架与工程栈**。

### 4.3 两种骨架本质上有什么不同？（单流 DiT vs 双流 DualStreamRoformer）

这里的“骨架”不是指“AR 还是 Diffusion”，而是指**同一套 Block Diffusion 训练/采样机制下面，负责输出 logits 的神经网络结构**。

#### 4.3.1 官方单流 DiT（BD3-LM repo 的 `DIT`）

**一句话**：把 `x_input` 当成**一条序列**做 Transformer，自注意力的“可看见关系”完全由 `block_diff_mask` 控制。

- **输入形态**：`indices = x_input`，训练时通常是 `x_input = [x_t | x_0]`（长度 `2L`）。
- **隐状态**：只有一个张量 `x ∈ R^{B×(2L)×D}`，每层都在这个张量上做 self-attention + MLP。
- **mask 的位置**：mask 直接作用在这条序列的 self-attention 上（训练用 `block_diff_mask`；采样时会裁剪到 block-causal 子矩阵）。
- **位置编码处理（很关键）**：官方实现中，`x_t` 半段和 `x_0` 半段会使用**同一套位置**（都按 `0..L-1` 做 RoPE），它把两半当作“同一序列的两份视图”，不是“长度翻倍的连续位置”。（见 `/tmp/bd3lms_repo_20260308b/models/dit.py::DDiTBlock.forward` 里对 `x[:,:n]` 与 `x[:,n:]` 分别做 RoPE 后再 concat 的逻辑）
- **条件注入方式**：DiT 骨架原生支持 `sigma/t` 这类“全局条件”通过 AdaLN/调制进入每层；BD3-LM 默认 `time_conditioning: False`，但官方仍会把 `sigma` 置零后走同一路径，等价于一个可学习的常量调制（不会随 `t` 变化）。

工程含义：
- forward API 很直接：`logits = backbone(indices, sigma)`。
- 易做高性能：mask 在单流结构里更容易配合 `flash_attn/flex` 与 KV-cache。

#### 4.3.2 本项目双流 DualStreamRoformer（`DualStreamRoformer`）

**一句话**：把条件 `cond` 和形状 token 序列 `shape` 作为**两条流**处理；形状流在注意力里读取条件流信息，从而实现 text-to-shape 条件生成。

- **输入形态**：
  - `cond ∈ R^{B×S×D}`：来自文本/框等条件（例如 CLIP hidden 经 `text_proj`）。
  - `embed ∈ R^{B×L×D}`：shape token 经 `wte` 的 embedding（AR 时 `L` 是当前已生成长度；BD 训练时 `L=2L0` 对应 `[x_t|x_0]`）。
- **双流层（dual blocks）的注意力计算方式**：
  - 会分别对 `cond` 和 `shape` 做 QKV 投影，然后把 `K,V` 拼起来让 shape 查询能够 attend 到 `[cond + shape]`。
  - cond 查询是否参与更新取决于 `cond_pre_only`（最后一层通常不更新 cond），且在我们提供的 mask 下 cond 永远看不到 shape（`_compose_cond_shape_mask` 里 `cond -> shape` 全是 False）。
  - 因此信息流大体是：`cond -> shape`（强），“shape -> cond”（默认没有）。
- **单流层（single blocks）**：只在 shape 序列上再做若干层 self-attention（mask 可自定义）。
- **mask 的位置**：本项目为了兼容双流 + 单流两段结构，需要同时提供：
  - dual blocks 用的 `full_mask`（作用在 `[cond | shape]` 上）
  - single blocks 用的 `shape_mask`（作用在 shape 上）
  - 这就是为什么实现上看起来“不像官方那样只有一个 block_diff_mask”。
- **位置编码处理（当前与官方不同的一点）**：
  - 目前 `DualStreamRoformer` 对 shape 序列是按 `0..(len-1)` 连续 RoPE。
  - 当训练输入是 `[x_t|x_0]`（长度 `2L`）时，这会让 `x_0` 半段的位置是 `L..2L-1`，而不是官方那种“与 `x_t` 共用 `0..L-1`”。
  - 这不影响“算法可运行”，但会导致与官方实现的 inductive bias 不同，可能带来性能/收敛差异。

工程含义：
- **复用收益**：你原来的 text-to-shape GPT 权重、数据管线、cond 组织方式都能继续用；改 Block Diffusion 时不需要把整个项目改成“token id 语言模型”风格。
- **改造代价**：要对齐官方的某些细节（例如 RoPE 的位置复用、KV-cache 的半自回归采样）需要在双流架构里额外打补丁，工程复杂度更高。

#### 4.3.3 关键差异汇总（你最关心的“本质/工程差别”）

|维度|官方单流 DiT（BD3-LM）|本项目 DualStreamRoformer（双流）|直接影响|
|---|---|---|---|
|序列形态|单序列 `x ∈ R^{B×T×D}`|两序列：`cond` 与 `shape`（dual+single 两段）|实现 mask 与 cache 的复杂度|
|条件方式|主要是 `sigma`（可 AdaLN），无外部 cond 序列|外部 `cond` 序列显式输入，shape attend cond|更适合 text-to-shape，但与官方不同构|
|BD3-LM mask|一个 `block_diff_mask` 作用在同一条序列|dual mask + single mask 拼装得到等效 shape-shape 可见性|能对齐连边关系，但实现更绕|
|RoPE（xt/x0）|两半共享同一位置 `0..L-1`|当前两半用连续位置 `0..2L-1`|可能影响收敛与性能（需要额外对齐）|
|采样加速|半 AR + KV cache + sliding window 更自然|现有 decode cache 不直接适配 block mask 模式|可跑但更慢，需要补齐|
|注意力后端|flash/flex 深度适配|主要 sdpa + bool mask|速度/显存占用差异|

### 4.4 不更改骨架，能把论文“关键点”对齐到什么程度？

先给结论：**Block Diffusion 的关键点大多数是“训练/采样机制”，不是“必须用 DiT 才成立”**。只要你的 backbone 能输出每个位置的 logits，并且能接收相同的 attention 可见性约束，就能在算法层面对齐。

#### 4.4.1 不换骨架也能严格对齐的部分（算法层）

- `x_input = [x_t | x_0]` 的训练输入构造
- 论文的 `block_diff_mask`（训练）与 block-causal mask（采样）
- `subs` 参数化（只在 `[MASK]` 位置产生有效损失）
- loglinear 语义下的 `1/t` 加权（以及 `var_min`/clipped schedule 的策略思想）
- 采样时的反向更新：`mask_prob = move_chance(s)/move_chance(t)` + carry-over unmasking

这些点我们之所以能在双流骨架里对齐，是因为它们本质上只依赖两件事：
1) **你对 `x_t/x_0` token id 的构造方式**，以及  
2) **你对 attention 图的约束方式（mask）**。

#### 4.4.2 不换骨架也能对齐，但需要额外工程的部分（实现细节/性能）

- **RoPE 位置对齐**：把 `[x_t|x_0]` 的两半位置编码改为共享 `0..L-1`（官方做法），以减少与论文/官方的结构性差异。
- **semi-AR 采样加速**：在 block-causal 采样时引入 KV-cache / sliding-window，避免每个 `t->s` 都全序列重算。
- **attention 后端**：如果要追求官方速度，需要把 mask 路径迁移到 `flash_attn` 或 `flex attention` 的 block mask。

这些都不要求“换成单流 DiT”，但确实会让双流实现变复杂。

#### 4.4.3 不换骨架很难做到“官方代码同构”的部分（骨架层）

- 官方 DiT block 的 AdaLN/门控/残差缩放/初始化等细节（即便 `time_conditioning=False` 也会形成不同的参数化与优化特性）。
- 官方对 BD3-LM 的一些高度工程化优化（Lightning 统计管线、一次验证 pass 收集所有 interval 统计等）。

这类差异通常不影响“原理正确性”，但会影响**吞吐、收敛速度、最终指标**，也是为什么我们说“当前实现是论文算法对齐，但骨架不与官方同构”。
