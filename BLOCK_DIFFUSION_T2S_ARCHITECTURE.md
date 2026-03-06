# Block Diffusion for Text-to-Shape 架构文档（对比 AR Text-to-Shape GPT）

本文档面向“没有训练大模型经验”的读者，用工程视角说明：

- **原始 Text-to-Shape GPT（AR，自回归）** 是怎样做推理的、模型由哪些模块组成
- **Block Diffusion Text-to-Shape（BD）** 的模型/推理/训练与 AR 的差异
- 哪些模块被**复用**，哪些模块是**新增/改动**并且需要（重新）训练

> 代码对应本仓库：
> - AR 推理引擎：`cube3d/inference/engine.py`
> - BD 推理引擎：`cube3d/inference/engine_block_diffusion.py`
> - AR 主干模型：`cube3d/model/gpt/dual_stream_roformer.py`
> - BD 主干模型：`cube3d/model/gpt/block_diffusion_roformer.py`
> - BD 训练：`cube3d/train/runners/train_block_diffusion_t2s.py`
> - BD 噪声/掩码：`cube3d/train/noise/masked_schedule.py`

---

## 0. 一句话总结：AR vs Block Diffusion 的“生成步数”差别

- **AR（原版）**：严格逐 token 生成，`L=1024` 就要 **1024 次 forward**（KV cache 只能加速，但步数不变）。
- **Block Diffusion（本仓库实现）**：把 1024 个 token 划成 block（例如 `block_size=32`），
  - 外层按 block 从左到右生成（**block 级 AR**）
  - 内层对一个 block 做多轮“去噪/解 mask”（**block 内 diffusion**）
  - forward 次数约为：`num_blocks * num_denoise_steps = (1024/32) * 8 = 256`（步数明显减少）

注意：这里的 “diffusion” 指 **离散 token 的 mask 去噪**（不是连续高斯噪声）。

## 1. 任务与表示：模型最终生成什么？

Text-to-Shape 的“输出”不是 mesh 文件，而是一串离散的 **shape token**：

- 序列长度固定：`L = 1024`
- 每个 token 是一个整数：`0 ~ 16383`（对应 shape tokenizer 的 codebook id）
- 拿到这 1024 个 token 后，通过 shape tokenizer 的解码器还原为 3D 网格（mesh）

你可以把它理解成：“模型先生成 1024 个‘积木编号’，再用解码器把编号拼回 3D 物体”。

---

## 2. 原始 Text-to-Shape GPT（AR）模型架构

### 2.1 模块分解（从输入到输出）

**(A) 文本条件编码（CLIP）**

- 模块：`CLIPTextModelWithProjection` + `CLIPTokenizerFast`
- 产物：`text_hidden`（形状一般是 `[B, 77, 768]`，来自 `openai/clip-vit-large-patch14`）
- 作用：把 prompt 变成“条件信息”，供 shape token 生成参考

> 对应：`cube3d/inference/engine.py::Engine.run_clip()`

**(B) 条件投影（把 CLIP 维度变成 GPT 维度）**

- 模块：`DualStreamRoformer.text_proj`（线性层）
- 形状变化：`[B, 77, 768] -> [B, 77, 1536]`
- 可选 bbox：`bbox_proj: [B, 3] -> [B, 1, 1536]` 并拼到 cond 后面

> 对应：`cube3d/model/gpt/dual_stream_roformer.py::encode_text()` + `bbox_proj`

**(C) shape token 嵌入（Embedding）**

- 模块：`transformer.wte`
- 输入：token id（整数序列）
- 输出：token embedding（形状 `[B, L, 1536]`）

> 对应：`cube3d/model/gpt/dual_stream_roformer.py::encode_token()`

**(D) Transformer 主干（Dual Stream + Single Stream）**

模型是一个 RoFormer 解码结构（带 RoPE），由两段组成：

- Dual-stream blocks：`n_layer=23`  
  输入两条流：shape 流 `x`（长度 1024）和 cond 流 `c`（长度 77/78）
- Single-stream blocks：`n_single_layer=1`  
  只处理 shape 流 `x`

> 对应：`cube3d/model/gpt/dual_stream_roformer.py::forward()`

**(E) 输出头（lm_head）**

- 模块：`lm_head`
- 输出：每个位置对词表的 logits（形状 `[B, L, vocab]`）

**(F) Shape tokenizer（解码到 mesh）**

- 模块：`OneDAutoEncoder`
- 用途：把 `shape_ids[1024]` 解码成几何表示，最终导出 mesh

> 对应：`cube3d/inference/engine.py::Engine.run_shape_decode()`

### 2.2 AR 推理算法（“像 GPT 一样逐 token 生成”）

AR 推理的核心规则：**一次只生成 1 个 token，并且必须按顺序从左到右**。

在 `Engine.run_gpt()` 里，它循环 `i=0..1023`：

1. 模型输入：`cond + 已经生成的前 i 个 token`
2. 模型输出：第 `i` 个位置（或 next-token）的 logits
3. 选择一个 token（`argmax` 或 `top_p` 采样）
4. 把这个 token 追加到序列末尾

此外还有两点常见工程增强（本仓库也有）：

- **KV cache**：加速逐 token 推理（`use_kv_cache`）
- **CFG（classifier-free guidance）**：同时算“有条件”和“无条件”，再线性组合 logits（`guidance_scale`）

> 对应：`cube3d/inference/engine.py::Engine.run_gpt()` 和 `EngineFast`

### 2.3 训练目标（AR 常见做法）

AR GPT 的经典训练目标是 **next-token 交叉熵**（teacher forcing）：

- 输入：把真实 token 右移一位（最前面放一个 `BOS`），让模型看到“到当前位置为止的真值历史”
- 预测：对每个位置输出一个分布，去预测当前位置的真实 token

在工程上，常见做法是用 `BOS` 开头，让模型学会“如何从空开始生成 1024 个 code”。

### 2.4 DualStreamRoformer 结构详解（Dual Stream + RoFormer + RoPE）

这一节专门把 `DualStreamRoformer` 讲清楚，因为 Block Diffusion 版本 **完全复用** 了它的 Transformer 主干；你一旦理解这里，就能同时理解 AR 与 BD 两条路径。

先给你一个“抓手”：

- 模型有两条流：**shape 流** `x` 和 **条件流** `c`
- DualStream 的核心思想是：在注意力里让 `x` 能读到 `c`（条件注入），同时保持 **自回归因果性**（不能看未来）
- 代码实现上，它不是写一个“显式 cross-attn 模块”，而是把注意力当成在 `[c | x]` 上做的 **一次 causal self-attention**（但用两套投影/归一化实现“dual stream”）

#### 2.4.1 模型有哪些子模块（从 `__init__()` 直接对应）

文件：`cube3d/model/gpt/dual_stream_roformer.py`

- 条件投影（把 CLIP 输出变成 GPT 维度）：
  - `text_proj: Linear(text_dim -> D)`，本仓库常用 `768 -> 1536`
  - `bbox_proj: Linear(3 -> D)`（`use_bbox=true` 时存在）
- shape codebook 对齐投影（用于初始化 embedding，不是每步 forward 都用）：
  - `shape_proj: Linear(shape_embed_dim -> D)`，本仓库常用 `32 -> 1536`
- 词表与特殊 token：
  - base vocab：`shape_model_vocab_size`（例如 16384 个 code）
  - 追加 3 个特殊 token：`shape_bos_id`, `shape_eos_id`, `padding_id`
  - Block Diffusion 子类会再追加 1 个：`[MASK]`（见 `BlockDiffusionRoformer.add_mask_token()`）
- 变换器主干：
  - `dual_blocks`：`n_layer` 个 `DualStreamDecoderLayerWithRotaryEmbedding`
  - `single_blocks`：`n_single_layer` 个 `DecoderLayerWithRotaryEmbedding`
- 输出头：
  - `ln_f` + `lm_head`

#### 2.4.2 forward() 的输入输出形状（你可以把它当成“条件前缀的 GPT”）

接口：`DualStreamRoformer.forward(embed, cond, kv_cache=None, curr_pos_id=None, decode=False)`

- `embed`: shape 流 embedding，`[B, L, D]`
- `cond`: 条件流 embedding，`[B, S, D]`（`S=77` 或 `78`）
- 输出 `logits`: `[B, L, vocab]`

关键点：**cond 不是 shape token**。shape token 是模型要生成的序列；cond 是“给模型参考的条件向量序列”。

#### 2.4.3 shape token 从哪来？`encode_token()` 在干什么？

`encode_token(tokens)`（见 `DualStreamRoformer.encode_token`）只有一句：

- `return self.transformer.wte(tokens)`

也就是说它只是“查表”：

- 输入：shape token id（整数）`tokens: [B, L]`
- 输出：shape token embedding（向量）`[B, L, D]`

shape token id 的来源：

- **训练时**：来自 shape tokenizer（`OneDAutoEncoder`）把 3D 形状量化后的 `shape_ids`
- **推理时**：来自模型采样出来的 id（AR 一次一个；BD 逐步把 `[MASK]` 填满）

#### 2.4.4 为什么 `wte`/`lm_head` 的前 16384 行会“像 shape codebook”？

因为推理/训练初始化时会做一次对齐（AR 与 BD 都这样做）：

1. 从 shape tokenizer 取 codebook：`codebook = shape_model.bottleneck.block.get_codebook()`，形状 `[num_codes, shape_embed_dim]`
2. 用 `shape_proj` 投影到 GPT 维度：`[num_codes, D]`
3. 拷贝到：
   - `wte.weight[:num_codes]`
   - `lm_head.weight[:num_codes]`

对应位置：

- AR 推理：`cube3d/inference/engine.py` 初始化里 copy codebook
- BD 训练：`cube3d/train/runners/train_block_diffusion_t2s.py` 初始化里 copy codebook

这样做的直觉是：shape token id `k` 的 embedding 一开始就落在 tokenizer codebook 的第 `k` 个向量附近，训练更稳。

#### 2.4.5 Dual-stream attention：到底是不是“cross-attn + self-attn”？

文件：`cube3d/model/transformers/dual_stream_attention.py`

每个 dual block 里，注意力部分是 `DualStreamAttentionWithRotaryEmbedding`。它的关键做法是：

先把一层 `DualStreamDecoderLayerWithRotaryEmbedding` 的“零件”列出来（见同文件 `DualStreamDecoderLayerWithRotaryEmbedding.__init__()`）：

- `ln_1`：对 shape 流做 LayerNorm
- `ln_2`：对 cond 流做 LayerNorm
- `attn`：`DualStreamAttentionWithRotaryEmbedding`
- `post_1`：`DismantledPostAttention`（更新 shape 流）
- `post_2`：`DismantledPostAttention`（更新 cond 流；当 `cond_pre_only=True` 时不会创建）

再看 `attn` 内部（见 `DualStreamAttentionWithRotaryEmbedding.__init__()`）：

- `pre_x = DismantledPreAttention(query=True)`：对 shape 流产生 `(Qx, Kx, Vx)`
- `pre_c = DismantledPreAttention(query=not cond_pre_only)`：
  - 绝大多数层 `cond_pre_only=False`：对 cond 流产生 `(Qc, Kc, Vc)`
  - 最后一层 `cond_pre_only=True`：对 cond 流**只产生** `(Kc, Vc)`（没有 `Qc`）

它们的张量形状（省略 batch）大致是：

- `Q, K, V`: `[B, n_head, T, head_dim]`
- 其中 `head_dim = D / n_head`

然后进入关键的“拼起来做一次注意力”：

1. 对 cond 流、shape 流分别做投影（各自一套 pre-attention 线性层 + RMSNorm）：
   - shape：总是产生 `Qx, Kx, Vx`
   - cond：通常产生 `Qc, Kc, Vc`
2. 在“序列维度”把它们拼起来当成一个整体做注意力：
   - `K = [Kc; Kx]`
   - `V = [Vc; Vx]`
   - `Q` 分两种：
     - 大多数层：`Q = [Qc; Qx]`（cond 和 shape 都会被更新）
     - 最后一层（`cond_pre_only=True`）：`Q = Qx`（cond **只提供 K/V，不作为 query**）

所以对 shape 流来说，它的一次注意力就同时做了两件事：

- 读 `Kc/Vc`：等价于 **cross-attention 到文本条件**
- 读 `Kx/Vx`（尤其是左侧已生成部分）：等价于 **shape 的 self-attention**

并不是“写了两段不同的 attention”，而是一次 attention 里同时发生。

注意力的底层算子在 `cube3d/model/transformers/rope.py::scaled_dot_product_attention_with_rotary_emb()`：

- 会先对 `Q/K` 应用 RoPE（旋转位置编码）
- 再调用 PyTorch 的 `scaled_dot_product_attention`（在 CUDA 上通常会走 FlashAttention/SDPA kernel）

最后，`DismantledPostAttention` 做的就是标准 Transformer 残差结构（见同文件）：

- `x = x + Linear(attn_out)`
- `x = x + MLP(LN(x))`（MLP 是 `SwiGLUMLP`）

如果这一层是最后一层并且 `cond_pre_only=True`：

- cond 不作为 query，因此注意力输出里没有 `a_c`
- 这一层返回时 `c` 会被置为 `None`（在 `DualStreamDecoderLayerWithRotaryEmbedding.forward()` 里）
- 之后进入 single-stream blocks，只更新 shape 流，不再更新 cond

#### 2.4.6 为什么 cond 看不到 shape，但 shape 能看 cond（因果 mask 的位置约定）

在 `DualStreamRoformer.forward()` 里会构造一个下三角 mask（注意是对 `[cond | shape]` 的联合长度）：

- `attn_mask = tril(ones(S+L, S+L))`

并且注意 token 顺序始终是：

- 左边：`cond`（长度 `S`）
- 右边：`shape`（长度 `L`）

这会带来两个非常重要的性质：

- cond token 在左侧，因此它的“可见区域”只包含 cond 自身（不会看到右侧 shape）→ **cond 不会被生成的 shape 反向污染**
- shape token 在右侧，因此它能看到 cond 全部 token + 自己左边的 shape token → **条件信息永远可用**

你可以用一个极小的例子理解（`[c0,c1 | x0,x1]`）：

- `x1` 可以看 `c0,c1,x0,x1`
- `c1` 只能看 `c0,c1`

这就是“把 cond 当作前缀”的 prefix-LM 写法。

#### 2.4.7 RoPE（旋转位置编码）是怎么塞进去的

同一个 forward 里预计算了两套 RoPE：

- dual-stream 用：`d_freqs_cis`（长度 `S+L`）
- single-stream 用：`s_freqs_cis`（长度 `L`）

实现上有一个容易忽略的细节：cond 的 position id 被置为 0（全 0），shape 的 position id 是 `0..L-1`。

直觉解释：

- 文本条件 `cond` 更像“一组条件向量”，不强调它们在 GPT 内部的绝对顺序
- shape 序列才是强序列结构，需要明确的位置编码

#### 2.4.8 KV cache / decode：为什么能像 GPT 那样逐 token 生成

当 `kv_cache` 启用时（见 `DualStreamRoformer.init_kv_cache()` 与 `DualStreamAttentionWithRotaryEmbedding.forward()`）：

- **prefill** 阶段：把 `[cond | 已有 shape 前缀]` 的 K/V 写进 cache
- **decode** 阶段：每次只输入 1 个新 token 的 embedding（shape 流），只算这 1 个 token 的 Q/K/V，然后把 K/V 追加进 cache

因此即便 decode 时注意力代码不再显式处理 cond 的 Q（甚至最后一层 cond 不当 query），cond 依然通过“cache 里已有的 K/V 前缀”持续影响后续 shape token。

#### 2.4.9 single-stream blocks 是干什么的？

dual blocks 结束后，还会跑 `n_single_layer` 个普通 RoFormer decoder 层（见 `cube3d/model/transformers/roformer.py::DecoderLayerWithRotaryEmbedding`）：

- 只对 shape 流做一次额外的自回归 self-attn + MLP
- 可以理解为“最后再精修一下 shape 序列内部一致性”（不再更新 cond）

### 2.5 AR 推理流程图（中文）

```mermaid
flowchart TD
  A[文本 prompt] --> B[CLIP tokenizer + text encoder]
  B --> C[text_hidden: 77x768]
  C --> D[text_proj(+bbox_proj) -> cond: 77/78x1536]
  D --> E[DualStreamRoformer]
  E --> F[逐 token 循环 i=0..1023]
  F --> G[lm_head logits -> 采样 next_id]
  G --> H[拼接到历史 tokens]
  H --> F
  F --> I[得到 shape_ids: 1024]
  I --> J[OneDAutoEncoder decode_indices]
  J --> K[extract_geometry -> mesh]
```

---

## 3. Block Diffusion Text-to-Shape（BD）模型架构

### 3.1 高层思想：把“逐 token”变成“逐 block”

把长度 `L=1024` 的序列切成连续 block：

- `block_size = 32`（可配置）
- `num_blocks = 1024 / 32 = 32`
- 第 `k` 个 block 覆盖位置：`[k*block_size, (k+1)*block_size)`

BD 的采样/推理由两层循环构成：

1. **外层（block 级 AR）**：block 从左到右依次生成（前面的 block 生成完就固定住）。
2. **内层（block 内 diffusion）**：在一个 block 内，先放满 `[MASK]`，再做多轮“去噪”，每轮填一部分 token。

这就是“AR 和 diffusion 的插值”：

- block 越小、每个 block 的去噪步数越少，行为越像 AR；
- block 越大、去噪步数越多，行为越像 diffusion（一次填更多 token）。

### 3.2 模型本体：尽量复用 AR GPT，只补一个 `[MASK]`

本仓库当前实现的核心策略是：

- **不改 Transformer 结构**：继续用 `DualStreamRoformer` 的 forward（同样的 RoPE、同样的 causal mask）。
- **只扩展词表 +1**：新增一个离散 token id 作为 `[MASK]`（噪声符号）。

对应代码：

- 模型包装：`cube3d/model/gpt/block_diffusion_roformer.py::BlockDiffusionRoformer`
- 扩展 API：`add_mask_token()` / `ensure_mask_token()`

### 3.3 `[MASK]` token 具体怎么加（以及为什么能复用 AR 权重）

`BlockDiffusionRoformer.add_mask_token()` 做了两件事：

1. `wte: Embedding[vocab, dim]` 增加一行（`vocab += 1`）
2. `lm_head: Linear[dim -> vocab]` 同样增加一行

新增这 1 行的初始化默认是“拷贝 padding 行”（也可以高斯随机）：

- 好处：可以直接加载 **AR checkpoint**（不含 mask），再补一行即可继续训练；
- 风险：`[MASK]` 这行一开始没有语义，需要通过 BD 训练学出来。

### 3.4 BD 训练方法（本仓库当前实现）

训练数据本身不变：每条样本就是 `(text_hidden, shape_ids)`。

训练时动态构造 “带噪声的输入”：

1. 随机选一个 block `b ~ Uniform({0..num_blocks-1})`
2. 从 schedule 采样一个 mask 比例 `r ~ U[beta_low, beta_high]`
3. 在这个 block 内随机选 `round(block_size * r)` 个位置，把 token 替换成 `[MASK]`
4. 喂给模型，要求它在这些 masked 位置预测回原 token

对应代码：

- schedule：`cube3d/train/noise/masked_schedule.py::ClippedMaskSchedule`
- 加噪：`cube3d/train/noise/masked_schedule.py::mask_one_block_per_sample`
- 训练 runner：`cube3d/train/runners/train_block_diffusion_t2s.py`

损失（实现等价表达）：

- 只在 masked 位置计算交叉熵：`CE(logits_masked, target_masked)`
- 用 `w(r) = 1 / r` 做权重（让“更难的高噪声样本”更重要）

此外，为了兼容 CFG（有条件/无条件），训练里还做了 **CFG dropout**：

- 以 `cfg_drop_prob` 的概率把整条样本的 `text_hidden` 置 0
- 使模型学到“没有文本条件时也能建模一个先验”

### 3.5 BD 推理算法（本仓库当前实现：逐 block + 逐步解 mask）

推理时初始化整条序列为 `[MASK]`：

- `shape_ids = [MASK] * 1024`

然后对每个 block `k = 0..num_blocks-1`：

1. 当前 block 的 32 个位置一开始全是 `[MASK]`
2. 重复 `num_denoise_steps` 轮（例如 8 轮）：
   - 前向一次得到 logits（当前序列里，已生成的 token + 仍为 `[MASK]` 的 token）
   - 在当前 block 内，对每个位置算置信度 `max softmax(logits)`
   - 采样一个候选 token（argmax 或 top-p）
   - 只挑选其中 **一部分** mask 位置（高置信度优先）填进去
3. block 结束若仍有 `[MASK]`，用 argmax 兜底补齐

对应代码：`cube3d/inference/engine_block_diffusion.py::EngineBlockDiffusion.run_gpt()`

### 3.6 BD 推理流程图（中文）

```mermaid
flowchart TD
  A[文本 prompt] --> B[CLIP tokenizer + text encoder]
  B --> C[text_hidden: 77x768]
  C --> D[text_proj(+bbox_proj) -> cond]
  D --> E[初始化 shape_ids 全为 MASK]
  E --> F{block_idx=0..num_blocks-1}
  F --> G[当前 block: start..end]
  G --> H{denoise_step=1..num_denoise_steps}
  H --> I[前向预测 logits]
  I --> J[对当前 block 选高置信度位置]
  J --> K[把部分 MASK 替换为 token]
  K --> H
  H --> L[block 填满后进入下一个 block]
  L --> F
  F --> M[得到 shape_ids: 1024]
  M --> N[OneDAutoEncoder 解码 -> mesh]
```

---

## 4. 对比：哪些复用？哪些新增/改动？哪些需要重新训练？

下面按“模块/代码层面”做对比（以本仓库实现为准）。

| 组件 | AR Text-to-Shape GPT | Block Diffusion T2S | 是否复用 | 是否需要（重新）训练 |
|---|---|---|---|---|
| 文本编码器（CLIP） | `CLIPTextModelWithProjection` | 同 AR | 复用 | **不训练**（固定；离线/在线都可） |
| Shape tokenizer（VQ/AE） | `OneDAutoEncoder` | 同 AR | 复用 | **不训练**（固定；只负责 encode/decode） |
| Transformer 主干 | `DualStreamRoformer` | 同结构（继承） | 复用 | **建议继续训练/微调**（目标从 next-token 变为去噪） |
| 条件投影 `text_proj`/`bbox_proj` | 有 | 同 AR | 复用 | 一般随主干一起微调即可 |
| `wte` / `lm_head`（codebook 行） | 有 | 同 AR | 复用 | 可从 AR 权重初始化；继续训练会更稳 |
| 新增 `[MASK]` 行 | 无 | `+1` vocab 行 | **新增** | **必须训练**（否则模型不会“理解 mask”） |
| 训练目标 | next-token CE | masked 去噪 CE（只算 masked 位） | 改动 | 需要训练（否则不会收敛到新目标） |
| 采样算法 | 逐 token | 逐 block + 去噪解 mask | 改动 | 不需要训练（是推理策略） |

结论（最工程实用的说法）：

- 如果你已有 AR 的 `shape_gpt.safetensors`：**可以当作 BD 的初始化**，只需要补一行 `[MASK]` 然后继续训练。
- 但想让 BD 真正“会用 mask 去噪”并且采样质量不崩：**必须至少训练 `[MASK]` 行，并建议微调整个主干**。
- CLIP/shape tokenizer 不需要训练，能省大量计算与数据。

---

## 5. “AR 和 diffusion 到底体现在哪里？”（回答你可能会卡住的点）

### 5.1 为什么 block 开头都是 `[MASK]`，AR 仍然有意义？

直觉上你会觉得：“既然一个 block 一开始全是 mask，那不是 AR 在‘生成 mask’吗？”

关键点：**AR 的意义不在于“生成了什么符号”，而在于“生成顺序和可用上下文”**。

在 BD 推理时：

- 当我们正在生成第 `k` 个 block：
  - `0..k-1` 的 block 已经是确定的真实 token（模型可以利用）
  - `k..end` 的未来 block 仍是 `[MASK]`（模型看不到未来信息）
- 模型本身依然是 **causal attention**（见 `DualStreamRoformer.forward()` 里的下三角 mask），所以预测永远遵守“只能看左边”。

因此这就是 **block 级的自回归**：每个 block 的生成都条件于前面 block 的结果。

### 5.2 那 “diffusion” 在 block 内到底是什么？

这里的 diffusion 是“离散 token 的 mask 去噪”：

- 用 `[MASK]` 表示“噪声/未知”
- 每一轮 denoise：模型给出所有位置的分布，然后我们把一部分 mask 替换成具体 token
- 重复多轮直到填满

它和“连续高斯扩散”不一样，但思想一致：**从高噪声（全 mask）逐步走向低噪声（全 token）**。

### 5.3 这是不是就不再像 GPT 那样逐 token？

是的：**token 级的外层循环被替换成了 block 级循环**。

但注意：我们的主干仍是一个“GPT 式的 causal decoder”，只是我们改变了：

- 训练目标（从 next-token 变成 masked 去噪）
- 推理策略（从逐 token 采样变成逐 block 去噪）

---

## 6. 读代码入口（建议按这个顺序）

1. AR 推理：`cube3d/inference/engine.py`（`run_gpt()` 的逐 token 循环）
2. BD 推理：`cube3d/inference/engine_block_diffusion.py`（`run_gpt()` 的双层循环）
3. 主干模型：`cube3d/model/gpt/dual_stream_roformer.py`（causal mask + dual/single blocks）
4. mask 扩展：`cube3d/model/gpt/block_diffusion_roformer.py`（只做 vocab+1）
5. 训练 runner：`cube3d/train/runners/train_block_diffusion_t2s.py`（mask 一个 block、算 masked CE）
6. 噪声调度：`cube3d/train/noise/masked_schedule.py`
