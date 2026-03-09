# Block Diffusion for Text-to-Shape 架构文档（当前实现）

最后更新：2026-03-09

本文档仅保留两个主线对照：

1. 原始 AR Text-to-Shape GPT（DualStream）
2. 当前官方同构 Block Diffusion Text-to-Shape（SingleStream DiT）

## 1. 原始 AR GPT（参考基线）

- 主干：`DualStreamRoformer`
- 训练目标：next-token prediction
- 推理方式：token-level 自回归
- 条件注入：cond 流 + shape 流双流注意力

## 2. 当前 Block Diffusion 主线（已替换）

- 主干：`BlockDiffusionDiT`（单流）
- 训练输入：`x_input=[x_t|x_0]`
- 训练掩码：`block_diff_mask`
- 参数化：`subs`
- 损失：`NLL * 1/t`
- 推理：semi-AR（按 block 从左到右）+ block 内 `t->s` 反向更新

### 2.1 条件注入（效果优先）

当前模型同时使用：

- **AdaLN 全局条件**：sigma + pooled text + bbox
- **Cross-Attn 条件 token**：每层 shape hidden 读取 text/bbox token

每个 block 顺序：

1. Self-Attention
2. Cross-Attention
3. MLP

三条分支都带可学习门控残差。

### 2.2 KV Cache 与 CFG

- 支持 KV cache（semi-AR 采样加速）
- 支持 CFG 推理（cond/uncond 双前向）
- 当 `cfg_scale != 1` 时，自动关闭 KV cache（当前实现不维护双分支 cache）

## 3. 与 Block Diffusion 原论文对齐情况

### 3.1 已对齐（核心机制）

- `[x_t|x_0]` + `block_diff_mask`
- `subs` 参数化
- `1/t` loss scaling
- semi-AR + first-hitting / ddpm-style block update

### 3.2 非 1:1 的部分

- text-to-shape 条件生成是扩展任务，不是原论文纯 LM 设定
- 条件注入（AdaLN+Cross-Attn）属于工程增强
- attention backend 受环境影响（flash_attn 不可用时回退 sdpa）

结论：
**无条件核心算法路径与论文/官方实现同构；条件分支是为任务效果加入的增强模块。**

## 4. 流程图（中文）

```mermaid
flowchart LR
  A[x0真值shape token] --> B[按t加噪得到xt]
  B --> C[拼接 x_input = xt | x0]
  D[text_hidden + bbox] --> E[DiT前向 bd_training mask]
  C --> E
  E --> F[restrict logits + subs]
  F --> G[NLL * 1/t 反向传播]
```

```mermaid
flowchart LR
  H[全MASK初始化] --> I[按block从左到右]
  I --> J[block内多步 t到s]
  J --> K[预测p(x0|xt,cond)]
  K --> L[按mask_prob更新并保留已解开token]
  L --> J
```
