# Single-Stream DiT 主线实现说明（官方同构 + 条件增强）

最后更新：2026-03-09

该文档记录当前项目中 Block Diffusion 的唯一主线实现：

- 核心路径对齐 BD3-LM 官方 DiT
- 条件路径采用 AdaLN + Cross-Attn 混合注入

## 1. 入口文件

- 模型：`cube3d/model/gpt/block_diffusion_dit.py`
- 训练：`cube3d/train/runners/train_block_diffusion_t2s.py`
- 推理：`cube3d/inference/engine_block_diffusion.py`
- 配置：`cube3d/configs/block_diffusion_t2s.yaml`

## 2. 核心结构

### 2.1 DiT block

每层结构：

1. Self-Attn（支持官方掩码语义）
2. Cross-Attn（读取 text+bbox token）
3. MLP

并使用 AdaLN 调制与门控残差。

### 2.2 掩码与模式

- `attention_mode="bd_training"`：`[x_t|x_0]` + `block_diff_mask`
- `attention_mode="block_causal"`：block 级因果采样
- `sample_mode/store_kv`：用于 semi-AR 推理缓存

### 2.3 条件构建

- 全局条件：`sigma + pooled(text) + bbox`
- token 条件：`text_token + bbox_token`
- 无条件分支（CFG）下自动处理空 mask（注入 learnable null token）

## 3. 训练对接

训练输入：`x_input=[x_t|x_0]`

训练目标：

- restrict 到 `[codes + MASK]`
- `subs_parameterization`
- `mean(NLL * 1/t)`

支持：

- `cond_drop_prob`（CFG 训练）
- `var_min` clipping 搜索

## 4. 推理对接

推理流程：

- semi-AR（按 block 左到右）
- block 内 `t->s` 反向更新
- 可选 `first_hitting`
- CFG 推理（`cfg_scale`）

KV cache 规则：

- `cfg_scale==1`：可启用 KV cache
- `cfg_scale!=1`：自动回退 no-cache（当前实现不维护 cond/uncond 双 cache）

## 5. 对齐说明

- 与官方同构：训练/采样核心机制
- 与官方不同：text-to-shape 条件化增强（AdaLN+Cross-Attn）

因此结论是：

**核心 BD3 算法同构，条件路径是任务扩展。**
