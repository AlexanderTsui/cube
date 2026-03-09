# Block Diffusion for Text-to-Shape 训练实现文档（官方同构主线）

最后更新：2026-03-09

## 1. 当前实现范围

本仓库的 Block Diffusion 训练已切换为**单流 DiT 官方同构主线**：

- 训练入口：`cube3d/train/runners/train_block_diffusion_t2s.py`
- 主干模型：`cube3d/model/gpt/block_diffusion_dit.py`
- 噪声/参数化：`cube3d/train/noise/bd3_schedule.py`

说明：Block Diffusion 训练不再走 dual-stream 路径；AR 的原始 dual-stream 推理代码仍保留在独立链路中。

## 2. 训练数据接口

`BlockDiffusionDataset` 当前读取：

- `shape_ids`：`[L]`
- `text_hidden`：`[S,768]`
- `text_attention_mask`：`[S]`
- `bbox_xyz`：`[3]`

训练样本组织与之前一致，不需要改数据格式。

## 3. 训练算法（当前代码）

每个 step 执行：

1. 采样 blockwise `t`（loglinear 区间 `[eps_min, eps_max]`）
2. 按 `move_chance=t` 构造 `x_t`
3. 训练输入 `x_input=[x_t|x_0]`
4. DiT 在 `attention_mode="bd_training"` 前向
5. 限制词表到 `[codebook + MASK]`
6. `subs_parameterization`
7. 损失 `mean(NLL * 1/t)`

并支持：

- `var_min` clipped schedule 搜索
- 训练 CFG drop（`cond_drop_prob`，把 text/bbox 置无条件）

## 4. 模型结构（训练侧关键）

当前 DiT 每层包含：

- self-attn（支持 BD training mask）
- cross-attn（读取 text/bbox 条件 token）
- MLP
- AdaLN 全局调制（sigma + pooled text + bbox）

即 **AdaLN + Cross-Attn 混合注入**，并带门控残差。

## 5. 推理相关配置（与训练耦合）

配置文件：`cube3d/configs/block_diffusion_t2s.yaml`

关键项：

- `diffusion.block_size`
- `diffusion.num_denoise_steps`
- `diffusion.first_hitting`
- `diffusion.kv_cache`
- `diffusion.cfg_scale`
- `diffusion.cond_drop_prob`
- `diffusion.model.attn_backend`（首选 `flash_attn`，不可用自动回退 `sdpa`）

## 6. 与论文对齐结论

已对齐核心机制：

- `[x_t|x_0]` 训练输入
- `block_diff_mask` 训练掩码语义
- `subs` 参数化
- `1/t` 加权目标
- semi-AR + block 内 `t->s` 更新

当前主要差异：

- 任务为 text-to-shape 条件生成（论文主任务是纯 LM）
- 条件注入采用 AdaLN + Cross-Attn 混合扩展
- 注意力后端受环境影响（可能回退 SDPA）

结论：
**核心 BD3 算法路径已同构，条件分支是任务扩展。**
