# Block Diffusion 论文对齐改造记录（全过程 + 结果 + 训练指南）

日期：2026-03-08  
项目目录：`/root/cube`  
目标：尽量严格对齐 Block Diffusion 原论文（BD3-LM）在训练与采样上的核心实现。

---

## 1. 参考来源

### 1.1 论文与官方代码

- 论文：Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models（ICLR 2025）
- 官方代码仓库：`https://github.com/kuleshov-group/bd3lms`
- 本地对照仓库（本次拉取）：`/tmp/bd3lms_repo_20260308b`
- 对照 commit：`1c3e8f4`

### 1.2 对照过的关键文件

- `diffusion.py`（训练目标、`x_t⊕x_0`、采样更新）
- `noise_schedule.py`（loglinear 噪声）
- `models/hf/modeling_bd3lm.py`（`block_diff_mask` 与训练/采样掩码）
- `configs/algo/bd3lm.yaml`（clip search / var_min 等策略）

---

## 2. 改造前状态（忠实记录）

改造前你项目的 Block Diffusion 路径主要是：

1. 训练：每个样本每步只随机一个 block，按比例 mask，masked CE
2. 推理：每轮前向后按置信度挑一批位置填 token（启发式）
3. 主干注意力：全 token-causal 下三角，没有论文训练态 `x_t⊕x_0` 掩码

这能跑通，但不等同论文 BD3-LM 核心算法。

---

## 3. 代码改造过程（逐步）

### 3.1 步骤 A：主干支持论文掩码注入

改动：

- 修改 `DualStreamRoformer.forward(...)`，新增可选参数：
  - `dual_attn_mask`
  - `single_attn_mask`
  - `dual_is_causal`
  - `single_is_causal`

影响：

- 不破坏原 AR 路径（默认保持旧行为）
- 为 Block Diffusion 注入论文掩码提供接口

文件：

- `cube3d/model/gpt/dual_stream_roformer.py`

### 3.2 步骤 B：在 BlockDiffusionRoformer 实现两种论文掩码

改动：

- 在 `BlockDiffusionRoformer` 新增：
  - `block_causal` 形状掩码（采样）
  - `bd_training` 掩码（`x_t⊕x_0`，复现官方 `block_diff_mask` 规则）
  - 条件前缀拼接后全掩码构造（`[cond | shape]`）
  - 掩码缓存
- 重载 `forward(..., attention_mode, block_size)`，支持：
  - `attention_mode="causal"`（旧路径）
  - `attention_mode="block_causal"`
  - `attention_mode="bd_training"`

影响：

- 训练与采样可走不同掩码模式
- 论文核心 mask 机制落地到 dual-stream 架构

文件：

- `cube3d/model/gpt/block_diffusion_roformer.py`

### 3.3 步骤 C：训练逻辑改为论文式 `x_t⊕x_0 + subs + 1/t`

改动：

- 新增噪声/目标工具模块：
  - `LogLinearSchedule`
  - `q_xt`
  - `subs_parameterization`
  - `restrict_logits_to_codes_and_mask`
- 训练 runner 由“单 block masked CE”切换为：
  1. 按 block 采样 `t`
  2. 构造 `x_t`
  3. 输入 `[x_t | x_0]`
  4. `attention_mode="bd_training"` 前向
  5. 仅使用前半段 logits
  6. `subs` 参数化
  7. `NLL * (1/t)` loss
- 验证阶段新增 `val_block_var`
- 新增 clipped schedule 搜索（`var_min` 相关配置）
- 修复了一个设备兼容问题：`runtime.device='cuda'` 时自动规范到 `cuda:0`

影响：

- 训练目标从启发式 masked CE 转为论文核心目标
- 可做噪声区间搜索以降低训练方差

文件：

- `cube3d/train/noise/bd3_schedule.py`（新增）
- `cube3d/train/runners/train_block_diffusion_t2s.py`（重构）
- `cube3d/train/noise/__init__.py`

### 3.4 步骤 D：采样器改为论文反向更新

改动：

- 删除旧版“置信度 top-k 解 mask”逻辑
- 采样改为反向转移更新：
  - `mask_prob = move_chance(s) / move_chance(t)`
  - masked 位置按 `q(x_s|x_t)` 采样
  - carry-over unmasking
- 推理前向切换到 `attention_mode="block_causal"`
- 增加 `first_hitting` 选项（默认关闭）

影响：

- 采样行为与训练目标更一致
- 更接近论文的 block 内 diffusion 更新机制

文件：

- `cube3d/inference/engine_block_diffusion.py`

### 3.5 步骤 E：配置与文档同步

改动：

- 配置由旧 `beta_low/beta_high` 迁移到论文语义：
  - `eps_min/eps_max`
  - `antithetic_sampling`
  - `resample`
  - `var_min`
  - `clip_search_delta`
  - `clip_search_widths`
  - `fix_clipping`
  - `first_hitting`
- 更新架构文档为三模型对照

文件：

- `cube3d/configs/block_diffusion_t2s.yaml`
- `BLOCK_DIFFUSION_T2S_ARCHITECTURE.md`

---

## 4. 验证与结果（忠实记录）

### 4.1 已完成的验证

1. 语法编译检查通过：
   - `python -m py_compile` 覆盖本次改动文件
2. 模型前向 smoke 通过（真实大模型配置）：
   - `causal` / `block_causal` / `bd_training` 三种模式均可前向
3. 训练目标 smoke 通过（小模型合成数据）：
   - 完整跑通：`x_t⊕x_0 -> bd_training -> subs -> loss -> backward -> optimizer.step`
4. 采样逻辑 smoke 通过（小模型合成数据）：
   - block-causal 多轮更新可将 mask 消除
5. 掩码公式一致性检查通过：
   - `bd_training` 掩码与官方 `block_diff_mask` 等价验证通过

### 4.2 未完成项与原因

- 用真实大模型权重跑 1-step runner smoke 时出现 GPU OOM（显存被占满），不是逻辑报错：
  - 报错：`torch.OutOfMemoryError`
  - 触发点：optimizer step

---

## 5. 改造后的架构（详细）

### 5.1 训练架构（当前）

1. `x_0`：shape token 真值
2. blockwise 采样 `t`（loglinear）
3. `q(x_t|x_0)`：按 `t` 将 token 变为 `[MASK]`
4. 构造输入 `[x_t | x_0]`
5. DualStream 主干前向（`bd_training` 掩码）
6. 输出前半段 logits（对应 `x_t` 位）
7. 限制 vocab 到 `codes + MASK`
8. `subs` 参数化（未 mask 位置强制 one-hot）
9. `NLL * (1/t)` 聚合并反向传播

### 5.2 推理架构（当前）

1. 全序列初始化为 `[MASK]`
2. block 从左到右（外层 block-AR）
3. 当前 block 内多步 `t -> s`：
   - `block_causal` 掩码前向得到 `p(x_0|x_t)`
   - 用 `mask_prob` 做反向更新
   - 已解开位置 carry-over
4. block 完成后进入下一 block

---

## 6. 与论文仍有差异（必须知道）

以下差异是工程层面的，不是目标函数层面的主偏差：

1. 主干架构：论文官方是单流 DiT 风格；本项目保留 DualStreamRoformer（为复用 text-to-shape 权重）
2. 注意力后端：官方有 `flex attention` 路径；当前主要是 `sdpa`
3. 官方完整训练生态（Lightning/评估脚本）未整体迁入，仅迁入关键算法路径

结论：

- 当前实现在“训练目标 + 训练掩码 + 采样更新”这三块已尽量对齐论文核心。
- 与官方代码仍非 1:1 代码同构（架构骨架不同），这是有意保留原项目能力的折中。

---

## 7. 训练指导（可直接执行）

### 7.1 训练前检查

1. 确认数据 manifest 可读：
   - `cube3d/configs/block_diffusion_t2s.yaml` 中 `data.manifest_path`
2. 确认权重路径存在：
   - `model.gpt_ckpt_path`
   - `model.shape_ckpt_path`
3. 确认输出目录在大盘：
   - 建议写到 `/root/autodl-fs/...`

### 7.2 关键配置建议（2x32G 5090）

建议从以下设置开始（再根据显存微调）：

- `train.micro_batch_size_per_gpu: 1`
- `train.grad_accum_steps: 8~16`
- `train.grad_checkpoint: true`
- `train.amp_dtype: bfloat16`
- `diffusion.block_size: 32`（先稳）
- `diffusion.first_hitting: false`（先稳定）
- `diffusion.var_min: true`

### 7.3 启动训练命令（示例）

```bash
cd /root/cube
torchrun --nproc_per_node=2 --standalone \
  -m cube3d.train.runners.train_block_diffusion_t2s \
  --config /tmp/block_diffusion_t2s_2x5090.yaml
```

### 7.4 TensorBoard

```bash
tensorboard --logdir /root/autodl-fs/block_diffusion_train/tensorboard --host 127.0.0.1 --port 6006
```

关注曲线：

- `train/loss`
- `val/loss`
- `val/block_var`
- `diffusion/eps_min`, `diffusion/eps_max`

### 7.5 常见问题

1. OOM：
   - 降低 `micro_batch_size_per_gpu`
   - 增加 `grad_accum_steps`
   - 降低 `block_size`
2. 收敛慢或不稳：
   - 先固定 `fix_clipping=true` 观察
   - 再打开 `var_min` 做区间搜索

---

## 8. 本次改动文件清单

- `cube3d/model/gpt/dual_stream_roformer.py`
- `cube3d/model/gpt/block_diffusion_roformer.py`
- `cube3d/train/noise/bd3_schedule.py` (new)
- `cube3d/train/noise/__init__.py`
- `cube3d/train/runners/train_block_diffusion_t2s.py`
- `cube3d/inference/engine_block_diffusion.py`
- `cube3d/configs/block_diffusion_t2s.yaml`
- `BLOCK_DIFFUSION_T2S_ARCHITECTURE.md`
- `BLOCK_DIFFUSION_PAPER_ALIGNMENT_LOG.md` (new)

