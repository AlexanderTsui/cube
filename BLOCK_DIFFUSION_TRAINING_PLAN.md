# Cube Text-to-Shape Block Diffusion 实现文档（已落地版本）

> 本文档记录当前仓库中 **已实现** 的 Block Diffusion for Text-to-Shape 架构、训练与推理流程，并明确与论文完整版方案的差距。
>
> 相关参考：
> - Block Diffusion: https://arxiv.org/abs/2503.09573
> - Cube: https://arxiv.org/abs/2503.15475

---

## 1. 目标与当前状态

### 1.1 目标

在保持现有 `shape tokenizer -> decode_indices -> extract_geometry` 链路不变的前提下，新增一条可训练、可采样、可解码 mesh 的 Block Diffusion 文本到形状路径。

### 1.2 当前实现状态（已完成）

已完成以下核心模块：

- Block Diffusion 模型包装：`cube3d/model/gpt/block_diffusion_roformer.py`
- 训练数据集加载：`cube3d/train/data/block_diffusion_dataset.py`
- 噪声与调度：`cube3d/train/noise/masked_schedule.py`
- 训练 runner：`cube3d/train/runners/train_block_diffusion_t2s.py`
- 训练配置：`cube3d/configs/block_diffusion_t2s.yaml`
- 推理引擎：`cube3d/inference/engine_block_diffusion.py`
- CLI 接入：`cube3d/generate.py` 增加 `--use-block-diffusion` 等参数
- 采样修复：`cube3d/inference/logits_postprocesses.py` 修复 `top_p` 被写死问题

---

## 2. 数据接口（训练输入）

### 2.1 数据来源与文件组织

当前训练使用离线构建后的 bdcube 数据集：

- manifest: `/root/autodl-tmp/bdcube_dataset/manifests/pairs_bdcube.jsonl`
- feature: `/root/autodl-tmp/bdcube_dataset/features/<uid_prefix>/<uid>.npz`

### 2.2 npz 字段规范

每个样本 `.npz` 包含：

- `shape_ids`: `int32 [1024]`
- `text_hidden`: `float16 [77, 768]`
- `text_attention_mask`: `int8 [77]`
- `bbox_xyz`: `float32 [3]`
- 额外保留：`text_input_ids`, `text_pooled`

### 2.3 Dataset 输出张量

`BlockDiffusionDataset` 输出：

- `shape_ids: LongTensor [1024]`
- `text_hidden: FloatTensor [77,768]`
- `text_attention_mask: BoolTensor [77]`
- `bbox_xyz: FloatTensor [3]`
- `uid`

### 2.4 数据切分策略

按 `uid` 和 seed 进行稳定 hash 切分：

- `uid_to_split(uid, val_ratio, seed)`
- 默认 `val_ratio` 由配置控制（当前示例配置为 `0.2`）

---

## 3. 模型架构实现细节

## 3.1 主干复用策略

实现采用“复用现有 DualStreamRoformer + 扩展词表”的方案：

- 类：`BlockDiffusionRoformer`
- 继承：`DualStreamRoformer`
- 新增能力：`add_mask_token()` / `ensure_mask_token()`

### 3.2 [MASK] token 扩展机制

通过追加词表最后一行实现：

- `self.shape_mask_id = old_vocab_size`
- `wte` 与 `lm_head` 同步扩容 `+1`
- 新行初始化策略（默认）：从 `padding` 行拷贝（可切换高斯初始化）

这保证 AR checkpoint 可平滑迁移到 Block Diffusion。

### 3.3 条件输入路径

条件流沿用原 Cube 设计：

- 文本条件：`text_hidden -> gpt_model.encode_text -> [B,77,1536]`
- bbox 条件：`bbox_proj(3->1536)` 追加为 1 个 token
- 最终 cond：`[B,78,1536]`（use_bbox=true）

### 3.4 序列与 block 定义

- 固定长度：`L=1024`
- `block_size` 配置化（默认 32）
- block 数：`B = L / block_size`

---

## 4. 训练实现（当前版本）

## 4.1 配置入口

配置文件：`cube3d/configs/block_diffusion_t2s.yaml`

关键项：

- `model.base_config_path`
- `model.gpt_ckpt_path`
- `model.shape_ckpt_path`
- `diffusion.block_size`
- `diffusion.beta_low / beta_high`
- `diffusion.cfg_drop_prob`
- `train.max_steps / eval_every / save_every`

### 4.2 参数初始化流程

`train_block_diffusion_t2s.py` 启动时：

1. 加载 `open_model_v0.5.yaml` 基础结构
2. 加载 GPT checkpoint 到 `BlockDiffusionRoformer`
3. 加载 shape tokenizer checkpoint（仅用于 codebook 对齐）
4. 用 shape codebook 通过 `shape_proj` 对齐到 `wte` / `lm_head` 前 `num_codes` 行
5. 扩展 `[MASK]` token

### 4.3 噪声调度与加噪

当前使用 `ClippedMaskSchedule`：

- `mask_ratio ~ U[beta_low, beta_high]`
- 权重代理：`weight = 1 / mask_ratio`

加噪函数 `mask_one_block_per_sample(...)` 的当前策略：

- 每个样本每步仅随机选 1 个 block
- 在该 block 内按比例随机位置置为 `[MASK]`
- `n_mask = max(1, round(block_size * ratio))`

### 4.4 训练目标

当前实现的损失：

- 仅在 masked 位置计算 CE
- 按 `1/mask_ratio` 做样本权重

公式（实现等价）：

- `L = mean( CE(logits_masked, target_masked) * w(mask_ratio) )`

### 4.5 CFG dropout

训练时按 `cfg_drop_prob` 将整条样本的 `text_hidden` 置零（bbox 仍保留），用于条件/无条件共享建模。

### 4.6 验证与保存

- `evaluate(...)` 复用同样一块加噪评估 `val_loss` 与 `val_mask_acc`
- 周期性保存：
  - `block_diffusion_step_<N>.safetensors`
  - `block_diffusion_step_<N>.json`（含 `shape_mask_id` 等元信息）

---

## 5. 推理实现（当前版本）

## 5.1 推理引擎

文件：`cube3d/inference/engine_block_diffusion.py`

类：`EngineBlockDiffusion`

### 5.2 checkpoint 兼容加载

加载逻辑支持两种情况：

- AR ckpt（不含 mask 行）: 先加载后 `ensure_mask_token()`
- BD ckpt（已含 mask 行）: 以 `add_mask_token=true` 结构加载

### 5.3 block 级采样过程

`run_gpt(...)` 的实现：

1. 全序列初始化为 `[MASK]`
2. 逐 block 处理（从前到后）
3. 每个 block 做 `num_denoise_steps` 轮：
   - 前向得到当前 block logits
   - 计算每个位置置信度（max prob）
   - 采样候选 token（argmax 或 top-p）
   - 每轮仅解开一部分 mask（`ceil(remaining/steps_left)`）
4. block 结束仍有 mask 时，fallback 用 argmax 填满

该策略实现“once-unmasked keep”近似，不会反复重掩码。

### 5.4 解码到 mesh

与现有引擎一致：

- `shape_ids -> shape_model.decode_indices -> extract_geometry`

---

## 6. CLI 集成

`cube3d/generate.py` 新增参数：

- `--use-block-diffusion`
- `--block-size`
- `--num-diffusion-steps`

行为约束：

- 与 `--fast-inference` 互斥（启用 block diffusion 时忽略 fast）

---

## 7. 与论文方案的差异（当前未实现）

当前实现为了先打通 smoke 闭环，和论文完整版仍有差异：

1. **未实现 x_t ⊕ x 向量化训练掩码（MBD/MOBC/MBC）**
   - 目前是“每样本每步一个 block”的简化训练。
2. **未实现数据驱动 `(beta, omega)` 网格搜索**
   - 当前是固定 clipped 区间。
3. **未实现论文中的完整方差估计与调度更新循环**。
4. **未实现更高效稀疏注意力 kernel（如 FlexAttention 路径）**。

---

## 8. 当前可运行命令（smoke）

### 8.1 训练 smoke

```bash
python -m cube3d.train.runners.train_block_diffusion_t2s \
  --config cube3d/configs/block_diffusion_t2s.yaml
```

### 8.1.1 双卡正式训练（DDP）

```bash
torchrun --nproc_per_node=2 --standalone \
  -m cube3d.train.runners.train_block_diffusion_t2s \
  --config cube3d/configs/block_diffusion_t2s.yaml
```

### 8.1.2 TensorBoard（本机端口映射）

```bash
tensorboard --logdir outputs/block_diffusion/tb --host 127.0.0.1 --port 6006
```

### 8.2 推理 smoke

```bash
python -m cube3d.generate \
  --use-block-diffusion \
  --config-path cube3d/configs/open_model_v0.5.yaml \
  --gpt-ckpt-path model_weights/shape_gpt.safetensors \
  --shape-ckpt-path model_weights/shape_tokenizer.safetensors \
  --prompt "a small toy car" \
  --block-size 32 \
  --num-diffusion-steps 8
```

---

## 9. 下一阶段建议（P1）

1. 实现论文式向量化训练（`x_t ⊕ x` + `MBD/MOBC/MBC`）
2. 增加 `(beta, omega)` 周期网格搜索并自动回写调度
3. 引入更完整评测：
   - token CE / masked acc
   - non-empty mesh rate
   - bbox adherence
   - 推理时延与显存
4. 增加与 AR 基线的 A/B 输出对比脚本

---

## 10. 本轮实测记录（2026-03-04，更新）

本节记录“用当前已下载 Objaverse 子集构建训练集 + 对本次 Block Diffusion 实现逐一测试”的实际执行结果。

### 10.1 训练集构建（已执行，使用当前下载子集）

执行命令：

```bash
python dataset/build_bdcube_dataset.py \
  --pairs-jsonl /root/autodl-tmp/objaverse_subset/manifests/pairs.jsonl \
  --output-root /root/autodl-tmp/bdcube_dataset \
  --device cuda
```

输入规模：

- Objaverse 子集 manifest：`447` 条（`/root/autodl-tmp/objaverse_subset/manifests/pairs.jsonl`）

构建结果：

- 成功样本：`436`
- 失败样本：`11`
  - `missing_glb`: `8`
  - `ValueError: no mesh geometry in scene`: `3`
- 产出 feature：`436` 个 `.npz`
- 进度文件耗时：`392.25s`（见 `progress.json`）

数据文件：

- `/root/autodl-tmp/bdcube_dataset/manifests/pairs_bdcube.jsonl`（436 行）
- `/root/autodl-tmp/bdcube_dataset/manifests/completed_uids.txt`（436 行）
- `/root/autodl-tmp/bdcube_dataset/manifests/failed_uids.txt`（11 行）
- `/root/autodl-tmp/bdcube_dataset/features/<shard>/<uid>.npz`

字段一致性抽检（npz）：

- `shape_ids`: 长度 `1024`
- `text_hidden`: 形状 `77 x 768`
- `text_attention_mask`: 长度 `77`
- `bbox_xyz`: 长度 `3`

### 10.2 模块逐一测试（已执行）

#### A) 数据加载模块 `BlockDiffusionDataset`

测试结果（`val_ratio=0.2, seed=42`）：

- train: `344`
- val: `92`
- batch 形状：
  - `shape_ids`: `(4, 1024)` `int64`
  - `text_hidden`: `(4, 77, 768)` `float32`
  - `text_attention_mask`: `(4, 77)` `bool`
  - `bbox_xyz`: `(4, 3)` `float32`

#### B) 噪声模块 `masked_schedule.py`

测试结果：

- `sample_ratio()` 在 `[0.3, 0.8]` 区间采样正常
- `weight_from_ratio()` 结果正常
- `mask_one_block_per_sample()` 能正确写入 `[MASK]`，且 mask 数量与 ratio 对齐

#### C) 模型模块 `BlockDiffusionRoformer`

测试结果：

- 从 `shape_gpt.safetensors` 成功加载
- `ensure_mask_token()` 后：
  - `shape_mask_id = 16387`
  - `vocab_size = 16388`
- 前向输出形状：`(2, 1024, 16388)`（CUDA）

### 10.3 训练 runner 测试（已执行）

#### A) CUDA 路径（发现当前瓶颈）

`device=cuda, batch_size=1` 在 `optimizer.step()` 阶段 OOM（约 31GB 显存已占满）。

结论：当前“全参数 AdamW”配置在 5090 32GB 上仍超显存，需要后续做训练内存优化（如参数高效微调/优化器状态压缩/冻结策略等）。

#### B) CPU smoke（完整流程跑通）

执行命令：

```bash
python -m cube3d.train.runners.train_block_diffusion_t2s \
  --config /tmp/block_diffusion_t2s_smoke_20260304_cpu2.yaml
```

关键配置：

- `device: cpu`
- `batch_size: 1`
- `max_steps: 2`
- `eval_every: 1`
- `save_every: 2`

结果（exit code 0）：

- `step=1 train_loss=124.4463 val_loss=129.6616`
- `step=2 train_loss=198.3432 val_loss=61.3649`
- checkpoint 成功写出：
  - `/root/autodl-tmp/block_diffusion_smoke_20260304_cpu2/block_diffusion_step_2.safetensors`
  - `/root/autodl-tmp/block_diffusion_smoke_20260304_cpu2/block_diffusion_step_2.json`

### 10.4 推理链路测试（已执行）

#### A) `EngineBlockDiffusion` + AR 基座权重

测试结果：

- `model_weights/shape_gpt.safetensors` 兼容加载成功（自动补 `[MASK]` 行）
- `run_gpt()` 正常返回 `shape_ids`，形状 `(1, 1024)`，取值在 codebook 范围内

#### B) `EngineBlockDiffusion` + 训练产出 checkpoint

使用：

- `/root/autodl-tmp/block_diffusion_smoke_20260304_cpu2/block_diffusion_step_2.safetensors`

测试结果：

- 采样 + 解码链路正常（CUDA）
- CLI 端到端命令成功：
  - 输出：`/tmp/block_diffusion_generate_20260304_step2/output.obj`
  - mesh 规模：`42 vertices / 56 faces`

### 10.5 `top_p` 修复回归验证

`cube3d/inference/logits_postprocesses.py` 测试通过：

- `top_p=0.2` 保留 token 数：`1`
- `top_p=0.9` 保留 token 数：`2`

说明 `top_p` 阈值已按传参生效，不再固定为常数。

---

## 11. 结论（本轮）

在当前“已下载的 447 条 Objaverse 子集”上，已完成并验证：

1. 训练集构建：`447 -> 436` 可用配对样本
2. 模块级测试：数据/噪声/模型/采样后处理均通过
3. 训练全流程：`train + val + ckpt save` 已在 CPU smoke 跑通
4. 推理全流程：`checkpoint -> block diffusion sampling -> decode -> obj` 跑通

当前主要问题不是功能正确性，而是 **CUDA 全参数训练显存不足（AdamW OOM）**。后续应优先做训练内存优化，再推进更长步数与正式训练。

---

## 12. 数据蒸馏扩展（2026-03-05）

本节记录“使用当前已下载 Objaverse 数据（约 49GB GLB）继续蒸馏 Block Diffusion 训练数据”的执行结果。

### 12.1 输入去重与可用样本

原始输入：

- `/root/autodl-tmp/objaverse_subset/manifests/pairs.jsonl`：`13535` 行

去重与可用性筛选（按 `uid` 去重，剔除缺失 `glb`）后：

- unique uid：`5130`
- 缺失 glb（unique）：`8`
- 可处理样本：`5122`
- 去重清单：`/tmp/objaverse_pairs_unique_existing_20260305.jsonl`

### 12.2 蒸馏执行

执行命令：

```bash
python dataset/build_bdcube_dataset.py \
  --pairs-jsonl /tmp/objaverse_pairs_unique_existing_20260305.jsonl \
  --output-root /root/autodl-tmp/bdcube_dataset \
  --device cuda
```

说明：

- 采用“续跑”方式（不加 `--overwrite`），复用已完成样本并继续追加。
- 历史基线：已完成 `436`，失败 `11`（2026-03-04）。

本次新增结果：

- 新增完成：`4471`
- 新增失败：`215`

累计结果（当前）：

- `pairs_bdcube.jsonl`：`4907` 行
- `completed_uids.txt`：`4907` 行
- `failed_uids.txt`：`226` 行
- `features/**/*.npz`：`4907` 个

### 12.3 训练可用性校验

一致性检查：

- manifest 行数 = completed 行数 = npz 文件数 = `4907`
- manifest 中 `feature_path` 缺失数：`0`

字段抽检（随机样本）：

- `shape_ids`: `(1024,) int32`
- `text_hidden`: `(77, 768) float16`
- `text_attention_mask`: `(77,) int8`
- `bbox_xyz`: `(3,) float32`

DataLoader 校验（`val_ratio=0.2, seed=42`）：

- train: `3930`
- val: `977`
- batch 形状：
  - `shape_ids`: `(4, 1024)` `int64`
  - `text_hidden`: `(4, 77, 768)` `float32`
  - `text_attention_mask`: `(4, 77)` `bool`
  - `bbox_xyz`: `(4, 3)` `float32`

训练 smoke（基于新蒸馏数据）：

```bash
python -m cube3d.train.runners.train_block_diffusion_t2s \
  --config /tmp/block_diffusion_train_smoke_20260305.yaml
```

结果（`device=cpu, max_steps=1`，exit code 0）：

- `step=1 train_loss=90.8017`
- `val_loss=148.7541`
- checkpoint: `/tmp/block_diffusion_train_smoke_20260305/block_diffusion_step_1.safetensors`
