# Cube Text-to-Shape Block Diffusion 训练计划（仅替换 GPT）

## 1. 目标与边界

- 目标：仅将现有 Text-to-Shape 的自回归 GPT 替换为 Block Diffusion。
- 保持不变：
  - `OneDAutoEncoder`（shape tokenizer）与 mesh 解码链路
  - 文本条件编码体系（CLIP）
  - bbox 条件输入与归一化规则（v0.5）
- 新模型输入/输出：
  - 输入：`text condition (+ bbox + timestep + noisy shape tokens)`
  - 输出：`shape tokens`（长度 1024，词表 16384）

---

## 2. 先决条件（冻结模块）

- 冻结 `shape tokenizer`（`shape_tokenizer.safetensors`）。
- 冻结 `text tokenizer + text encoder`（CLIP）。
- 训练时不更新上述参数，仅训练 Block Diffusion 模型。

---

## 3. 数据构建计划

## 3.1 原始样本字段

每个样本至少包含：

- `mesh_path`
- `caption`（可多条）
- `bbox_xyz`
- `split`

## 3.2 离线预处理（推荐）

离线生成并缓存：

- `shape_ids`: `[1024]`（int32）
- `text_embed`: `[77, 768]`（float16/float32，CLIP last hidden state）
- `bbox_xyz`: `[3]`（float32）

推荐 manifest（jsonl/parquet）字段：

- `asset_id`
- `shape_tokens_path`
- `text_embed_path`（或内联）
- `bbox_xyz`
- `caption_len_bucket`（short/medium/long）
- `split`

> 说明：若想灵活更换文本编码器，可只存 `input_ids/attention_mask`，训练时在线跑 CLIP；但吞吐会明显下降。

---

## 4. Block Diffusion 训练接口定义

- `x0`: 真实 shape token 序列 `[B, 1024]`
- `xt`: 加噪后的序列（block mask/random replace）
- `cond_text`: `[B, 77, D]`（或投影后维度）
- `cond_bbox`: `[B, 3] -> bbox token`
- `t`: 扩散时间步
- 模型输出：`logits [B, 1024, 16384]`
- 损失：只在被破坏位置计算 CE（可加权）

---

## 5. 训练步骤（分阶段）

## 阶段 A：数据与管线打通

1. 用冻结 tokenizer 离线编码 mesh 得到 `shape_ids`。
2. 用冻结 CLIP 编码 caption 得到 `text_embed`。
3. 生成统一 manifest。
4. 编写 dataloader，支持多 caption 随机采样与 bbox 读取。

验收：

- 能从一个 batch 正确拿到 `x0/text_embed/bbox`；
- 随机抽样可被 `shape_model.decode_indices` 正常解码成 mesh。

## 阶段 B：Block Diffusion 主训练

1. 定义 block 加噪策略：
   - 随机选若干连续 block 位置；
   - block 内 token 做 `[MASK]` 或随机替换；
   - 随 `t` 控制破坏强度。
2. 加入条件：
   - 文本条件（CLIP embedding）
   - bbox 条件 token（保留 v0.5）
3. 加入 CFG 训练：
   - 10% 概率置空文本条件（对齐论文 CFG 训练思想）
4. 加入 bbox jitter：
   - 训练时对 `bbox_xyz` 做小扰动（对齐 v0.5）
5. 训练目标：
   - masked positions 的 token CE
   - 可选 auxiliary loss（如全序列 CE 小权重）

验收：

- 训练/验证 CE 稳定下降；
- 采样得到的 token 解码后非空 mesh 比例高。

## 阶段 C：采样器与推理对接

1. 实现 iterative denoising：
   - 从全 mask（或高噪声）开始；
   - 迭代多步预测并回填 tokens；
   - 支持 CFG（cond/uncond 组合）。
2. 输出最终 `shape_ids` 接回现有解码：
   - `decode_indices -> extract_geometry`
3. 提供新引擎入口（如 `EngineBlockDiffusion`），不改旧 GPT 路径。

验收：

- 与当前 `generate.py` 风格一致输出 `.obj`；
- 支持 prompt + bbox 条件生成。

---

## 6. 评估与对比

至少跟现有 GPT 基线对比：

- token-level：验证 CE / NLL
- 几何有效性：non-empty mesh rate
- prompt adherence：人工打分或自动文本一致性评估
- bbox adherence：生成 mesh 的 AABB 与条件 bbox 误差
- 采样效率：步数 vs 质量曲线

---

## 7. 配置建议（首版）

以下为首版可调项（建议写入 `configs/block_diffusion_t2s.yaml`）：

- `num_timesteps`
- `block_size_range`
- `mask_ratio_schedule`
- `noise_type`（mask / random / mixed）
- `cfg_drop_prob`（默认 0.1）
- `bbox_jitter_std`
- `lr / betas / weight_decay / warmup / max_steps`
- `batch_size / grad_accum / amp_dtype`

---

## 8. 风险与优先级

P0（必须先做）：

- 数据离线化质量（shape_ids 正确、caption 对齐、bbox 合法）
- 采样器稳定性（迭代恢复是否崩塌）

P1（高优先）：

- block 加噪策略与时间步设计
- CFG 与 bbox 的平衡（防止模型只看 bbox）

P2（优化）：

- 采样加速（少步高质）
- 推理时多样性控制（temperature/top-k 类策略）

---

## 9. 里程碑（建议）

- M1（1-2 周）：数据离线化 + tiny 训练能收敛 + 能解码 mesh
- M2（2-4 周）：中等规模训练 + 采样器可用 + 与 GPT 基线可比较
- M3（4+ 周）：大规模训练 + 质量/速度优化 + 完整推理集成

---

## 10. 一句话执行顺序

先固定两端 tokenizer（text/shape），把任务降维为“条件 embedding -> shape token 去噪恢复”，再做 Block Diffusion 训练与迭代采样，最后无缝接回现有 mesh 解码链路。

