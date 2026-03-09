# Block Diffusion 论文对齐记录（当前代码状态）

最后更新：2026-03-09

## 1. 本轮改造目标

将项目中的 Block Diffusion 路径替换为：

- 核心算法与 BD3-LM 官方实现同构
- 面向 text-to-shape 场景加入条件增强（AdaLN + Cross-Attn）

## 2. 已完成的关键替换

### 2.1 模型骨架

- 重写 `cube3d/model/gpt/block_diffusion_dit.py`
- 支持官方核心机制：
  - `block_diff_mask`
  - `sample_mode/store_kv`
  - `reset_kv_cache`
  - `bd_training` 与 `block_causal`

### 2.2 训练路径

- 重写 `cube3d/train/runners/train_block_diffusion_t2s.py` 为单流主线
- 训练目标对齐：`[x_t|x_0] + subs + 1/t`
- 支持 `var_min` clipped schedule 搜索
- 支持训练 CFG drop（无条件分支采样）

### 2.3 推理路径

- 重写 `cube3d/inference/engine_block_diffusion.py`
- 采样流程对齐 semi-AR + block 内反向更新
- 支持 `first_hitting`
- 支持 CFG 推理与 nucleus
- 支持 KV cache（当 `cfg_scale==1`）

### 2.4 配置

- 更新 `cube3d/configs/block_diffusion_t2s.yaml`
- 移除 dual/single 切换语义，统一到官方同构单流主线

## 3. 对齐结论

### 3.1 已对齐（核心）

- [x] `[x_t|x_0]` 训练输入
- [x] `block_diff_mask` 训练掩码
- [x] `subs` 参数化
- [x] `1/t` 损失加权
- [x] semi-AR 采样流程
- [x] `first_hitting` 采样选项

### 3.2 非 1:1 的点（有意）

- [ ] 任务从纯 LM 扩展为 text-to-shape 条件生成
- [ ] 条件分支采用 AdaLN + Cross-Attn 混合增强
- [ ] attention 后端可能因环境回退到 SDPA

## 4. 影响

正向：

- 核心 Block Diffusion 机制与论文路径一致性显著提高
- 条件控制能力较“无条件官方骨架”更强

代价：

- CFG 推理成本更高
- `cfg_scale!=1` 时当前实现会关闭 KV cache（保证正确性优先）
- 旧 BD checkpoint 不再兼容新主线
