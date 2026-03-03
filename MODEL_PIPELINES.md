# Cube 模型 Pipeline 代码级详解（v0.5，按当前仓库实现）

本文基于当前仓库源码逐行梳理，目标是把 **Shape tokenizer** 与 **Text-to-Shape** 两条主链路讲清楚：

1. `shape tokenizer`：几何 <-> 离散 token（`OneDAutoEncoder`）
2. `text -> token -> mesh`：文本到形状（`Engine`/`EngineFast` + `DualStreamRoformer`）

---

## 0. 入口、配置与记号

### 0.1 主要入口文件

- 命令行入口：`cube3d/generate.py`
- 推理编排：`cube3d/inference/engine.py`
- GPT 主体：`cube3d/model/gpt/dual_stream_roformer.py`
- 双流注意力：`cube3d/model/transformers/dual_stream_attention.py`
- RoPE 与注意力核：`cube3d/model/transformers/rope.py`
- Shape tokenizer 主体：`cube3d/model/autoencoder/one_d_autoencoder.py`
- VQ 量化器：`cube3d/model/autoencoder/spherical_vq.py`
- Fourier embedder：`cube3d/model/autoencoder/embedder.py`
- 几何提取：`cube3d/model/autoencoder/grid.py`
- mesh 后处理：`cube3d/mesh_utils/postprocessing.py`

### 0.2 默认配置（`cube3d/configs/open_model_v0.5.yaml`）

- `shape_model.num_encoder_latents = 1024`
- `shape_model.num_codes = 16384`
- `shape_model.width = 768`
- `shape_model.embed_dim = 32`
- `gpt_model.n_embd = 1536`
- `gpt_model.n_layer = 23`
- `gpt_model.n_single_layer = 1`
- `gpt_model.n_head = 12`
- `gpt_model.use_bbox = true`
- `text_model_pretrained_model_name_or_path = openai/clip-vit-large-patch14`

### 0.3 记号约定

- `B`：batch size
- `N`：输入点云点数（示例脚本默认 `8192`）
- `L`：shape token 长度（v0.5 默认 `1024`）
- `S`：条件 token 长度（CLIP hidden 默认 `77`，加 bbox 后 `78`）
- `D_s`：shape model 宽度（`768`）
- `D_g`：GPT 隐藏维（`1536`）
- `H`：头数（`12`）
- `Hd`：每头维度（`D/H`）

---

## 1. CLI 端到端 Pipeline（`python -m cube3d.generate`）

### 1.1 参数解析与设备选择

`cube3d/generate.py` 中主要参数：

- 必填：`--gpt-ckpt-path`、`--shape-ckpt-path`、`--prompt`
- 可选：`--fast-inference`、`--top-p`、`--bounding-box-xyz`、`--render-gif`
- 网格提取相关：`--resolution-base`（默认 8.0）
- 输出目录：`--output-dir`（默认 `outputs/`）

设备通过 `select_device()` 自动选：`cuda -> mps -> cpu`。

### 1.2 Engine 初始化分支

- `--fast-inference` 开启时优先构建 `EngineFast`（CUDA Graph 方案）。
- 若 fast 模式触发 CUDA OOM（当前仓库实现已处理），脚本会重启进程并去掉 `--fast-inference`，自动降级到 `Engine`。
- 普通模式直接使用 `Engine`。

### 1.3 bbox 预处理

若传入 `--bounding-box-xyz x y z`，先经过 `normalize_bbox()`：

- 先取 `max_l = max(x, y, z)`
- 再按 `BOUNDING_BOX_MAX_SIZE = 1.925` 等比例缩放
- 输出尺寸满足最长边为 `1.925`

### 1.4 主执行

`generate_mesh()` 调用：

1. `engine.t2s([prompt], use_kv_cache=True, ...)`
2. 得到 `(vertices, faces)`
3. 输出 `output.obj`
4. 如安装了 `pymeshlab`：可做清理/去浮块/简化
5. 未安装则退化为 `trimesh.Trimesh(...).export(obj)`
6. `--render-gif` 时额外调用 Blender 渲染 turntable gif

---

## 2. Shape Tokenizer Pipeline（`OneDAutoEncoder`）

核心是：

- 编码：`surface point cloud -> latent queries -> VQ token ids`
- 解码：`token ids -> latents -> occupancy field -> marching cubes mesh`

### 2.1 输入数据形态

在 `cube3d/vq_vae_encode_decode.py`（示例脚本）里：

1. 读取 mesh，清理坏几何
2. 归一化到近似单位尺度（`MESH_SCALE=0.96`）
3. 采样表面点和对应法线，默认 `8192` 点
4. 拼成 `point_cloud: [1, 8192, 6]`（`xyz + normal`）

`OneDAutoEncoder.encode(x)` 内部拆分为：

- `pts = x[..., :3]` -> `[B, N, 3]`
- `feats = x[..., 3:]` -> `[B, N, 3]`

### 2.2 坐标嵌入：`PhaseModulatedFourierEmbedder`

文件：`cube3d/model/autoencoder/embedder.py`

设 `num_freqs=128`、`input_dim=3`，则：

- `out_dim = input_dim * (2*num_freqs + 1) = 3 * 257 = 771`

前向构造三部分并拼接：

- 原坐标 `x`
- `cos(fm) + cos(pm)`
- `sin(fm) + sin(pm)`

其中 `fm` 由可学习 `weight` 调制，`pm` 由固定 `carrier` + 相位项构造。

### 2.3 Encoder：query latent 聚合

文件：`one_d_autoencoder.py` -> `OneDEncoder`

- 可学习 query 参数：
  - 若 `with_cls_token=true`：query 长度 `1 + num_latents`，v0.5 为 `1025`
  - 维度 `D_s=768`
- query 初始化后会通过 `init_sort`（按到第一个向量距离排序）
- 输入点特征路径：
  1. `embedder(pts)` -> `[B, N, 771]`
  2. `embed_point_feats=false` 时，与原始 `feats=[B,N,3]` 拼接 -> `[B,N,774]`
  3. `feat_in(MLPEmbedder)` -> `[B,N,768]`

Transformer 堆叠：

- 总层数 `num_encoder_layers=13`
- cross-attn 层索引：`[0,2,4,8]`（其余为 self-attn）
- 最终 `ln_f`

输出：`z_e = [B, 1025, 768]`（v0.5）。

若有 cls token：

- `z_cls = z_e[:,0,:]` -> `[B,768]`
- `z_e = z_e[:,1:,:]` -> `[B,1024,768]`

### 2.4 Bottleneck：`SphericalVectorQuantizer`

文件：`spherical_vq.py`

参数（v0.5）：

- `embed_dim=32`
- `num_codes=16384`
- `width=768`

关键结构：

- `codebook: Embedding[16384, 32]`
- `c_in: 768 -> 32`
- `c_out: 32 -> 768`
- `RMSNorm` + codebook 归一化（`get_codebook()`）

量化步骤：

1. `z_e = norm(c_in(z))` -> `[B,1024,32]`
2. 展平后和 codebook 做 `torch.cdist`
3. 最近邻 `argmin` 得到 `q` -> `[B,1024]`
4. 取回 `z_q` -> `[B,1024,32]`
5. 直通估计（STE）：`z + (z_q - z).detach()`
6. `c_out` 回到 `[B,1024,768]`

`encode()` 的关键信息：

- `d["indices"]`：离散 token 序列（长度固定 1024）
- `d["z_q"]`：量化 latent（32 维空间）
- `d["z_cls"]`：若启用 cls

### 2.5 Decoder：token -> latent

`decode_indices(shape_ids)` 流程：

1. `lookup_codebook(shape_ids)`
   - 先 embedding 到 `[B,1024,32]`
   - 再 `c_out` -> `[B,1024,768]`
2. `decode(z_q)` -> `OneDDecoder`

`OneDDecoder`：

- 可学习 `positional_encodings: [num_latents, 768]`（v0.5 为 `[1024,768]`）
- 加位置编码后通过 24 层 `EncoderLayer`
- 输出 `latents: [B,1024,768]`

注：`OneDDecoder` 里有 `self.query` 缓冲（用于可选 pad），但当前默认配置下为空，主路径不触发 padding。

### 2.6 Occupancy 查询头

`OneDOccupancyDecoder`：

1. `embedder(queries)`（3D 坐标）
2. `query_in(MLPEmbedder)` -> query feature
3. 与 `latents` 做 cross-attn
4. `ln_f + c_head` 输出占据 logits

输入/输出：

- `queries: [B, M, 3]`
- `latents: [B, 1024, 768]`
- `logits: [B, M]`

### 2.7 几何提取：`extract_geometry`

文件：`one_d_autoencoder.py` + `grid.py`

1. 在 `bounds` 内生成稠密网格点（默认 `[-1.05,1.05]^3`）
2. 网格边长 `G = 2^resolution_base + 1`
   - 例如默认 `resolution_base=8.0`，`G=257`
3. 总点数 `G^3`，按 `chunk_size` 分块查询 occupancy
4. reshape 为 `grid_logits: [B, G, G, G]`
5. 提取等值面 `level=0.0`：
   - 优先 `warp.MarchingCubes`（GPU）
   - 失败回退 `skimage.measure.marching_cubes`（CPU）
6. 顶点坐标从网格索引映射回真实 bbox
7. 面索引做轴顺序调整 `faces[:, [2,1,0]]`

返回：

- `mesh_v_f`: `[(vertices, faces), ...]`
- `has_surface`: 每个样本是否提取成功

---

## 3. Text -> Token Pipeline（`Engine` + `DualStreamRoformer`）

### 3.1 Engine 初始化时的关键“对齐”步骤

`Engine.__init__()` 中除加载模型外，有一步很关键：

1. 从 shape tokenizer 取 codebook（32维）
2. 通过 `gpt_model.shape_proj: 32 -> 1536`
3. 拷贝到 GPT token embedding `wte` 的前 `num_codes` 行

这使得 GPT 输出的 shape token id 与 tokenizer 的 codebook 语义空间对齐。

### 3.2 文本条件编码（`prepare_inputs`）

#### A) CLIP 编码

- tokenizer 固定 padding/trunc 到 `model_max_length`（CLIP-L/14 实际是 77）
- `CLIPTextModelWithProjection` 输出：
  - `use_pooled_text_embed=false` 时使用 `last_hidden_state: [B,77,768]`
- 经 `gpt_model.encode_text`（线性投影） -> `[B,77,1536]`

#### B) 形状序列起点（BOS）

- `shape_bos_id` 经 `wte` 得到 `embed: [B,1,1536]`

#### C) bbox 条件 token

若模型启用 bbox（v0.5 为 true）：

- `bbox_proj([B,3]) -> [B,1536]`
- `unsqueeze(1)` 后拼到 cond 序列尾部
- cond 从 `[B,77,1536]` 变为 `[B,78,1536]`

若未提供 bbox，使用全零 bbox 向量。

#### D) CFG 批拼接

当 `guidance_scale > 0`：

- `embed` 复制一份：`[2B,1,1536]`
- 构造空字符串 `""` 的 uncond 文本编码
- `cond = cat([cond_text, uncond_text], dim=0)` -> `[2B,S,1536]`

### 3.3 GPT 自回归采样（`Engine.run_gpt`）

默认长度：

- `max_new_tokens = num_encoder_latents = 1024`

主要张量：

- `embed_buffer: [2B, 1+1024, 1536]`（CFG 时）
- `kv_cache`：
  - dual-stream 层：`[2B, H, S + (1024+1), Hd]`
  - single-stream 层：`[2B, H, (1024+1), Hd]`

每步 `i=0..1023`：

1. `gpt_model(...)` 前向
2. 取当前步 logits 并裁剪到 shape vocab 区间 `[0, num_codes)`
3. CFG 融合：
   - `logits_cond, logits_uncond = chunk(2, dim=0)`
   - `gamma = guidance_scale * (max_new_tokens - i) / max_new_tokens`
   - `logits = (1+gamma)*cond - gamma*uncond`
4. token 选择：`process_logits(logits, top_p)`
5. 选出的 `next_id` 再 embedding，写回 `embed_buffer` 下一位置

最终得到：`output_ids: [B, 1024]`。

### 3.4 `process_logits` 细节（当前实现）

文件：`cube3d/inference/logits_postprocesses.py`

- `top_p is None`：argmax（确定性）
- `top_p is not None`：走 top-p + multinomial 采样

**代码现状注意点**：

- `process_logits` 内部当前调用是 `top_p_filtering(logits, top_p=0.9)`，即传入的 `top_p` 值没有被真正使用，而是固定 0.9。

### 3.5 `run_shape_decode`

- 从 `output_ids` 截取前 `num_encoder_latents`（1024）
- clamp 到 `[0, num_codes-1]`
- `shape_model.decode_indices -> latents`
- `shape_model.extract_geometry(use_warp=True)` -> `mesh_v_f`

`Engine.t2s()` 只是把 `run_gpt` 与 `run_shape_decode` 串联起来。

---

## 4. DualStreamRoformer 结构细节

文件：`cube3d/model/gpt/dual_stream_roformer.py`

### 4.1 词表与特殊 token

初始化时：

- 初始词表大小：`shape_model_vocab_size`（v0.5 为 `16384`）
- 依次追加三个特殊 token：
  - `shape_bos_id`
  - `shape_eos_id`
  - `padding_id`

最终 `vocab_size = 16387`。

### 4.2 双流与单流 block

- `dual_blocks`: `n_layer=23`
  - 最后一层设置 `cond_pre_only=True`
- `single_blocks`: `n_single_layer=1`

整体顺序：

1. 先过全部 dual blocks（`h` 与 `c` 交互）
2. 再过 single blocks（只处理 `h`）
3. `ln_f + lm_head` 输出 logits

### 4.3 RoPE 与位置编号

在 `forward()` 中分别计算两套频率：

- `s_freqs_cis`：给 single-stream（长度 `L`）
- `d_freqs_cis`：给 dual-stream（长度 `S+L`）

dual-stream 的位置构造：

- 条件流 `c` 位置全 0
- 形状流 `h` 位置为 `0..L-1`

### 4.4 注意力 mask

- 构造 `attn_mask = tril(ones(S+L, S+L))`
- decode 模式下按 `curr_pos_id` 截取对应行

在 KV cache decode 场景中，query 长度通常为 1。

---

## 5. 双流注意力层细节（`dual_stream_attention.py`）

### 5.1 `DismantledPreAttention`

- x 流：始终生成 `q,k,v`
- c 流：
  - 常规层生成 `q,k,v`
  - `cond_pre_only=True` 时只生成 `k,v`

q/k 做 `RMSNorm` 后进入注意力。

### 5.2 拼接策略

- prefill（非 decode）：
  - `q = cat(q_c, q_x)`（若 cond_pre_only 则只用 `q_x`）
  - `k = cat(k_c, k_x)`
  - `v = cat(v_c, v_x)`
- decode（启用 cache）：
  - 只计算当前 token 的 `q,k,v`（来自 x 流）
  - `k,v` 与缓存拼合后参与注意力

### 5.3 输出拆分

注意力输出 `y` 形状 `[B, T_total, D]`，再按长度切分：

- `y_c`（条件流）
- `y_x`（形状流）

`cond_pre_only` 的最后 dual block 中，条件流不再更新（`c=None`）。

### 5.4 Post-Attention 残差

- `x = x + c_proj(a_x)`
- `x = x + SwiGLUMLP(LN(x))`
- 条件流同理（若该层允许更新）

---

## 6. EngineFast（CUDA Graph）专门流程

文件：`cube3d/inference/engine.py` -> `EngineFast`

### 6.1 约束

- 仅支持 `cuda`
- `run_gpt` 只支持单 prompt（`assert len(prompts) == 1`）

### 6.2 warmup + graph capture

1. 用固定 warmup prompt：`"A cube"`
2. 分配持久化 buffer：
   - `embed_buffer`
   - `cond_buffer`
   - `kv_cache`
3. 先做 prefill，再做若干 decode warmup（默认 10 次）
4. 捕获单步 decode 前向到 `torch.cuda.CUDAGraph`

### 6.3 推理循环

- 第 0 token：走 eager prefill
- 第 1..1023 token：只 `graph.replay()`，从 `logits_buffer` 取结果
- 每步采样后把 next token embedding 写回 `embed_buffer`

该路径的核心收益来自固定形状缓存 + 图重放，代价是显存占用更高。

---

## 7. bbox 条件的完整链路

1. CLI 传入 `--bounding-box-xyz x y z`
2. `normalize_bbox` 把最长边缩放到 `1.925`
3. `Engine.prepare_inputs` 构造 `cond_bbox`
4. `prepare_conditions_with_bbox` 通过 `bbox_proj(3->1536)` 生成额外条件 token
5. 该 token 和文本 token 一起进入 dual-stream 条件流

未传 bbox 时，默认零向量，仍会生成一个 bbox 条件 token（如果模型配置启用了 `use_bbox`）。

---

## 8. v0.5 典型 shape/张量速查（`B=1`，CFG 开启）

- CLIP 文本 hidden：`[1, 77, 768]`
- 投影后 cond text：`[1, 77, 1536]`
- 加 bbox 后 cond：`[1, 78, 1536]`
- CFG 拼接后 cond：`[2, 78, 1536]`
- BOS embed：`[1, 1, 1536]`
- CFG 拼接后 embed：`[2, 1, 1536]`
- GPT 输出 token ids：`[1, 1024]`
- VQ lookup 后 z_q：`[1, 1024, 768]`
- decoder latents：`[1, 1024, 768]`
- occupancy 查询输入：`queries=[1, M, 3]`
- occupancy logits：`[1, M]`
- 网格体素：`[1, G, G, G]`，`G=2^resolution_base+1`

---

## 9. 与 v0.1 配置差异（代码层可见）

`cube3d/configs/open_model.yaml` 相比 v0.5：

- `num_encoder_latents = 512`（v0.5 为 1024）
- `gpt_model.use_bbox` 在 v0.1 配置未开启

因此 v0.5 的主要 pipeline 变化可归纳为：

- 更长 shape token 序列（更高几何表达容量）
- 条件流支持 bbox token

---

## 10. 代码现状注意事项（阅读源码可见）

1. `top_p` 参数目前在 `process_logits` 中被固定为 0.9 过滤阈值（不是传入值）。
2. `EngineFast` 显存需求明显高于 `Engine`，在 24GB 卡上也可能 OOM（项目 README 也有相关提示）。
3. `pymeshlab` 是可选依赖，缺失时不会阻断主流程，但会跳过 mesh 后处理。
4. `OneDDecoder.num_decoder_latents` 配置项在当前主推理路径中未直接驱动长度，实际长度由输入 token 序列（通常 1024）决定。

---

## 11. 一句话总览（端到端）

`prompt (+optional bbox)` -> `CLIP` -> `DualStreamRoformer autoregressive 1024 ids` -> `VQ codebook lookup + decoder latents` -> `occupancy field` -> `marching cubes` -> `mesh (.obj)`。
