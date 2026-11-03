# ComfyUI-PJ-SegmentSampler (PJ分段采样器)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-orange.svg)](https://github.com/comfyanonymous/ComfyUI)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-green.svg)]()
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-Zero_Extra-brightgreen.svg)]()

专门为 **MiniMax H3**、**LTXV** 及通用长视频**二采精修（高分辨率潜空间重采样 / 超分降噪）**设计的原生潜空间分段采样器节点。

---

## 📖 背景与痛点 (Motivation)

在现代视频扩散模型（如 MiniMax H3 混合精度模型、LTX-Video 等）的工作流中，“**一采生成低清整体结构 -> 3D潜空间放大 -> 二采高清分段重采样降噪**”是制作影视级高清视频的核心范式。

然而，当二采进入 **2.0MP（如 1088×1952）高清分辨率**，且视频长度达到 **10秒 ~ 15秒以上** 时，原生的全序列单次采样方案面临极其严峻的工程瓶颈：
1. **显存直接爆炸 (OOM)**：全序列送入 DiT 注意力层，峰值显存会瞬间突破 40GB+；
2. **传统方案画质断层**：以往的视频分段往往采用“先解码为图片 -> 裁切重采样 -> 再重新编码”的笨办法，多次经过 VAE 带来剧烈画质损耗、模糊与时间接缝闪烁；
3. **MiniMax H3 严苛的 RoPE 周期相位**：H3 模型具有严苛的 $17k + 5$ 帧率结构与 $(1, 4, 4, 4, 4)$ 周期性 RoPE 步进，随意裁切极易导致**口型脱节、音频漂移、画面突变或局部变成灰褐色噪点**；
4. **尾部 VAE 解码爆显存**：二采结束后如果 20GB DiT 主干模型仍然滞留 GPU，随后的高清全长 VAE 解码会瞬间触发 24GB 物理显存溢出（导致系统内存巨量借用或直接崩溃）。

`ComfyUI-PJ-SegmentSampler` 专为解决上述痛点而生，在**纯潜空间（Pure Latent Space）**中实现无损、相位自适应对齐的串行分段采样与平滑融合。

---

## ⚡ 核心技术特性 (Key Features)

### 1. 纯潜空间时空切片 (Pure Latent Temporal Slicing)
- **零编解码损耗**：100% 直接沿时间维度（Time / `dim=2`）精准切分视频 Latent 与 40Hz 音频 Latent；
- 采样完成后在潜空间中原地平滑无缝缝合，避免反复 VAE 编解码造成的画质衰减。

### 2. MiniMax H3 严苛相位对齐与 51 帧整数吸附 (`align_51_frames`)
- **严格遵循周期起点**：每个切片窗口起始步必为 5 的整倍数（$v_{start} \equiv 0 \pmod 5$），长度严格为 $5m + 2$；
- **51 帧吸附开关**：
  - `align_51_frames = True`（推荐用于对口型要求极高的影视级场景）：将重叠与窗口长度强制对齐至 $51m$ 视频帧（对应 15 潜空间时间步、85 音频潜空间步），确保每个分段的 RoPE 绝对相位恒定一致，彻底消灭口型漂移；
  - `align_51_frames = False`（经典连续切分模式）：严格遵循用户设定的秒数（如 8 秒 / 2 秒重叠），提供最灵活的自定义切分。

### 3. 重叠区全局噪声切片共享 (Global Noise Slice Sharing)
- 在分段采样开始前，为整个序列生成一张**统一样本全局随机噪声张量**；
- 各分段在各自的重叠时间区间内截取对应的全局噪声切片送入采样器；
- 相邻片段重叠区初态随机扰动 100% 相同，彻底杜绝分段接缝处的闪烁与随机跳变。

### 4. 余弦平滑归一化融合 (Cosine Cross-Fade Latent Blending)
- 重叠区域应用余弦交叉淡化（Cosine Weighting），在数学上满足任意时间点权重和严格恒等于 1.0（$\sum w(t) \equiv 1.0$）；
- 消除任何因能量叠加造成的重叠区过亮、发白或画面跳变。

### 5. 100% 原版连续音频无损直通 (`preserve_original_audio`)
- 音频波形对相位与连续性极度敏感。本节点默认开启原版音频直通，分段重采样仅针对视频画面做细节雕琢，**音频 100% 保留一采的高保真连续潜空间**，杜绝相位抵消与机械杂音。

### 6. 二采后自动卸载模型释放显存 (`unload_model_after_sampling`)
- 二采完成后，自动将 20GB DiT 主模型从显卡卸载至 CPU（调用 `model.detach()` 与 `soft_empty_cache`）；
- 使后续 260 帧 2.0MP 的 VAE 解码显存峰值从危险的 **23.9GB 暴降至 8.6GB ~ 9.5GB**，彻底杜绝爆显存崩溃。

### 7. 前端智能动态 UI (Dynamic Widget Visibility)
- 配套轻量级前端 JavaScript 扩展；
- 随着 `split_mode`（`auto_by_seconds` / `fixed_segments` / `custom_cuts`）的切换，自动收起与展开相关参数，UI 清爽直观。

---

## 🛠️ 安装方法 (Installation)

### 推荐方法：Git Clone

进入 ComfyUI 的 `custom_nodes` 目录并克隆本仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/pongjoo/ComfyUI-PJ-SegmentSampler.git
```

重启 ComfyUI 即可。

### 零额外依赖 (Zero Extra Dependencies)
- 本节点仅依赖 Python 标准库（`math`, `os`, `sys`, `logging`, `gc`）和 ComfyUI 原生模块（`torch`, `comfy.sample`, `comfy.model_management`, `comfy.nested_tensor`）；
- **无需执行任何 `pip install`**，开箱即用。

---

## 🎛️ 参数说明 (Parameters Reference)

| 参数名 | 类型 | 默认值 | 详细说明 |
| :--- | :--- | :--- | :--- |
| `noise` | `NOISE` | - | 噪声源（通常连接 `RandomNoise` 节点） |
| `guider` | `GUIDER` | - | 引导器（通常连接 `BasicGuider` 节点） |
| `sampler` | `SAMPLER` | - | 采样算法（推荐 `er_sde` 或 `euler`） |
| `sigmas` | `SIGMAS` | - | 调度器步长（通常连接 `BasicScheduler`，如 2 步 refine 降噪） |
| `latent_image` | `LATENT` | - | 输入待精修的潜空间（来自 3D 超分放大后的 AV Latent） |
| `split_mode` | `COMBO` | `auto_by_seconds` | 切分模式：<br>• `auto_by_seconds`：按秒自动规划分段<br>• `fixed_segments`：固定分段数<br>• `custom_cuts`：用户自定义切分范围 |
| `chunk_seconds` | `FLOAT` | `8.0` | 自动模式下单段时长（推荐 6.0 ~ 8.0 秒） |
| `overlap_seconds` | `FLOAT` | `2.0` | 相邻片段重叠过渡时长（推荐 1.5 ~ 2.5 秒） |
| `num_segments` | `INT` | `2` | 固定段数模式下的分段数量（2 ~ 32） |
| `custom_cuts` | `STRING` | `0-8, 6-14` | 自定义时间切片范围，用逗号分隔（秒） |
| `align_51_frames` | `BOOLEAN` | `False` | **51帧整数吸附**：<br>• `True`：对齐至 51 帧/85 音频 Token 整数网格（RoPE 相位绝对一致，极高口型一致性）<br>• `False`：经典连续切分，严格尊重用户设定的秒数 |
| `blend_mode` | `COMBO` | `cosine` | 重叠区平滑算法：`cosine`（余弦平滑，推荐）、`linear`（线性）、`hard_cut`（硬切） |
| `fps` | `FLOAT` | `24.0` | 视频帧率（MiniMax H3 官方基准为 24fps） |
| `preserve_original_audio`| `BOOLEAN` | `True` | **100% 直通保留一采原音频**（推荐开启，避免音频相位抵消） |
| `unload_model_after_sampling`| `BOOLEAN`| `True` | **二采后自动卸载模型**（推荐开启，释放 20GB 显存，保卫后续 VAE 解码） |

---

## 🔌 工作流节点连接 (Workflow Wiring)

`PJ_SegmentSampler` 是官方 `SamplerCustomAdvanced`（自定义采样器）的**完全即插即用升级替代品**：

```
[RandomNoise] ─────────► noise ────────┐
[BasicGuider] ─────────► guider ───────┤
[KSamplerSelect] ──────► sampler ──────┼──► [PJ_SegmentSampler] ──► output ──► [VAEDecode] (视频解码)
[BasicScheduler] ──────► sigmas ───────┤                               └──► [VAEDecodeAudio] (音频解码)
[3D Upscaled Latent] ──► latent_image ─┘
```

---

## 📈 实测显存与性能表现 (Benchmarks)

在单卡 **NVIDIA GeForce RTX 5090 D (24GB VRAM)** / RTX 4090 上实测 **260 帧（10.8 秒）、2.0 MP（1088×1952 高清）** 全流程：

| 执行阶段 | 潜空间分辨率 | 显存峰值 | 状态评估 |
| :--- | :--- | :--- | :--- |
| **一采低清生成** | 0.4 MP (544×960) | **16.8 ~ 17.1 GB** | 🟢 极度宽裕 |
| **3D 潜空间超分** | 30×54 ➔ 68×122 | **12.5 ~ 14.2 GB** | 🟢 超分完成立即释放回 CPU |
| **二采分段精修 (本节点)** | **2.0 MP (1088×1952)** | **20.5 ~ 20.8 GB** | 🟢 **压满 24GB 极致性能，0 内存溢出** |
| **二采后 VAE 解码** | 260 帧 2.0 MP | **8.6 ~ 9.5 GB** | 🟢 **得益于自动卸载，彻底消除 OOM 风险** |

---

## 📄 开源许可证 (License)

本项目基于 [MIT License](LICENSE) 许可协议开源。
欢迎 Star、Fork 与提 Issue！
