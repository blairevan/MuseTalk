# MuseTalk CLI Wrapper (`run_cli.py`) Usage Guide

`run_cli.py` is a convenient command-line interface (CLI) wrapper for running single inference jobs in MuseTalk without manually writing config YAML files. It automatically handles the generation of temporary task configurations and schedules model runs.

---

## 🚀 Quick Start (快速开始)

To run inference using the CLI wrapper, use the following template:

```bash
python run_cli.py --face <path_to_face> --audio <path_to_audio> --outfile <path_to_output_mp4> [Optional Arguments]
```

### Example (使用示例)

```bash
python run_cli.py \
  --face ./data/video/select_ref.mp4 \
  --audio ./data/audio/monalisa.wav \
  --outfile ./output/result.mp4 \
  --model_version v1.5 \
  --bbox_shift 0 \
  --parsing_mode jaw
```

---

## 📋 Arguments Reference (参数详解)

### 1. Required Arguments (必填参数)

| Argument (参数名) | Type (类型) | Description (说明) |
| :--- | :--- | :--- |
| `--face` | `str` | Path to the input face video, image file (`.jpg`, `.png`, etc.), or directory containing images. (输入的人脸视频、图像文件或图像目录的路径) |
| `--audio` | `str` | Path to the input audio file (`.wav`, `.mp3`, etc.). (输入音频文件的路径) |
| `--outfile` | `str` | Output video path. Must be a specific path ending with `.mp4`. (输出视频文件的保存路径，必须以 `.mp4` 结尾) |

---

### 2. General Model Arguments (通用/模型参数)

| Argument (参数名) | Type (类型) | Default (默认值) | Choices (候选值) | Description (说明) |
| :--- | :--- | :--- | :--- | :--- |
| `--model_version` | `str` | `"v1.5"` | `["v1.0", "v1.5"]` | Select the MuseTalk model version. (选择 MuseTalk 模型版本) |
| `--resolution` | `int` | `auto` | `[256, 512]` | Generation resolution. Auto-selects `512` when `v1.5` and `./models/musetalkV15_512/{unet.pth,musetalk.json}` exist; otherwise falls back to `256`. Pass `--resolution 256` to force legacy 256. (生成分辨率；本地存在 512 权重时自动启用 `512`，否则回退 `256`) |
| `--unet_model_path` | `str` | `auto` | | Optional override for UNet weights path. (可选自定义 UNet 权重路径) |
| `--unet_config` | `str` | `auto` | | Optional override for UNet config JSON path. (可选自定义 UNet 配置路径) |
| `--batch_size` | `int` | `auto` | | Inference batch size (`2` for 512, `8` for 256 unless explicitly specified). (推理批大小；512 默认 `2`，256 默认 `8`) |
| `--use_float16` / `--no_float16` | `flag` | `auto` | | FP16 toggle (`enabled` by default for 512 to stay within 12 GB VRAM; `disabled` by default for 256). Pass `--no_float16` to force FP32 at 512. (FP16 开关；512 默认开启防显存溢出，256 默认关闭) |

---

### 3. Face Bounding Box & Crop Tuning (人脸边界框与裁剪微调)

These parameters control the face area detected and processed by MuseTalk. Tuning them can significantly affect lip openness and facial blend results.

| Argument (参数名) | Type (类型) | Default (默认值) | Description (说明) |
| :--- | :--- | :--- | :--- |
| `--bbox_shift` | `int` | `0` | Bounding box vertical shift in pixels. Positive values shift down (increases mouth openness), negative values shift up (decreases mouth openness). (人脸边界框的垂直偏移量，单位为像素。正值向下移动增加嘴部开合度，负值向上移动减少嘴部开合度) |
| `--extra_margin` | `int` | `8` | Extra margin for face cropping. (人脸裁剪的额外页边距/边缘留白) |
| `--bbox_left_ratio` | `float` | `0.0` | Adjust ratio for the left border of bounding box; positive expands outward, negative shrinks inward. (左侧人脸框调整比例，正数向外扩展，负数向内收缩) |
| `--bbox_right_ratio` | `float` | `0.0` | Adjust ratio for the right border of bounding box. (右侧人脸框调整比例) |
| `--bbox_top_ratio` | `float` | `0.0` | Adjust ratio for the top border of bounding box. (顶部人脸框调整比例) |
| `--bbox_bottom_ratio` | `float` | `0.0` | Adjust ratio for the bottom border of bounding box. (底部人脸框调整比例) |

---

### 4. Facial Blend & Mask Control (面部融合与掩码控制)

These parameters affect how the generated mouth area is blended back into the original video frame.

| Argument (参数名) | Type (类型) | Default (默认值) | Choices (候选值) | Description (说明) |
| :--- | :--- | :--- | :--- | :--- |
| `--parsing_mode` | `str` | `"jaw"` | `["jaw", "raw"]` | Face blending parsing mode. (面部融合解析模式) |
| `--left_cheek_width` | `int` | `120` | | Width of left cheek editing region. (左脸颊编辑区域的宽度) |
| `--right_cheek_width` | `int` | `120` | | Width of right cheek editing region. (右脸颊编辑区域的宽度) |
| `--side_protect_ratio` | `float` | `0.08` | | Side-edge blend protection ratio. (侧边融合边缘保护比例) |
| `--bbox_smooth_window` | `int` | `7` | | Centered moving-average window size for bounding box smoothing; Set to `1` to disable smoothing. (用于 bounding box 平滑的中心移动平均窗口大小；设为 `1` 将禁用平滑) |

---

### 5. Clarity & Pause Options (清晰度与停顿选项)

The first four options below are **on by default** since 2026-11-20, after 4K A/B runs on an RTX 5070 (lip sharpness +13% to +24%, no extra run time, upper face and glasses untouched). Pass the legacy value to reproduce the previous output. The remaining options stay off. (前四项自 2026-11-20 起默认开启，依据为 RTX 5070 上两条 4K 底板的 A/B：唇部锐度提升 13%~24%、耗时不变、上半脸与镜框不受影响。如需旧版输出，传入括号内的旧值；其余选项仍默认关闭。)

| Argument (参数名) | Type (类型) | Default (默认值) | Description (说明) |
| :--- | :--- | :--- | :--- |
| `--vae_latent_mode` | `str` | `"mode"` | `mode` is deterministic and matches training; legacy `sample` draws from the VAE posterior. (VAE 编码方式，旧值 `sample`) |
| `--paste_interpolation` | `str` | `"lanczos"` | `lanczos`: area when shrinking, Lanczos when enlarging; legacy `linear`. (回贴缩放插值，旧值 `linear`) |
| `--enhance_mode` | `str` | `"usm"` | `usm` sharpens the lip area on the luma channel; legacy `none`. (唇齿局部锐化，旧值 `none`) |
| `--protect_upper_face` | `bool` | `true` | Keep exact source pixels above the blend boundary; legacy `false`. (上半脸严格保持原帧像素，旧值 `false`) |
| `--usm_amount` | `float` | `-1.0` | Sharpening amount; negative picks it from how much the face is enlarged. (锐化强度，负数为自动) |
| `--fix_teeth_whiteness` | `bool` | `false` | Tone down over-bright teeth inside the parsed mouth interior; about +5% run time. (嘴内亮区修正，耗时约 +5%) |
| `--close_mouth_pauses` | `bool` | `false` | Fade to the neutral (first-frame) mouth during mid-sentence pauses. Known issue: the chin shifts when the speaker's head has moved since frame 0. (句中停顿渐变闭嘴；已知问题：讲者头部相对第 0 帧有移动时下巴会错位) |
| `--pause_min_frames` | `int` | `8` | Shortest pause, in frames, that closes the mouth. (触发闭嘴的最短停顿帧数) |
| `--pause_fade_frames` | `int` | `4` | Frames used for each fade in and out. (渐入、渐出各用的帧数) |
| `--pause_top_db` | `float` | `38.0` | Silence threshold below the audio peak, in dB; raise it for audio with background noise or music. (停顿检测门限，相对峰值的分贝数；有底噪或背景音乐时调大) |
| `--use_audio_pe` | `bool` | `auto` | Add positional encoding to Whisper features (`true` for 256, `false` for 512 unless explicitly set). (音频位置编码；256 默认 `true`，512 默认 `false`) |
| `--ref_frame_mode` | `str` | `auto` | UNet reference latent source: `self`, `first`, `offset`, or `auto` (DWPose 68-pt closed-mouth frontal selection). Defaults to `auto` for 512 and `self` for 256. (参考帧策略；512 默认 `auto` 自动检测正脸闭嘴帧，256 默认 `self`) |

Reproduce the previous (pre-2026-11-20) output (复现旧版输出):

```bash
python run_cli.py --face ./data/video/select_ref.mp4 --audio ./data/audio/monalisa.wav \
  --outfile ./output/result_legacy.mp4 \
  --vae_latent_mode sample --paste_interpolation linear --enhance_mode none \
  --protect_upper_face false
```

The final audio mux now copies the rendered video stream instead of re-encoding it, so the output keeps the single CRF 18 encode; files are larger than before. (合音轨改为直接拷贝视频流，避免二次有损编码；输出文件会比之前大。)

---

## ⚙️ How it works internally (内部运行机制)

1. **Input Validation**: It validates whether the face and audio paths exist, and checks for unsupported files.
2. **Auto YAML Generation**: It generates a private temporary task file `configs/inference/temp_cli_task_*.yaml` (unique per run, so concurrent runs do not overwrite each other).
3. **Task Launching**: It invokes the underlying inference script `scripts.inference` with the designated weights path.
4. **Cleanup**: Finally, it automatically cleans up the generated temporary configurations.

---

## 修订记录

| 修订时间 | 修订摘要 |
| --- | --- |
| 2026-11-19 21:48:00 +08:00 | 新增第 5 节：记录清晰度与句中停顿相关的 10 个参数（默认值均保持旧版输出）及全开对照示例。此前版本无修订记录。 |
| 2026-11-20 17:20:00 +08:00 | 依据 RTX 5070 两条 4K 底板 A/B 结果，将 `vae_latent_mode`、`paste_interpolation`、`enhance_mode`、`protect_upper_face` 改为默认开启；补充旧版复现命令、`close_mouth_pauses` 已知问题，以及合音轨改为拷贝视频流的说明。 |
| 2026-11-20 17:33:00 +08:00 | 新增 `--pause_top_db` 参数说明（此前底层脚本支持但包装入口未透传）。实测输出视频码率约由 2.5 Mbps 升至 4.6 Mbps（4K、58 s 样例文件 18.6 MB → 34.4 MB），系去掉二次编码所致。 |
| 2026-11-20 17:52:00 +08:00 | “内部运行机制”更新：每次运行使用唯一的临时任务文件 `temp_cli_task_*.yaml`；底层推理默认不再在输出目录的上级写人脸坐标文件（仅 `--saved_coord` 时写入）。 |
| 2026-11-24 00:29:00 +08:00 | 补充 `--resolution`、`--unet_model_path`、`--unet_config`、`--batch_size`、`--use_float16`/`--no_float16` 与 `--ref_frame_mode` 说明：`v1.5` 检测到本地 `./models/musetalkV15_512/{unet.pth,musetalk.json}` 时默认自动启用 `512` 并联动安全默认值（`batch_size=2`、`use_float16=true`、`use_audio_pe=false`、`ref_frame_mode=auto`），无 512 权重或显式 `--resolution 256` 时保持旧版 `256` 行为。 |
