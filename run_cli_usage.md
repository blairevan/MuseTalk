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

## ⚙️ How it works internally (内部运行机制)

1. **Input Validation**: It validates whether the face and audio paths exist, and checks for unsupported files.
2. **Auto YAML Generation**: It generates a temporary yaml task file at `configs/inference/temp_cli_task.yaml`.
3. **Task Launching**: It invokes the underlying inference script `scripts.inference` with the designated weights path.
4. **Cleanup**: Finally, it automatically cleans up the generated temporary configurations.
