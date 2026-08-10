# MuseTalk 任务交接记录

> 本文件用于记录当前任务状态、验证边界和后续工作。后续修改不得删除历史内容；如需更正旧信息，应保留原记录并在文末追加修订记录。

## 当前任务

审查并修复 MuseTalk 推理链中的三个问题：

1. 多任务推理时，上一任务的生成模型与下一任务的 landmark/face detection 模型发生显存重叠。
2. 普通推理和实时推理的 Ping-Pong 周期重复首帧，循环边界出现连续相同帧。
3. 全静音音频经过 `librosa.effects.trim()` 后可能被误认为整段有效音频。

同时明确首尾静音嘴型策略：

- `close_mouth_end=true`：结尾静音区使用首帧闭嘴区域覆盖。
- `close_mouth_end=false`：不使用首帧嘴型特殊处理，依赖视频生成模型自然处理。
- `close_mouth_end` 默认值保持 `false`。

## 已完成内容

### 本次修复

- `musetalk/utils/audio_utils.py`
  - 在 `librosa.effects.trim()` 前增加绝对峰值阈值判断，默认阈值为 `1e-4`。
  - 空音频或全静音音频直接返回 `(0, 0)`。
- `musetalk/utils/sequence_utils.py`
  - 新增共享的 `build_ping_pong_cycle()`。
  - 周期从 `0,1,2,3,2,1,0` 修正为 `0,1,2,3,2,1`，避免首帧重复。
- `scripts/inference.py`
  - 接入共享 Ping-Pong 周期函数。
  - 每个任务通过 `finally` 释放 VAE、UNet、PE、Whisper、FaceParsing 和中间 GPU 张量。
- `scripts/realtime_inference.py`
  - 接入共享 Ping-Pong 周期函数。
- `musetalk/utils/resource_utils.py`
  - 新增 best-effort Torch 资源清理函数：模型迁回 CPU、执行垃圾回收并清理 CUDA cache。
- 测试覆盖：
  - 全静音返回 `(0, 0)`。
  - Ping-Pong 周期不重复端点，并覆盖空列表、单元素列表。
  - Torch 资源全部清理，以及单个资源清理失败时继续处理后续资源。

### 已验证内容

- `python3 -m unittest discover -s tests`：15 个测试通过。
- 已通过相关 Python 文件的 `py_compile` 语法检查。
- `python3 run_cli.py --help` 已确认 `close_mouth_start`、`close_mouth_end`、`batch_size`、`use_float16` 参数可见。
- `git diff --check` 通过。
- 未创建本次修复提交；当前修复保留在工作区未提交 diff 中。

## 卡住的问题 / 验证限制

当前没有代码实现层面的硬阻塞，但仍有以下验收缺口：

1. 尚未在真实 GPU 环境运行包含多个 task 的完整推理，模型释放逻辑的实际显存峰值尚未量化。
2. 尚未使用真实音频生成视频，未完成首部/尾部静音的画面连续性和嘴型听感验收。
3. 当前本地轻量验证环境缺少完整 Torch/目标音频运行依赖，因此资源和音频测试使用了最小替身验证纯逻辑；目标运行环境仍需安装项目 requirements。
4. 全静音返回 `(0, 0)` 后，是否让整段音频闭嘴由 `close_mouth_end` 开关决定：开关为 `false` 时仍按约定交给生成模型处理。

## 下一步计划

1. 在具备完整模型和 GPU 的环境中运行 `configs/inference/test.yaml` 的多任务推理，记录 landmark 阶段、生成阶段和任务切换时的显存峰值。
2. 使用带有首部/尾部静音的真实音频分别测试 `close_mouth_end=true` 与 `false`，检查首帧嘴型覆盖和模型自然生成两条路径。
3. 进行最终视频播放验收，重点检查 Ping-Pong 周期边界是否仍有停顿、跳帧或姿态反转。
4. 若验收通过，再由用户决定是否提交当前工作区改动。

## 踩过的坑与注意事项

- `librosa.effects.trim()` 是相对峰值检测；对全零波形不能单独依赖它，必须先做绝对峰值判断。
- Ping-Pong 反向切片若使用 `[-2::-1]`，会把索引 `0` 再追加一次，下一周期又从 `0` 开始，形成重复首帧。
- 多任务循环中仅调用 `torch.cuda.empty_cache()` 不够；仍被 Python 引用的模型和 GPU 张量不会释放，必须先迁移/断开引用，再清理 cache。
- 首尾静音闭嘴只能局部融合首帧闭嘴区域，不能用完整首帧替换当前帧，否则会造成背景、头部或身体姿态跳变。
- `librosa` 和 `torch` 属于运行时重依赖，工具模块可采用延迟导入以便纯函数测试，但不能据此认为完整推理环境不需要这些依赖。
- 当前分支在本次修复期间被外部推进到提交 `c3d862b`；后续操作应先确认新的 HEAD 和工作区状态，避免误覆盖其他人的提交或未提交修改。

## 修订记录

### 2026-08-10 10:08 +0800

- **修改小结**：创建初始 `HANDOFF.md`，记录当前修复任务、已完成的代码与测试、GPU/视觉验收限制、下一步计划和历史踩坑。
- **当前基线**：分支 HEAD 为 `c3d862b`；本次修复相关文件仍有未提交改动和新增测试/工具文件。
