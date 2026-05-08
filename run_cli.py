import argparse
import os
import yaml
import subprocess

def main():
    parser = argparse.ArgumentParser(description="MuseTalk CLI Wrapper")
    parser.add_argument("--face", type=str, required=True, help="Input face video path")
    parser.add_argument("--audio", type=str, required=True, help="Input audio path")
    parser.add_argument("--outfile", type=str, required=True, help="Output video path (must be a specific .mp4 file)")
    parser.add_argument("--version", type=str, default="v1.5", choices=["v1.0", "v1.5"])
    
    args = parser.parse_args()

    # 将输入的路径转化为绝对路径，确保 MuseTalk 底层能够正确识别
    face_path = os.path.abspath(args.face)
    audio_path = os.path.abspath(args.audio)
    outfile_path = os.path.abspath(args.outfile)

    # MuseTalk 默认是通过 yaml 配置文件来执行批量任务的，为了支持单次 CLI 调用，这里动态生成一个临时的 yaml
    config_dict = {
        "task_0": {
            "video_path": face_path,
            "audio_path": audio_path
        }
    }
    
    temp_yaml = os.path.abspath("configs/inference/temp_cli_task.yaml")
    os.makedirs(os.path.dirname(temp_yaml), exist_ok=True)
    with open(temp_yaml, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)
        
    # 提取输出目录
    out_dir = os.path.dirname(outfile_path)
    os.makedirs(out_dir, exist_ok=True)

    # 根据选定的版本，自动组装底层的模型路径
    if args.version == "v1.0":
        unet_model_path = "./models/musetalk/pytorch_model.bin"
        unet_config = "./models/musetalk/musetalk.json"
        version_arg = "v1"
    else:
        unet_model_path = "./models/musetalkV15/unet.pth"
        unet_config = "./models/musetalkV15/musetalk.json"
        version_arg = "v15"

    # 组装最终的底层调用命令
    cmd = [
        "python", "-m", "scripts.inference",
        "--inference_config", temp_yaml,
        "--result_dir", out_dir,
        "--output_vid_name", outfile_path,  # 传入绝对路径，底层代码 os.path.join 遇到绝对路径会自动使用绝对路径
        "--unet_model_path", unet_model_path,
        "--unet_config", unet_config,
        "--version", version_arg
    ]
    
    print(f"🚀 开始执行数字人合成任务...\n底层调用命令: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 任务圆满完成！\n视频已保存至: {outfile_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 推理过程中发生错误。")
    finally:
        # 清理临时的 yaml 配置文件
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)

if __name__ == "__main__":
    main()
