from moviepy.editor import VideoFileClip
import os
import sys

# 定义日志文件夹路径
log_dir = 'video_splitter/logs'  # 你可以更改这个路径为任何有效路径

# 如果日志文件夹不存在，自动创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 定义日志文件路径
output_log_path = os.path.join(log_dir, 'output.log')
error_log_path = os.path.join(log_dir, 'error.log')

# 记录原始 stdout 和 stderr
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = open(output_log_path, 'w', encoding="utf-8")  # 记录正常日志
sys.stderr = open(error_log_path, 'w', encoding="utf-8")  # 记录错误日志

def split_video(video_path, output_folder, segment_length=30):
    """
    将视频按 segment_length 秒切割成多个片段，并存储到 output_folder。

    参数：
    video_path      - 输入视频文件路径
    output_folder   - 输出文件夹路径
    segment_length  - 每段视频的长度（秒）
    """
    try:
        video = VideoFileClip(video_path)
        duration = video.duration  # 获取视频总时长（秒）

        os.makedirs(output_folder, exist_ok=True)  # 确保输出文件夹存在

        segment_index = 1
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)  # 确保不会超出原视频长度
            output_path = os.path.join(output_folder, f"segment_{segment_index:03d}.mp4")

            # 剪辑视频
            subclip = video.subclip(start_time, end_time)
            subclip.write_videofile(output_path, codec="libx264", audio=True, audio_codec="aac")

            print(f"✅ 影片切割完成：{output_path}")
            segment_index += 1

        video.close()
        print("🎉 影片分割完成！")

    except Exception as e:
        # 发生错误时，打印错误信息并记录到日志
        print("❌ 处理过程中发生错误：", e)
        sys.stdout.write(f"❌ 错误: {e}\n")

    finally:
        # 关闭日志文件并恢复 stdout / stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        print("✅ 处理完成！")

if __name__ == "__main__":
    input_video = "D:/Vs.code/AI_Anchor/video_download/download/badminton.mp4"  # 替换成你的影片路径
    output_folder = "D:/Vs.code/AI_Anchor/video_splitter/badminton_segments"  # 输出文件夹
    split_video(input_video, output_folder)