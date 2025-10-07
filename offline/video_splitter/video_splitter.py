from moviepy.editor import VideoFileClip
import os
import sys

def split_video(video_path, output_folder, segment_length=30, use_log=False):
    """
    將影片按 segment_length 秒切割成多個片段，並儲存到 output_folder。 
    回傳結果包含每段影片路徑。 

    use_log=True 時，會寫入 logs 資料夾；Flask 呼叫時建議使用 False。
    """
    results = []
    log_dir = 'video_splitter/logs'
    if use_log:
        os.makedirs(log_dir, exist_ok=True)
        sys.stdout = open(os.path.join(log_dir, 'output.log'), 'w', encoding="utf-8")
        sys.stderr = open(os.path.join(log_dir, 'error.log'), 'w', encoding="utf-8")

    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        os.makedirs(output_folder, exist_ok=True)

        segment_index = 1
        for start_time in range(0, int(duration), segment_length):
            end_time = min(start_time + segment_length, duration)
            output_path = os.path.join(output_folder, f"segment_{segment_index:03d}.mp4")

            subclip = video.subclip(start_time, end_time)
            subclip.write_videofile(output_path, codec="libx264", audio=True, audio_codec="aac")

            results.append(output_path)
            segment_index += 1

        video.close()
        return {"status": "success", "segments": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        if use_log:
            sys.stdout.close()
            sys.stderr.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

# ✅ 后端测试模式（直接运行这个文件时才会跑）
if __name__ == "__main__":
    input_video = "D:/Vs.code/AI_Anchor/video_download/download/badminton.mp4"
    output_folder = "D:/Vs.code/AI_Anchor/video_splitter/badminton_segments"
    result = split_video(input_video, output_folder, segment_length=30, use_log=True)
    print(result)
