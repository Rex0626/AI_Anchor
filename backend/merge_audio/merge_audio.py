import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

# ✅ 將時間字串轉為秒數
def time_str_to_seconds(time_str):
    """將 H:MM:SS.f 時間碼轉換為秒數 (支持浮點數)。"""
    try:
        parts = time_str.split(':')
        seconds = 0.0
        if len(parts) == 3: # H:MM:SS.f
            seconds += float(parts[0]) * 3600
            seconds += float(parts[1]) * 60
            seconds += float(parts[2])
        elif len(parts) == 2: # MM:SS.f
            seconds += float(parts[0]) * 60
            seconds += float(parts[1])
        elif len(parts) == 1: # SS.f
             seconds += float(parts[0])
        return seconds
    except ValueError:
        print(f"❌ time_str_to_seconds 轉換錯誤，輸入值: {time_str}")
        return 0.0

# ✅ 單段影片合成
def merge_segment_video_with_audio(video_path, json_path, tts_dir, output_path, audio_delay=0.3):
    print(f"\n🎬 合併影片片段：{os.path.basename(video_path)}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video = VideoFileClip(video_path)
    commentary = data.get("commentary", [])
    if not commentary:
        print(f"⚠️ 沒有旁白內容：{json_path}")
        return {"status": "skip", "segment": video_path, "reason": "no_commentary"}

    segment_name = os.path.splitext(os.path.basename(video_path))[0]
    segment_tts_folder = os.path.join(tts_dir, segment_name)
    if not os.path.exists(segment_tts_folder):
        print(f"⚠️ 沒有找到語音資料夾：{segment_tts_folder}")
        return {"status": "skip", "segment": video_path, "reason": "tts_missing"}

    audio_clips = []

    for idx, sentence in enumerate(commentary):
        start_time = time_str_to_seconds(sentence["start_time"])
        end_time = time_str_to_seconds(sentence["end_time"])

        adjusted_start = max(0, min(start_time + audio_delay, end_time - 0.2))

        voice_file = None
        for f in os.listdir(segment_tts_folder):
            if f.startswith(f"{idx+1:03d}_") and f.endswith(".mp3"):
                voice_file = os.path.join(segment_tts_folder, f)
                break

        if not voice_file or not os.path.exists(voice_file):
            print(f"❌ 缺漏語音檔案：{idx+1:03d} @ {segment_name}")
            continue

        try:
            audio_clip = AudioFileClip(voice_file).set_start(adjusted_start)
            audio_clips.append(audio_clip)
        except Exception as e:
            print(f"❌ 載入語音失敗：{voice_file}，錯誤：{e}")

    if not audio_clips:
        print("❌ 沒有可用語音片段，跳過：", segment_name)
        return {"status": "skip", "segment": video_path, "reason": "no_audio_clips"}

    final_audio = CompositeAudioClip(audio_clips)
    video = video.set_audio(final_audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"✅ 合併完成：{output_path}")

    return {"status": "success", "segment": video_path, "output": output_path}

# ✅ 批次處理所有影片片段
def batch_merge_all_segments(video_folder, json_folder, tts_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    results = []
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    for file in sorted(video_files):
        base_name = os.path.splitext(file)[0]
        video_path = os.path.join(video_folder, file)
        json_path = os.path.join(json_folder, base_name + ".json")
        output_path = os.path.join(output_folder, base_name + "_final.mp4")

        if not os.path.exists(json_path):
            print(f"⚠️ 找不到對應 JSON：{json_path}")
            results.append({"status": "skip", "segment": video_path, "reason": "json_missing"})
            continue

        result = merge_segment_video_with_audio(video_path, json_path, tts_folder, output_path)
        results.append(result)

    return {"status": "done", "results": results}

# ✅ 主程式
if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/batch_badminton_outputs"
    tts_folder = "D:/Vs.code/AI_Anchor/backend/TextToSpeech/final_tts_google"
    output_folder = "D:/Vs.code/AI_Anchor/backend/merge_audio/final_output_videos"

    result = batch_merge_all_segments(video_folder, json_folder, tts_folder, output_folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
