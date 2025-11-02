import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip

# ✅ 時間字串轉秒數
def time_str_to_seconds(time_str):
    parts = list(map(int, time_str.strip().split(":")))
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    elif len(parts) == 2:
        return parts[0]*60 + parts[1]
    return 0


# ✅ 合併單一片段
def merge_segment_video_with_audio(video_path, json_path, tts_dir, output_path, audio_delay=0.3):
    print(f"\n🎬 合併影片片段：{os.path.basename(video_path)}")

    # 讀取 Gemini 輸出的 JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    video = VideoFileClip(video_path)
    commentary = data.get("commentary", [])
    if not commentary:
        print(f"⚠️ 沒有旁白內容：{json_path}")
        return None

    segment_name = os.path.splitext(os.path.basename(video_path))[0]
    segment_tts_folder = os.path.join(tts_dir, segment_name)
    if not os.path.exists(segment_tts_folder):
        print(f"⚠️ 沒有找到語音資料夾：{segment_tts_folder}")
        return None

    audio_clips = []

    # ✅ 支援新版命名規則（001.mp3、002.mp3）
    for idx, sentence in enumerate(commentary):
        start_time = time_str_to_seconds(sentence["start_time"])
        end_time = time_str_to_seconds(sentence["end_time"])
        adjusted_start = max(0, min(start_time + audio_delay, end_time - 0.2))

        # 優先找簡化命名（001.mp3）
        voice_file = os.path.join(segment_tts_folder, f"{idx+1:03d}.mp3")

        # 若找不到，再退而求其次找舊格式（001_情緒.mp3）
        if not os.path.exists(voice_file):
            for f in os.listdir(segment_tts_folder):
                if f.startswith(f"{idx+1:03d}_") and f.endswith(".mp3"):
                    voice_file = os.path.join(segment_tts_folder, f)
                    break

        if not os.path.exists(voice_file):
            print(f"❌ 缺漏語音檔案：{idx+1:03d} @ {segment_name}")
            continue

        try:
            audio_clip = AudioFileClip(voice_file).set_start(adjusted_start)
            audio_clips.append(audio_clip)
        except Exception as e:
            print(f"⚠️ 無法載入語音檔 {voice_file}：{e}")

    if not audio_clips:
        print(f"❌ 沒有可用語音片段，跳過：{segment_name}")
        return None

    final_audio = CompositeAudioClip(audio_clips)
    video = video.set_audio(final_audio)
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")
    print(f"✅ 合併完成：{output_path}")

    return output_path


# ✅ 批次處理所有片段
def batch_merge_all_segments(video_folder, json_folder, tts_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    last_output_path = None
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

    for file in sorted(video_files):
        base_name = os.path.splitext(file)[0]
        video_path = os.path.join(video_folder, file)
        json_path = os.path.join(json_folder, base_name + ".json")
        output_path = os.path.join(output_folder, base_name + "_final.mp4")

        if not os.path.exists(json_path):
            print(f"⚠️ 找不到對應 JSON：{json_path}")
            continue

        result = merge_segment_video_with_audio(video_path, json_path, tts_folder, output_path)
        if result:
            last_output_path = result

    return last_output_path
