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

    try:
        video = VideoFileClip(video_path)
        video_duration = video.duration 
    except Exception as e:
        print(f"❌ 無法讀取影片檔：{video_path}，錯誤：{e}")
        return {"status": "error", "segment": video_path, "reason": "video_load_error"}

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

    # 取得該資料夾下所有 mp3 檔案
    all_files = os.listdir(segment_tts_folder)

    for idx, sentence in enumerate(commentary):
        start_time = time_str_to_seconds(sentence["start_time"])
        end_time = time_str_to_seconds(sentence["end_time"])

        adjusted_start = max(0, min(start_time + audio_delay, video_duration - 0.1))
        allowed_duration = end_time - start_time
        
        if idx == len(commentary) - 1:
            allowed_duration = video_duration - adjusted_start

        # 🔍 [核心修正] 根據圖片格式尋找檔案
        # JSON idx 是 0 -> 檔案是 001_xxx.mp3
        # JSON idx 是 1 -> 檔案是 002_xxx.mp3
        target_prefix = f"{idx + 1:03d}" # 格式化為 001, 002...
        
        voice_file_name = None
        for f in all_files:
            # 只要檔名是以 "001" 開頭且是 mp3 就匹配 (忽略後面的情緒文字)
            if f.startswith(target_prefix) and f.endswith(".mp3"):
                voice_file_name = f
                break

        if not voice_file_name:
            print(f"⚠️ 找不到對應音檔 (預期開頭: {target_prefix}) @ {segment_name}")
            continue

        voice_file_path = os.path.join(segment_tts_folder, voice_file_name)

        try:
            # 載入音檔 (使用不同變數名稱避免混淆)
            clip_to_add = AudioFileClip(voice_file_path)
            
            # 截斷邏輯
            if clip_to_add.duration > (allowed_duration + 0.1):
                print(f"   ✂️ [截斷] {voice_file_name}: {clip_to_add.duration:.2f}s -> {allowed_duration:.2f}s")
                clip_to_add = clip_to_add.subclip(0, allowed_duration)
            
            # 設定開始時間
            clip_to_add = clip_to_add.set_start(adjusted_start)
            audio_clips.append(clip_to_add)
            
        except Exception as e:
            print(f"❌ 處理音檔失敗：{voice_file_name}，錯誤：{e}")

    if not audio_clips:
        print("❌ 沒有可用語音片段，跳過：", segment_name)
        return {"status": "skip", "segment": video_path, "reason": "no_audio_clips"}

    # 合成
    try:
        final_audio = CompositeAudioClip(audio_clips)
        video = video.set_audio(final_audio)
        video.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None) # logger=None 減少輸出雜訊
        print(f"✅ 合併完成：{output_path}")
    except Exception as e:
        print(f"❌ 寫入影片失敗：{output_path}，錯誤：{e}")
        return {"status": "error", "segment": video_path, "reason": "write_error"}

    return {"status": "success", "segment": video_path, "output": output_path}

# ✅ 批次處理所有影片片段
def batch_merge_all_segments(video_folder, json_folder, tts_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    results = []
    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    
    if not video_files:
        print(f"❌ 錯誤：在 {video_folder} 找不到任何 .mp4 影片")
        return {"status": "error", "reason": "no_videos_found"}

    for file in sorted(video_files):
        base_name = os.path.splitext(file)[0]
        video_path = os.path.join(video_folder, file)
        json_path = os.path.join(json_folder, base_name + ".json")
        output_path = os.path.join(output_folder, base_name + "_final.mp4")

        if not os.path.exists(json_path):
            print(f"⚠️ 跳過 (無 JSON)：{base_name}")
            results.append({"status": "skip", "segment": video_path, "reason": "json_missing"})
            continue

        result = merge_segment_video_with_audio(video_path, json_path, tts_folder, output_path)
        results.append(result)

    return {"status": "done", "results": results}

# ✅ 主程式
if __name__ == "__main__":
    # 路徑設定 (請確保這些路徑存在)
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"
    tts_folder = "D:/Vs.code/AI_Anchor/backend/TextToSpeech/final_tts_google"
    output_folder = "D:/Vs.code/AI_Anchor/backend/merge_audio/final_output_videos"

    print("🚀 開始執行音訊合併...")
    result = batch_merge_all_segments(video_folder, json_folder, tts_folder, output_folder)
    
    # 簡單輸出結果統計
    success_count = sum(1 for r in result["results"] if r["status"] == "success")
    print(f"\n🏁 處理結束。成功合併：{success_count} / {len(result['results'])}")