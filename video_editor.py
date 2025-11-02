#音訊合併至影片
import os
import json
import numpy as np 
from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips, AudioArrayClip
import main

# 從 main.py 導入設定
PLAN_FILE = main.PLAN_FILE
VIDEO_PATH = main.VIDEO_PATH
AUDIO_OUTPUT_DIR = main.AUDIO_OUTPUT_DIR
OUTPUT_VIDEO_PATH = main.OUTPUT_VIDEO_PATH

# 替換 pydub 的靜音生成函式 (使用 numpy，已修正 Python 3.13 錯誤)
def create_silence_clip(duration_sec, fps=44100):
    """直接使用 numpy 和 moviepy 創建一個靜音 AudioClip"""
    # 確保持續時間不為負
    duration_sec = max(0, duration_sec) 
    samples = int(duration_sec * fps)
    silent_array = np.zeros((samples, 2), dtype=np.float32) # 立體聲, float32
    
    silence_clip = AudioArrayClip(silent_array, fps=fps)
    return silence_clip

def combine_audio_and_video(audio_segments_info):
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"找不到原始影片: {VIDEO_PATH}")

    if not audio_segments_info:
        print("沒有語音片段資訊，無法合併音訊。")
        return

    # 1. 載入原始影片並獲取其長度
    print("載入原始影片...")
    video_clip = VideoFileClip(VIDEO_PATH)
    original_duration = video_clip.duration
    print(f"原始影片長度為: {original_duration:.2f} 秒。")
    
    final_audio_segments = []
    current_time = 0.0

    # 排序以防萬一
    audio_segments_info.sort(key=lambda x: x["start"])

    for segment in audio_segments_info:
        start = segment["start"]
        end = segment["end"]
        audio_path = segment["audio_path"]
        
        # 檢查片段是否超出影片範圍
        if start >= original_duration:
            print(f"跳過片段 {audio_path}：開始時間 {start:.2f}s 已超出影片長度 {original_duration:.2f}s。")
            continue

        # 處理靜音間隔
        if start > current_time:
            silence_duration = start - current_time
            print(f"插入靜音: {current_time:.2f}s 到 {start:.2f}s (時長: {silence_duration:.2f}s)")
            silence_clip = create_silence_clip(silence_duration)
            final_audio_segments.append(silence_clip)
        
        # 處理語音片段
        audio_clip_full = AudioFileClip(audio_path)
        
        # 計算影片中可用的最大時長
        max_duration_in_video = original_duration - start
        
        # 根據 Plan 期望的時長
        required_duration_from_plan = end - start
        
        # 最終插入的時長：取三者最小值 (Plan 時長, 影片剩餘時長, 實際音訊檔案長度)
        duration_to_insert = min(required_duration_from_plan, max_duration_in_video, audio_clip_full.duration)
        
        if duration_to_insert <= 0:
             print(f"警告：片段 {audio_path} 插入時長為零或負數，跳過。")
             continue
             
        # ** 關鍵修正：將 .subclip(0, duration_to_insert) 替換為切片語法 **
        # 截取音訊片段至所需長度 (這是防止影片被拉長的關鍵)
        # 由於 AudioFileClip 載入時 duration 已經確定，這裡切片到所需時長
        audio_clip = audio_clip_full[0:duration_to_insert] 
        
        print(f"插入語音: {audio_path} (時長: {audio_clip.duration:.2f}s)")
        final_audio_segments.append(audio_clip)
        
        # current_time 必須根據實際插入的音訊長度來更新
        current_time = start + audio_clip.duration 

    # 3. 處理結尾的靜音
    if original_duration > current_time:
        silence_duration = original_duration - current_time
        print(f"插入結尾靜音: {current_time:.2f}s 到 {original_duration:.2f}s (時長: {silence_duration:.2f}s)")
        silence_clip = create_silence_clip(silence_duration)
        final_audio_segments.append(silence_clip)

    # 4. 合併所有音訊片段
    print("合併所有音訊片段...")
    final_audio_clip = concatenate_audioclips(final_audio_segments)

    # 5. 將新的音軌設定給影片
    final_clip = video_clip.with_audio(final_audio_clip)

    # 6. 寫出最終影片
    print(f"寫出最終影片: {OUTPUT_VIDEO_PATH}")
    final_clip.write_videofile(
        OUTPUT_VIDEO_PATH,
        codec='libx264', # 影片編碼
        audio_codec='aac', # 音訊編碼
        temp_audiofile='temp-audio.m4a', # 暫存音訊檔
        remove_temp=True, # 完成後刪除暫存檔
        fps=video_clip.fps # 使用原始影片的影格率
    )
    
    print("轉播影片生成完畢！")

# =========================================================
# 獨立執行邏輯
# =========================================================
def reconstruct_audio_segments_info():
    """從 plan.json 讀取資料並重建 audio_segments_info 列表。"""
    if not os.path.exists(PLAN_FILE):
        raise FileNotFoundError(f"錯誤：找不到計劃檔案 {PLAN_FILE}。請先運行 generate_text.py")

    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    audio_segments_info = []
    
    if isinstance(plan_data, dict) and "segments" in plan_data:
        for i, segment in enumerate(plan_data["segments"]):
            # 這是 tts_generator.py 的命名規則 
            # 必須使用跟 tts_generator.py 一致的檔名格式
            filename = f"segment_{i:02d}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
            filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
            
            # 必須確認音訊檔案存在
            if os.path.exists(filepath):
                audio_segments_info.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "audio_path": filepath
                })
            else:
                print(f"警告：找不到音訊檔案 {filepath}。請確保 tts_generator.py 已成功運行。")
        
        return audio_segments_info
    else:
        print(f"警告: {PLAN_FILE} 格式錯誤或無 'segments' 欄位。")
        return []


if __name__ == "__main__":
    print("正在以獨立模式運行 video_editor.py...")
    try:
        segments_info = reconstruct_audio_segments_info()
        if segments_info:
            combine_audio_and_video(segments_info)
        else:
            print("無法執行影片合併，因為沒有找到或重建有效的音訊片段資訊。")
    except FileNotFoundError as e:
        print(f"致命錯誤：{e}")
    except Exception as e:
        print(f"發生錯誤：{e}")