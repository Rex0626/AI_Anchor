# run_broadcast.py

import json
import os
import sys
import generate_text
import main
from tts_generator import process_plan_to_audio
from video_editor import combine_audio_and_video

# 將當前目錄添加到系統路徑，以確保能找到其他模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 檢查必要的檔案
if not os.path.exists(main.VIDEO_PATH):
    print(f"錯誤：找不到影片檔案 {main.VIDEO_PATH}。請確認檔案存在。")
    sys.exit(1)

# 步驟 1: 生成轉播文本 (Generate_text.py)
print("--- 步驟 1/3: 生成轉播文本 ---")
try:
    generate_text.generate_plan()
    print("轉播文本生成成功。")
except Exception as e:
    print(f"生成文本失敗: {e}")
    sys.exit(1)

# 步驟 2: 文本轉語音 (tts_generator.py)
print("\n--- 步驟 2/3: 文本轉語音 ---")
try:
    audio_segments_info = process_plan_to_audio()
    if not audio_segments_info:
        print("語音片段生成失敗或文本為空。")
        sys.exit(1)
    print("語音片段生成成功。")
except Exception as e:
    print(f"文本轉語音失敗: {e}")
    print("請檢查 Google Cloud Text-to-Speech API 是否啟用與驗證設定。")
    sys.exit(1)

# 步驟 3: 音訊合併至影片 (video_editor.py)
print("\n--- 步驟 3/3: 音訊合併至影片 ---")
try:
    combine_audio_and_video(audio_segments_info)
    print("影片與語音合併成功！")
except Exception as e:
    print(f"音訊合併失敗: {e}")
    sys.exit(1)

print("\n--- 運動轉播系統生成完成！ ---")