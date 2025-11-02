#文本轉語音
import os
import json
from google.cloud import texttospeech
import main

PLAN_FILE = main.PLAN_FILE
AUDIO_OUTPUT_DIR = main.AUDIO_OUTPUT_DIR
VOICE_NAME = main.TTS_VOICE_NAME

def generate_tts(client, text, filename):
    """將單一段文本轉換為語音並儲存為 MP3/WAV"""
    # 設置 TTS 請求
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice_dict = {
            "name": VOICE_NAME,
            "language_code": "zh-TW"
        }
    # 選擇音訊輸出格式
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3 # 也可以選擇 LINEAR16 (WAV)
    )

    # 執行 TTS 請求
    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice_dict,
        audio_config=audio_config
    )

    # 將音訊內容寫入檔案
    filepath = os.path.join(AUDIO_OUTPUT_DIR, filename)
    with open(filepath, "wb") as out:
        out.write(response.audio_content)
    print(f"已生成語音檔案: {filepath}")
    return filepath

def process_plan_to_audio():
    # 初始化 Google Cloud Text-to-Speech Client
    # 這裡假設您的環境已完成 Google Cloud 驗證設定
    # 如果您想使用 Gemini-TTS，需要替換成對應的 API 呼叫 (請參考 API 文件)
    try:
        tts_client = texttospeech.TextToSpeechClient()
    except Exception as e:
        print("無法初始化 Text-to-Speech 客戶端。請確認已啟用 Google Cloud TTS API 並設定驗證。")
        print(f"錯誤: {e}")
        return

    if not os.path.exists(PLAN_FILE):
        raise FileNotFoundError(f"找不到計劃檔案: {PLAN_FILE}")

    os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

    with open(PLAN_FILE, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    audio_files = []
    # 確保 plan_data 是 JSON 格式，且包含 'segments'
    if isinstance(plan_data, dict) and "segments" in plan_data:
        for i, segment in enumerate(plan_data["segments"]):
            text = segment.get("text", "")
            if text:
                # 檔案名包含索引和時間，方便追蹤
                filename = f"segment_{i:02d}_{segment['start']:.2f}-{segment['end']:.2f}.mp3"
                filepath = generate_tts(tts_client, text, filename)
                audio_files.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "audio_path": filepath
                })
        print("所有語音片段生成完畢。")
        return audio_files
    else:
        print(f"警告: {PLAN_FILE} 格式錯誤或無 'segments' 欄位。")
        return []

if __name__ == "__main__":
    # 僅為測試，實際運行將由 run_broadcast.py 呼叫
    # 請先運行 Generate_text.py 確保 plan.json 存在
    process_plan_to_audio()