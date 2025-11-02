#參數設定

import os

# 檔案與路徑設定
VIDEO_PATH = "very_long.mp4" # 您的原始影片檔案名稱
PLAN_FILE = "plan.json"
AUDIO_OUTPUT_DIR = "generated_audio"
OUTPUT_VIDEO_PATH = "broadcast_video.mp4"
MODEL_TEXT_GEN = "gemini-2.5-flash"

# === 您的 API KEY 設定 ===
# !!! 請將此處替換為您的實際 Gemini API Key !!!
API_KEY = "AIzaSyDoAC6ks_fa4kSV_zhjvR4UaZUI1WRYqGM" 

# === Google Cloud / Vertex AI 設定 ===
PROJECT_ID = "sports-broadcast-system"  # <--- 請替換成您的 Project ID
LOCATION = "asia-east1"                  

# TTS (Text-to-Speech) 設定
TTS_VOICE_NAME = "cmn-TW-Wavenet-B"

#PROMPT設定
PROMPT = """
您是一位專業的體育賽事轉播評論員，請根據提供的影片內容生成一段旁白腳本。
您的任務是生成一個 JSON 格式的列表，其中每個元素都是一個語音片段。

**【重要輸出規則】**

1.  自然語氣：確保每個 'text' 欄位的內容都是**完整且連貫的語句**，並且在邏輯上是一個可以暫停呼吸的自然段落。**絕對不要在一個句子的中間將其打斷為兩個片段。**
2.  對齊原則：'start' 和 'end' 必須精確對齊到影片中說明的動作時間點，但**優先保證文本的自然度和完整性**。如果一個語句為了保持完整性而稍微超出 'end' 時間幾毫秒，這是可以接受的。
3.  格式嚴格：嚴格遵循以下 JSON 格式。
4.  可以增加一些抑揚頓挫，讓整個轉播影片更為生動。
5.  不用每個動作都描述到，有關鍵情況(如得分、受傷...)再描述就好，其他時間可以講一些選手的背景知識，如最近的比賽狀況，賽程等等...
6.  一定要講完一句再講下一句，若下一句的時間不夠了則跳過該句。
**【輸出格式】**

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "這是比賽開始的第一分鐘，雙方正在中場進行激烈的爭奪。"
    },
    {
      "start": 3.5,
      "end": 8.1,
      "text": "傳球！前鋒拿到了球，他似乎找到了射門的機會，這會是一個進球嗎？"
    },
    // ... 所有片段都應包含在這個單一的 JSON 結構中
  ]
}"""