import os
import json
import re
from google.cloud import texttospeech

# ========== 憑證載入、設定 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 取得項目根目錄（假設每個模組都在根目錄下一層）
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json") # 憑證路徑
assert os.path.exists(cred_path), f"❌ 憑證不存在: {cred_path}" # 確認憑證存在
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path    # 設定環境變數



client = texttospeech.TextToSpeechClient()

# 情緒到語速(rate)和音量(volume_gain_db)的映射（rate 最大1.5，volume 最大1.5倍約等於 +3.5dB）
EMOTION_TTS_PARAMS = {
    "激動": {"rate": 1.5, "volume_gain_db": 3.5},
    "平穩": {"rate": 1.0, "volume_gain_db": 0.0},
    "緊張": {"rate": 1.4, "volume_gain_db": 2.0},
    "疑問": {"rate": 1.0, "volume_gain_db": 1.5},
    "強調": {"rate": 1.2, "volume_gain_db": 3.0},
    "精彩": {"rate": 1.5, "volume_gain_db": 3.0},
}

def clean_emotion_tag(text):
    # 從句子開頭擷取【情緒】標籤
    m = re.match(r"【(.+?)】(.*)", text)
    if m:
        return m.group(1), m.group(2).strip()
    else:
        return "平穩", text  # 沒標籤就當平穩

def synthesize_sentence(sentence_text, emotion, output_path, voice="cmn-TW-Wavenet-A"):
    params = EMOTION_TTS_PARAMS.get(emotion, EMOTION_TTS_PARAMS["平穩"])
    ssml = f"<speak><prosody rate='{params['rate']}' volume='{params['volume_gain_db']}dB'>{sentence_text}</prosody></speak>"
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="cmn-TW",
        name=voice,
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    try:
        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice_params,
            audio_config=audio_config,
        )
        with open(output_path, "wb") as f:
            f.write(response.audio_content)
        print(f"✅ 生成語音：{output_path}")
    except Exception as e:
        print(f"❌ 語音生成失敗：{output_path}，錯誤：{e}")

def process_segment_json(json_path, output_base_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segment_name_raw = data.get("segment", os.path.splitext(os.path.basename(json_path))[0])
    segment_name = segment_name_raw.replace(".mp4", "")  # ✅ 去掉 .mp4 副档名

    commentary = data.get("commentary", [])

    if not commentary:
        print(f"⚠️ 空旁白，跳過：{segment_name}")
        return

    segment_dir = os.path.join(output_base_dir, segment_name)
    os.makedirs(segment_dir, exist_ok=True)

    # commentary 是 [{"start_time":..., "end_time":..., "text":...}, ...]
    for idx, item in enumerate(commentary):
        emotion, text = clean_emotion_tag(item["text"])
        out_path = os.path.join(segment_dir, f"{idx+1:03d}_{emotion}.mp3")
        synthesize_sentence(text, emotion, out_path)

def batch_process(input_json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_json_folder) if f.endswith(".json")]
    for jf in sorted(json_files):
        json_path = os.path.join(input_json_folder, jf)
        process_segment_json(json_path, output_folder)

if __name__ == "__main__":
    input_folder = "D:/Vs.code/AI_Anchor/gemini/batch_badminton_outputs"  # 你的旁白 JSON 片段資料夾
    output_folder = "D:/Vs.code/AI_Anchor/TextToSpeech/emotional_outputs"  # 生成語音輸出資料夾
    batch_process(input_folder, output_folder)
