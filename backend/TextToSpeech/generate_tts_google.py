import os
import json
import re
from google.cloud import texttospeech

# ========== 憑證載入、設定 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(os.path.dirname(PROJECT_ROOT), "credentials", "ai-anchor-462506-7887b7105f6a.json")
assert os.path.exists(cred_path), f"❌ 憑證不存在: {cred_path}"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


client = texttospeech.TextToSpeechClient()

# 情緒到語速(rate)和音量(volume_gain_db)的映射
EMOTION_TTS_PARAMS = {
    "激動": {"rate": 1.5, "volume_gain_db": 3.5},
    "平穩": {"rate": 1.0, "volume_gain_db": 0.0},
    "緊張": {"rate": 1.4, "volume_gain_db": 2.0},
    "疑問": {"rate": 1.0, "volume_gain_db": 1.5},
    "強調": {"rate": 1.2, "volume_gain_db": 3.0},
    "精彩": {"rate": 1.5, "volume_gain_db": 3.0},
}

def clean_emotion_tag(text):
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
        return {"status": "success", "output": output_path, "emotion": emotion}
    except Exception as e:
        print(f"❌ 語音生成失敗：{output_path}，錯誤：{e}")
        return {"status": "error", "message": str(e), "output": output_path}

def process_segment_json(json_path, output_base_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segment_name_raw = data.get("segment", os.path.splitext(os.path.basename(json_path))[0])
    segment_name = segment_name_raw.replace(".mp4", "")
    commentary = data.get("commentary", [])

    if not commentary:
        msg = f"⚠️ 空旁白，跳過：{segment_name}"
        print(msg)
        return {"status": "warning", "message": msg, "segment": segment_name}

    segment_dir = os.path.join(output_base_dir, segment_name)
    os.makedirs(segment_dir, exist_ok=True)

    results = []
    seen_texts = set()
    for idx, item in enumerate(commentary):
        emotion, text = clean_emotion_tag(item["text"])

        # 新增重複檢查
        if text in seen_texts:
            print(f"⚠️ 重複旁白，跳過：{text}")
            continue

        seen_texts.add(text)    # 記錄已處理過的文本
        out_path = os.path.join(segment_dir, f"{idx+1:03d}_{emotion}.mp3")
        res = synthesize_sentence(text, emotion, out_path)
        results.append(res)

    return {"status": "success", "segment": segment_name, "results": results}

def batch_process(input_json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_json_folder) if f.endswith(".json")]

    all_results = []
    for jf in sorted(json_files):
        json_path = os.path.join(input_json_folder, jf)
        res = process_segment_json(json_path, output_folder)
        all_results.append(res)

    return {"status": "success", "processed_files": len(json_files), "details": all_results}

# ✅ 後端單測模式
if __name__ == "__main__":
    input_folder = "D:/Vs.code/AI_Anchor/backend/gemini/batch_badminton_outputs"
    output_folder = "D:/Vs.code/AI_Anchor/backend/TextToSpeech/emotional_outputs"
    result = batch_process(input_folder, output_folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
