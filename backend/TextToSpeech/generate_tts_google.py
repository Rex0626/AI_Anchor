import os
import json
import re
import html
from google.cloud import texttospeech

# ========== 憑證載入、設定 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
assert os.path.exists(cred_path), f"❌ 憑證不存在: {cred_path}"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

client = texttospeech.TextToSpeechClient()

# 🎭 情緒對應參數（語速與音量）
EMOTION_TTS_PARAMS = {
    "激動": {"rate": 1.5, "volume_gain_db": 3.5},
    "平穩": {"rate": 1.0, "volume_gain_db": 0.0},
    "緊張": {"rate": 1.4, "volume_gain_db": 2.0},
    "疑問": {"rate": 1.0, "volume_gain_db": 1.5},
    "強調": {"rate": 1.2, "volume_gain_db": 3.0},
    "精彩": {"rate": 1.5, "volume_gain_db": 3.0},
}

# 🧩 從文本中取出情緒標籤與正文
def clean_emotion_tag(text):
    m = re.match(r"【(.+?)】(.*)", text)
    if m:
        return m.group(1), m.group(2).strip()
    else:
        return "平穩", text.strip()

# 🗣 產生單句語音（Google TTS）
def synthesize_sentence(sentence_text, emotion, output_path, voice="cmn-TW-Wavenet-A"):
    params = EMOTION_TTS_PARAMS.get(emotion, EMOTION_TTS_PARAMS["平穩"])

    # 轉義 SSML 特殊符號
    clean_text = html.escape(sentence_text.strip())
    if not clean_text:
        print(f"⚠️ 空白文本，略過生成：{output_path}")
        return {"status": "skip", "output": output_path}

    ssml = (
        f"<speak>"
        f"<prosody rate='{params['rate']}' volume='{params['volume_gain_db']}dB'>"
        f"{clean_text}</prosody></speak>"
    )

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

        # 檢查音訊內容是否為空
        if not response.audio_content:
            print(f"⚠️ API 回傳空音訊，跳過：{output_path}")
            return {"status": "empty", "output": output_path}

        with open(output_path, "wb") as f:
            f.write(response.audio_content)

        # 檢查檔案大小
        if os.path.getsize(output_path) == 0:
            print(f"⚠️ 生成後檔案為空：{output_path}")
            return {"status": "empty", "output": output_path}

        print(f"✅ 生成語音（{emotion}）→ {output_path}")
        return {"status": "success", "output": output_path, "emotion": emotion}

    except Exception as e:
        print(f"❌ 語音生成失敗：{output_path}，錯誤：{e}")
        return {"status": "error", "message": str(e), "output": output_path}

# 🎯 單一 segment 的 TTS 處理邏輯
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

    for idx, item in enumerate(commentary):
        try:
            emotion, text = clean_emotion_tag(item.get("text", ""))
            out_path = os.path.join(segment_dir, f"{idx+1:03d}.mp3")

            # 檢查是否已有音檔
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"🟡 已存在且有效，略過：{out_path}")
                continue

            res = synthesize_sentence(text, emotion, out_path)
            results.append(res)
        except Exception as e:
            print(f"❌ 處理失敗（{item.get('text', '')}）：{e}")

    return {"status": "success", "segment": segment_name, "results": results}

# 🚀 批次處理所有 JSON
def batch_process(input_json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    json_files = [f for f in os.listdir(input_json_folder) if f.endswith(".json")]
    if not json_files:
        print("⚠️ 沒有可用 JSON 檔案")
        return {"status": "warning", "message": "no_json_files"}

    all_results = []
    for jf in sorted(json_files):
        json_path = os.path.join(input_json_folder, jf)
        res = process_segment_json(json_path, output_folder)
        all_results.append(res)

    return {"status": "success", "processed_files": len(json_files), "details": all_results}

# ✅ 單獨測試執行模式
if __name__ == "__main__":
    input_folder = "D:/Vs.code/AI_Anchor/gemini/batch_badminton_outputs"
    output_folder = "D:/Vs.code/AI_Anchor/TextToSpeech/emotional_outputs"
    result = batch_process(input_folder, output_folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
