import os
import json
import re
from google.cloud import texttospeech
import hashlib

# ========== 憑證載入、設定 ==========
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
cred_path = os.path.join(os.path.dirname(PROJECT_ROOT), "credentials", "ai-anchor-462506-7887b7105f6a.json")
assert os.path.exists(cred_path), f"❌ 憑證不存在: {cred_path}"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path


client = texttospeech.TextToSpeechClient()

# 情緒到語速(rate)和音量(volume_gain_db)的映射
EMOTION_TTS_PARAMS = {
    "激動": {"rate": 1.7, "volume_gain_db": 3.5},
    "平穩": {"rate": 1.5, "volume_gain_db": 0.0},
    "緊張": {"rate": 1.6, "volume_gain_db": 2.0},
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
    
    # 1. 將中文標點替換成 SSML 停頓標籤
    ssml_text = sentence_text
    ssml_text = ssml_text.replace("，", "<break time='200ms'/>") # 逗號：短暫停頓/換氣
    ssml_text = ssml_text.replace("、", "<break time='100ms'/>") # 頓號：極短停頓
    ssml_text = ssml_text.replace("。", "<break time='400ms'/>") # 句號：正常語氣結束
    ssml_text = ssml_text.replace("！", "<break time='500ms'/>") # 驚嘆號：較長且有力的停頓
    
    # 2. 使用處理後的 ssml_text 組合最終的 SSML 字串
    ssml = f"<speak><prosody rate='{params['rate']}' volume='{params['volume_gain_db']}dB'>{ssml_text}</prosody></speak>"

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
        # <<<< 修正點：直接讀取獨立的 emotion 欄位 >>>>
        text = item["text"]
        emotion = item.get("emotion", "平穩") # 如果沒有 emotion 欄位，預設為平穩

        # 1. 計算當前文本的 SHA256 雜湊值
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        
        # 2. 定義 MP3 檔案和伴隨的雜湊檔案路徑
        out_path_mp3 = os.path.join(segment_dir, f"{idx+1:03d}_{emotion}.mp3")
        out_path_hash = os.path.join(segment_dir, f"{idx+1:03d}_{emotion}.hash") # 伴隨雜湊檔案

        # 3. 檢查跳過條件 (只有 MP3 和 Hash 文件都存在且雜湊匹配時才跳過)
        is_mp3_present = os.path.exists(out_path_mp3)
        is_hash_present = os.path.exists(out_path_hash)

        should_regenerate = True
        
        if is_mp3_present and is_hash_present:
            with open(out_path_hash, 'r', encoding='utf-8') as hf:
                stored_hash = hf.read().strip()
            
            if stored_hash == text_hash:
                print(f"✅ 語音已存在且文本未修改，跳過生成：{out_path_mp3}")
                should_regenerate = False
            else:
                print(f"⚠️ 文本已修改，需要重新生成語音：{out_path_mp3}")
        
        if not should_regenerate:
            continue # 跳過 TTS API 呼叫

        # 4. 原有的重複文本檢查 (防止同一 JSON 內重複生成)
        if text in seen_texts:
            print(f"⚠️ 重複旁白，跳過：{text}")
            continue

        seen_texts.add(text)
        
        # 5. 執行語音生成 (如果文件不存在或雜湊不匹配)
        res = synthesize_sentence(text, emotion, out_path_mp3) 
        
        # 6. 如果生成成功，儲存新的雜湊值到 .hash 檔案
        if res['status'] == 'success':
            with open(out_path_hash, 'w', encoding='utf-8') as hf:
                hf.write(text_hash)
            
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
    input_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"
    output_folder = "D:/Vs.code/AI_Anchor/backend/TextToSpeech/final_tts_google"
    result = batch_process(input_folder, output_folder)
    print(json.dumps(result, ensure_ascii=False, indent=2))
