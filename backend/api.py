import os, json, shutil, uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from video_splitter.video_splitter import split_video
from gemini.videogen import process_video_segments
from TextToSpeech.generate_tts_google import batch_process
from merge_audio.merge_audio import batch_merge_all_segments

app = Flask(__name__, static_folder="static")
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
SEGMENTS_DIR = os.path.join(BASE_DIR, "static", "segments")
JSON_DIR = os.path.join(BASE_DIR, "static", "jsons")
TTS_DIR = os.path.join(BASE_DIR, "static", "tts")
OUTPUTS_DIR = os.path.join(BASE_DIR, "static", "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "static", "temp_segment")

for d in [UPLOADS_DIR, SEGMENTS_DIR, JSON_DIR, TTS_DIR, OUTPUTS_DIR, TEMP_DIR]:
    os.makedirs(d, exist_ok=True)

def to_url(path_abs):
    rel = os.path.relpath(path_abs, os.path.join(BASE_DIR, "static")).replace("\\", "/")
    return f"/static/{rel}"

# -------------------------
# 初始化 Job：切片
# -------------------------
@app.route("/api/init_job", methods=["POST"])
def init_job():
    file = request.files.get("video")
    if not file:
        return jsonify({"status": "error", "message": "缺少影片"}), 400

    job_id = str(uuid.uuid4())[:8]
    upload_path = os.path.join(UPLOADS_DIR, f"{job_id}_{file.filename}")
    file.save(upload_path)

    # ==============================
    # 🧹 STEP 1: 清理舊的暫存資料
    # ==============================
    folders_to_clear = [SEGMENTS_DIR, JSON_DIR, TTS_DIR, OUTPUTS_DIR, TEMP_DIR]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            print(f"🧹 清理資料夾：{folder}")
            for root, dirs, files in os.walk(folder):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except PermissionError:
                        print(f"⚠️ 無法刪除檔案：{f}，可能被占用，跳過。")
                for d in dirs:
                    dir_path = os.path.join(root, d)
                    try:
                        shutil.rmtree(dir_path)
                    except PermissionError:
                        print(f"⚠️ 無法刪除資料夾：{dir_path}，跳過。")

    # ==============================
    # 🧩 STEP 2: 重新建立資料夾
    # ==============================
    seg_out_dir = os.path.join(SEGMENTS_DIR, job_id)
    os.makedirs(seg_out_dir, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(TTS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # ==============================
    # 🎬 STEP 3: 影片切片
    # ==============================
    print(f"🎬 開始切片：{upload_path}")
    split_res = split_video(upload_path, seg_out_dir, 30)
    if not split_res or "segments" not in split_res:
        return jsonify({"status": "error", "message": "切片失敗"}), 500

    segments = [to_url(p) for p in split_res["segments"]]

    # ==============================
    # ✅ STEP 4: 回傳結果
    # ==============================
    return jsonify({
        "status": "success",
        "job_id": job_id,
        "video_name": file.filename,
        "segments": segments
    })


# -------------------------
# 單段處理：文本 + TTS + 合併
# -------------------------
@app.route("/api/process_segment_step", methods=["POST"])
def process_segment_step():
    video_name = request.form.get("video_name")
    description = request.form.get("description", "")
    segment_index = int(request.form.get("segment_index", 1))

    all_segments = []
    for root, _, files in os.walk(SEGMENTS_DIR):
        for f in files:
            if f.endswith(".mp4"):
                all_segments.append(os.path.join(root, f))
    all_segments = sorted(all_segments)

    if segment_index > len(all_segments):
        return jsonify({"status": "done", "message": "所有片段完成"})

    segment_path = all_segments[segment_index - 1]
    seg_file = os.path.basename(segment_path)

    # 清空 temp
    for f in os.listdir(TEMP_DIR):
        try: os.remove(os.path.join(TEMP_DIR, f))
        except: pass
    shutil.copy(segment_path, os.path.join(TEMP_DIR, seg_file))

    # 1. Gemini 文字
    process_video_segments(TEMP_DIR, JSON_DIR, description)

    commentary_array = []
    try:
        latest_json = sorted([os.path.join(JSON_DIR, f) for f in os.listdir(JSON_DIR) if f.endswith(".json")], key=os.path.getmtime, reverse=True)[0]
        with open(latest_json, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        commentary_array = data_json.get("commentary", [])
    except Exception as e:
        print("⚠️ 無法讀取 JSON:", e)

    # 2. TTS
    batch_process(JSON_DIR, TTS_DIR)

    # 3. 合併
    merged_path = batch_merge_all_segments(TEMP_DIR, JSON_DIR, TTS_DIR, OUTPUTS_DIR)
    if not merged_path:
        return jsonify({"status": "error", "message": "找不到輸出的影片"}), 500

    video_url = to_url(merged_path)
    return jsonify({"status": "success", "segment_index": segment_index, "video_url": video_url, "commentary": commentary_array})

# -------------------------
# 靜態檔案服務
# -------------------------
@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
