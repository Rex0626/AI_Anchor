from flask import Flask, request, jsonify
import os
import sys

# 確保能引入 backend 模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gemini.videogen import process_video_segments

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate_commentary():
    data = request.json
    video_folder = data.get("video_folder")
    output_folder = data.get("output_folder")
    intro_text = data.get("intro", "")

    if not video_folder or not output_folder:
        return jsonify({"error": "缺少必要參數 video_folder/output_folder"}), 400

    try:
        results = process_video_segments(video_folder, output_folder, intro_text)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
