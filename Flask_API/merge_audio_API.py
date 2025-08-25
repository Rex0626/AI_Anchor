from flask import Flask, request, jsonify
from merge_audio.merge_audio import batch_merge_all_segments

app = Flask(__name__)

@app.route("/merge", methods=["POST"])
def merge():
    data = request.json
    result = batch_merge_all_segments(
        data["video_folder"],
        data["json_folder"],
        data["tts_folder"],
        data["output_folder"]
    )
    return jsonify(result)
