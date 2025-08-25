from flask import Flask, request, jsonify
from video_merger.video_merge import merge_videos

app = Flask(__name__)

@app.route('/merge_videos', methods=['POST'])
def api_merge_videos():
    data = request.json
    input_folder = data.get("input_folder")
    output_video = data.get("output_video")
    result = merge_videos(input_folder, output_video)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
