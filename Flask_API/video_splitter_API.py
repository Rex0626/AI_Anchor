from flask import Flask, request, jsonify
from video_splitter.video_splitter import split_video

app = Flask(__name__)

@app.route('/split_video', methods=['POST'])
def api_split_video():
    data = request.json
    video_path = data.get("video_path")
    output_folder = data.get("output_folder")
    segment_length = data.get("segment_length", 30)
    
    # Flask 调用时 use_log=False
    result = split_video(video_path, output_folder, segment_length, use_log=False)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
