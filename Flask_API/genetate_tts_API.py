from flask import Flask, request, jsonify
from TextToSpeech.generate_tts_google import batch_process

app = Flask(__name__)

@app.route("/tts_batch", methods=["POST"])
def api_tts_batch():
    data = request.json
    input_folder = data.get("input_folder")
    output_folder = data.get("output_folder")
    result = batch_process(input_folder, output_folder)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
