import os
import json
from datetime import timedelta
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from google.cloud import storage


# ========== 憑證載入 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ✅ 工具函數
def seconds_to_timecode(seconds):
    return str(timedelta(seconds=round(seconds)))



# ========== 組件 ==========
@component
class Upload2GCS:
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    @component.output_types(uri=str)
    def run(self, file_path: str):
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        file_name = os.path.basename(file_path)
        blob = bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        return {"uri": f"gs://{self.bucket_name}/{file_name}"}

@component
class AddVideo2Prompt:
    @component.output_types(prompt=list)
    def run(self, uri: str, prompt: str):
        return {"prompt": [Part.from_uri(uri, mime_type="video/mp4"), prompt]}

@component
class GeminiGenerator:
    def __init__(self, project_id, location, model):
        self.project_id = project_id
        self.location = location
        self.model = model

    @component.output_types(replies=list)
    def run(self, prompt: list):
        generator = VertexAIGeminiGenerator(
            project_id=self.project_id,
            location=self.location,
            model=self.model
        )
        return {"replies": generator.run(prompt)["replies"]}

# ========== Prompt ==========
prompt_template = """ 
你是一位專業的運動主播，正在為一段羽球比賽影片撰寫逐段旁白。

🎯 你的任務：根據影片內容，每 5 秒產出一句旁白，自然描述場上正在發生的動作與事件。

📌 撰寫規則如下：
1. 這段影片總長度為 {{ duration }} 秒，因此你只需撰寫 {{ sentence_count }} 句旁白。
2. 不用回覆，不需額外說明，也不要重述規則，直接產出旁白句子，共 {{ sentence_count }} 句。
3. 每句格式為：「【情緒】角色 + 動作 + 事件」
4. 情緒標籤為：【平穩】、【緊張】、【激動】
5. 特別事件（如：開球、結束、失誤、得分）必須描述。
6. 可加入「漂亮一擊！」「精彩救球！」等情緒詞。

📽️ 影片背景資料如下：
{{ intro }}

📜 目前為止的旁白內容如下：
{{ context }}

請繼續為這段影片撰寫旁白：
 """

prompt_builder = PromptBuilder(
    template=prompt_template,
    required_variables=["intro", "context", "duration", "sentence_count"]
)

# ========== Pipeline ==========
upload2gcs = Upload2GCS(bucket_name="ai_anchor")
add_video_2_prompt = AddVideo2Prompt()
gemini_generator = GeminiGenerator(
    project_id="ai-anchor-462506",
    location="us-central1",
    model="gemini-2.5-flash"
)

pipeline = Pipeline()
pipeline.add_component(instance=upload2gcs, name="upload2gcs")
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=add_video_2_prompt, name="add_video")
pipeline.add_component(instance=gemini_generator, name="llm")

pipeline.connect("upload2gcs", "add_video")
pipeline.connect("prompt_builder", "add_video")
pipeline.connect("add_video.prompt", "llm")


# ========== 主邏輯（可供 Flask 調用） ==========
def process_video_segments(video_folder, output_folder, intro_text):
    os.makedirs(output_folder, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    results = []

    for file_name in video_files:
        segment_path = os.path.join(video_folder, file_name)
        with VideoFileClip(segment_path) as clip:
            duration = round(clip.duration)
            sentence_count = max(1, duration // 5)

        prompt_input = {
            "upload2gcs": {"file_path": segment_path},
            "prompt_builder": {
                "intro": intro_text,
                "context": "",
                "duration": duration,
                "sentence_count": sentence_count
            }
        }

        result = pipeline.run(prompt_input)
        reply = result["llm"]["replies"][0].strip()
        commentary_lines = [line.strip() for line in reply.split("\n") if line.strip()]

        per_line_duration = duration / len(commentary_lines)
        commentary_with_time = []
        for idx, line in enumerate(commentary_lines):
            start = idx * per_line_duration
            end = start + per_line_duration
            commentary_with_time.append({
                "start_time": seconds_to_timecode(start),
                "end_time": seconds_to_timecode(end),
                "text": line
            })

        segment_obj = {
            "segment": file_name,
            "commentary": commentary_with_time
        }
        results.append(segment_obj)

        json_filename = f"{os.path.splitext(file_name)[0]}.json"
        with open(os.path.join(output_folder, json_filename), "w", encoding="utf-8") as f:
            json.dump(segment_obj, f, ensure_ascii=False, indent=2)

    return results

# ✅ 後端單測模式
if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/video_splitter/badminton_segments"
    output_folder = "D:/Vs.code/AI_Anchor/gemini/batch_badminton_outputs"
    intro_text = input("請輸入影片背景介紹：")
    process_video_segments(video_folder, output_folder, intro_text)
