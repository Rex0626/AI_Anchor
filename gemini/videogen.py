import os
import json
from datetime import timedelta
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from google.cloud import storage

# ========== 憑證載入、設定 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 取得項目根目錄（假設每個模組都在根目錄下一層）
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json") # 憑證路徑
assert os.path.exists(cred_path), f"❌ 憑證不存在: {cred_path}" # 確認憑證存在
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path    # 設定環境變數


# ✅ 將秒數轉換成 HH:MM:SS 格式
def seconds_to_timecode(seconds):
    return str(timedelta(seconds=round(seconds)))


# ========== Step 1：自定義組件：上傳影片到 GCS ==========
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

# ========== Step 2：影片 URI + prompt 合併 ==========
@component
class AddVideo2Prompt:
    @component.output_types(prompt=list)
    def run(self, uri: str, prompt: str):
        return {"prompt": [Part.from_uri(uri, mime_type="video/mp4"), prompt]}

# ========== Step 3：Gemini 生成組件 ==========
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

# ========== Step 4：旁白撰寫提示詞 ==========
prompt_template = """
你是一位專業的運動主播，正在為一段羽球比賽影片撰寫逐段旁白。

🎯 你的任務：根據影片內容，每 5 秒產出一句旁白，自然描述場上正在發生的動作與事件。

📌 撰寫規則如下：
1. 這段影片總長度為 {{ duration }} 秒，因此你只需撰寫 {{ sentence_count }} 句旁白。
2. 不用回覆，不需額外說明，也不要重述規則，直接產出旁白句子，共 {{ sentence_count }} 句。
3. 每句格式為：「【情緒】角色 + 動作 + 事件」
4. 情緒標籤為：【平穩】、【緊張】、【激動】
    - 例子：「【激動】林選手殺球得分！」、「【平穩】雙方來回對打中。」
    - 每句請控制在 20 個字以內，具備節奏感與現場感。
5. 特別事件（如：開球、結束、失誤、得分）必須描述，一般來回過程可略過。
6. 可加入「漂亮一擊！」「精彩救球！」等情緒詞。

📽️ 影片背景資料如下：
{{ intro }}

📜 目前為止的旁白內容如下：
{{ context }}

請繼續為這段影片撰寫旁白：
"""

# ✅ PromptBuilder
prompt_builder = PromptBuilder(
    template=prompt_template,
    required_variables=["intro", "context", "duration", "sentence_count"]
)

# ========== Step 5：建立 Pipeline ==========
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

# ========== Step 6：處理影片段落 ==========
video_folder = "D:/Vs.code/AI_Anchor/video_splitter/badminton_segments"
output_folder = "D:/Vs.code/AI_Anchor/gemini/batch_badminton_outputs"
os.makedirs(output_folder, exist_ok=True)

video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
full_context = ""

# 只輸入一次 intro，所有片段共用
intro_text = input("請輸入影片背景介紹（intro）：")

# 累積總時間軸用
total_elapsed = 0

for i, file_name in enumerate(video_files):
    segment_path = os.path.join(video_folder, file_name)
    print(f"\U0001F3AC 處理片段：{file_name}...")

    # ✅ 使用 moviepy 讀取實際影片秒數
    with VideoFileClip(segment_path) as clip:
        duration = round(clip.duration)
        sentence_count = max(1, duration // 5)

    prompt_input = {
        "upload2gcs": {"file_path": segment_path},
        "prompt_builder": {
            "intro": intro_text, 
            "context": full_context,
            "duration": duration,
            "sentence_count": sentence_count
        }
    }

    try:
        result = pipeline.run(prompt_input)
        reply = result["llm"]["replies"][0].strip()
        commentary_lines = [line.strip() for line in reply.split("\n") if line.strip()]

        per_line_duration = duration / len(commentary_lines)
        start_sec = 0  # ✅ 每段影片都從 0 開始
        commentary_with_time = []

        for idx, line in enumerate(commentary_lines):
            start = idx * per_line_duration
            end = start + per_line_duration
            commentary_with_time.append({
                "start_time": seconds_to_timecode(start),
                "end_time": seconds_to_timecode(end),
                "text": line
            })

        # segment 起訖時間還是用全局累計來標示，但 commentary 是相對時間！
        end_sec = total_elapsed + duration
        segment_obj = {
            "segment": file_name,
            "start_time": seconds_to_timecode(total_elapsed),
            "end_time": seconds_to_timecode(end_sec),
            "commentary": commentary_with_time
        }
        total_elapsed = end_sec

        # 儲存 JSON 檔案
        segment_obj = {
            "segment": file_name,
            "start_time": seconds_to_timecode(start_sec),
            "end_time": seconds_to_timecode(end_sec),
            "commentary": commentary_with_time
        }

        segment_index = int(file_name.split("_")[1].split(".")[0])
        json_filename = f"segment_{segment_index:03d}.json"

        with open(os.path.join(output_folder, json_filename), "w", encoding="utf-8") as f:
            json.dump(segment_obj, f, ensure_ascii=False, indent=2)


        full_context += f"\n\n[{file_name}]\n" + "\n".join(commentary_lines)
        print(f"✅ 完成旁白：{file_name}")

    except Exception as e:
        print(f"❌ 發生錯誤：{e}")

print("🎉 所有段落旁白已完成！")