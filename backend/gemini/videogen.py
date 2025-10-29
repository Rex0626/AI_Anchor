import os
import json
from datetime import timedelta
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from google.cloud import storage
from tqdm  import tqdm

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
你是一位**資深的羽球賽事即時分析員與主播**，正在為一段羽球比賽影片撰寫逐段旁白。

🎯 你的任務：
1. **分析畫面**：透過影片分析，辨識選手的具體動作與擊球路徑。
2. **撰寫旁白**：根據分析結果，以專業主播的口吻，**每 {{ time_interval }} 秒為一個播報時段**，針對時段內發生的重要事件撰寫旁白。

📌 撰寫規則如下：
1. 這段影片總長度為 {{ duration }} 秒。**請務必在不超過 {{ sentence_count }} 句的上限下，只挑選最關鍵的開頭銜接、重要擊球或得分事件撰寫旁白。**
2. **【內容絕對規則】**：**旁白句子必須精確到個人！** 無論總結的時長多久，請務必在句子中：**(1) 點名選手 (使用括號內的名字，如：馬來西亞(Tan))；(2) 描述明確擊球動作；(3) 提到結果或落點。**
3. 每句旁白必須**簡潔有力**，字數請控制在**30個中文字內**，以確保播報流暢度。
4. 每句格式為：「【情緒】**[隊伍/球員名]** + **[具體動作]** + **[結果]**」
5. 情緒標籤為：【平穩】、【緊張】、【激動】
6. 可加入「漂亮一擊！」「精彩救球！」等情緒詞。
7. **【絕對優先規則】**：**本段旁白總句數絕不允許超過 {{ sentence_count }} 句！** 這是基於語音時長計算的上限，你必須嚴格遵守。

📽️ 影片背景資料如下（包含隊伍與球員資訊）：
{{ intro }}

🔄 **上一個片段的結尾動作（請確保本段旁白與之銜接）：**
{{ last_action }}
# 👆 新增：實現片段間的連貫性。

請為這段影片撰寫旁白：
 """

prompt_builder = PromptBuilder(
    template=prompt_template,
    required_variables=["intro", "last_action", "duration", "sentence_count", "time_interval"]
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

    # <<<< 新增這行：初始化連貫性變數 >>>>
    last_segment_summary = "賽事畫面開始。"

    for file_name in tqdm(video_files, desc="[AI主播] 處理影片片段"):
        segment_path = os.path.join(video_folder, file_name)
        with VideoFileClip(segment_path) as clip:
            duration = round(clip.duration)
            
            # <<<< 修改點 A：定義並使用更長的安全間隔 >>>>
            TIME_INTERVAL = 8 # 將間隔從 5 秒增加到 8 秒，緩解語音重疊
            sentence_count = max(1, duration // TIME_INTERVAL) # 計算句子數量的安全上限

        prompt_input = {
            "upload2gcs": {"file_path": segment_path},
            "prompt_builder": {
                "intro": intro_text,
                "last_action": last_segment_summary, # <<<< 傳遞連貫性變數 >>>>
                "duration": duration,
                "sentence_count": sentence_count,
                "time_interval": TIME_INTERVAL # <<<< 傳遞時間間隔給 Prompt >>>>
            }
        }

        result = pipeline.run(prompt_input)

        reply = result["llm"]["replies"][0].strip()
        # 目的：只保留以情緒標籤「【」開頭的行，忽略 LLM 的回覆、說明、分隔符號。
        commentary_lines = [
            line.strip() 
            for line in reply.split("\n") 
            if line.strip().startswith("【") # 確保行是以「【」開頭的有效旁白
    ]

        if not commentary_lines:
             continue
        
        # <<<< 確保使用實際產生的行數來計算 per_line_duration >>>>
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
        
        # <<<< 新增這行：更新連貫性變數 >>>>
        last_segment_summary = commentary_with_time[-1]["text"]

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
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/batch_badminton_outputs"
    intro_text = input("請輸入影片背景介紹：")
    process_video_segments(video_folder, output_folder, intro_text)
