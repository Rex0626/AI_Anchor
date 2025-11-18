import os
import json
import re
from datetime import timedelta
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from google.cloud import storage
from tqdm import tqdm
from google.api_core import exceptions # 用於更詳細的錯誤處理

# ========== 憑證載入 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ✅ 工具函數
def seconds_to_timecode(seconds):
    return str(timedelta(seconds=round(seconds)))


# ========== 組件 (與您提供的一致) ==========
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

# ========== Prompt (Stage 1: Event Analysis) ==========
event_analysis_template = """ 
你是一位**客觀且極速的事件分析器**，你的任務是將影片內容分解成結構化 JSON 數據。

🎯 你的任務：分析影片中的所有動作，並輸出一個 JSON 陣列。

📌 撰寫規則如下：
1. **輸出格式必須嚴格遵守 JSON 陣列結構**，不允許任何額外說明或文字。
2. **time_range 必須精確到毫秒**（例如：0:01.2），作為第二階段時間標記的依據。
3. 盡可能將多個同時發生的動作（例如：扣殺與防守）歸入同一個 `time_range` 內的 `events` 陣列中。
4. 輸出必須完整涵蓋該影片片段的所有重要動作。

📽️ 影片背景資料如下：
{{ intro }}

JSON 輸出範例（請直接輸出 JSON 陣列）：
[
    { 
        "time_range": "0:01.2-0:03.5",
        "events": [
            { 
                "player": "日本隊的Miyau",
                "action": "發球",
                "location": "後場右側"
            },
            { 
                "player": "馬來西亞隊的Thinaah",
                "action": "接發球",
                "location": "網前"
            }
        ],
        "result": "球權轉換",
        "is_crucial": false
    },
    { 
        "time_range": "0:08.1-0:09.6",
        "events": [
            // ... (其他事件)
        ],
        "result": "得分",
        "is_crucial": true
    }
]
"""
prompt_builder_event = PromptBuilder(
    template=event_analysis_template,
    required_variables=["intro"]
)

# ========== Pipeline (Stage 1) ==========
# <<<< 修正點：拆分為兩個 Pipeline 以解決輸出問題 >>>>

# --- 組件實例化 ---
upload2gcs = Upload2GCS(bucket_name="ai_anchor")
add_video_2_prompt = AddVideo2Prompt()
gemini_generator = GeminiGenerator(
    project_id="ai-anchor-462506",
    location="us-central1",
    model="gemini-2.5-flash"
)

# --- Pipeline 1: 專門用於上傳 ---
pipeline_upload = Pipeline()
pipeline_upload.add_component(instance=upload2gcs, name="upload2gcs")
# (upload2gcs 是此 Pipeline 的終端節點，其輸出會被回傳)


# --- Pipeline 2: 專門用於事件分析 (接收 URI) ---
pipeline_event_analysis = Pipeline()
pipeline_event_analysis.add_component(instance=prompt_builder_event, name="prompt_builder") 
pipeline_event_analysis.add_component(instance=add_video_2_prompt, name="add_video")
pipeline_event_analysis.add_component(instance=gemini_generator, name="llm")

pipeline_event_analysis.connect("prompt_builder", "add_video")
pipeline_event_analysis.connect("add_video.prompt", "llm")


# ========== 主邏輯（重構為只執行 Stage 1） ==========
def process_stage1_events(video_folder, output_folder, intro_text):
    os.makedirs(output_folder, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    
    for file_name in tqdm(video_files, desc="[AI主播] Stage 1 事件分析"):
        segment_path = os.path.join(video_folder, file_name)
        
        json_str = "" # 初始化 json_str
        video_uri = ""

        try:
            # --- 步驟 1: 呼叫 Pipeline 1 (上傳影片) ---
            upload_input = {"upload2gcs": {"file_path": segment_path}}
            upload_result = pipeline_upload.run(upload_input)
            video_uri = upload_result["upload2gcs"]["uri"] # 獲取 URI

            # --- 步驟 2: 呼叫 Pipeline 2 (事件分析) ---
            prompt_input_event = {
                "add_video": {"uri": video_uri}, # <<<< 傳入 URI
                "prompt_builder": {"intro": intro_text}
            }
            event_result = pipeline_event_analysis.run(prompt_input_event)
            json_str = event_result["llm"]["replies"][0].strip()
            
            # 強化 JSON 輸出清理 (移除 Markdown 標記和額外文本)
            if json_str.startswith("```json"):
                 json_str = json_str[7:].strip()
            if json_str.endswith("```"):
                 json_str = json_str[:-3].strip()
            
            start_index = json_str.find('[')
            end_index = json_str.rfind(']')
            
            if start_index != -1 and end_index != -1 and end_index > start_index:
                 json_str = json_str[start_index : end_index + 1]
            
            # 確保 LLM 輸出的是有效的 JSON
            event_data = json.loads(json_str) 
            
            # --- 成功：將 URI 和事件 JSON 儲存在同一個檔案 ---
            final_event_data = {
                "segment_video_uri": video_uri, # 儲存影片的 GCS 路徑
                "events": event_data          # 儲存 LLM 分析的事件陣列
            }
            
            json_filename = f"{os.path.splitext(file_name)[0]}_event.json"
            output_path = os.path.join(output_folder, json_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                 json.dump(final_event_data, f, ensure_ascii=False, indent=2)

            print(f"\n✅ Stage 1 成功！事件 JSON 已儲存至：{json_filename}")

        except exceptions.GoogleAPIError as e:
            print(f"\n❌ Stage 1 失敗 (API/連線錯誤)：{file_name}, 錯誤: {e}")
            print(f"原始 LLM 輸出開頭: {json_str[:100]}...")
            continue
            
        except json.JSONDecodeError as e:
            print(f"\n❌ Stage 1 失敗 (JSON 格式錯誤)：{file_name}, 錯誤: {e}")
            print(f"原始 LLM 輸出開頭: {json_str[:100]}...")
            continue
            
        except Exception as e:
            print(f"\n❌ Stage 1 失敗 (其他錯誤)：{file_name}, 錯誤: {e}")
            print(f"原始 LLM 輸出開頭: {json_str[:100]}...")
            continue

    print("\nStage 1 事件分析完成。請檢查輸出的 _event.json 檔案。")
    return {"status": "Stage 1 completed"}

# ✅ 後端單測模式
if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    intro_text = input("請輸入影片背景介紹：")
    process_stage1_events(video_folder, output_folder, intro_text)