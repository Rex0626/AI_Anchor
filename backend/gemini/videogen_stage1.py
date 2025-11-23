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
from google.api_core import exceptions

# ========== 憑證載入 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ========== 關鍵參數 (新增) ==========
# ⚡️ 強制最小時間區塊 (秒)，小於此長度會被合併
MIN_CHUNK_DURATION = 1.5 

# ========== 工具函數 (新增與修改) ==========
def seconds_to_timecode(seconds):
    return str(timedelta(seconds=round(seconds)))

def parse_time_str(t_str):
    """解析 '0:01.2' 為秒數 (float)"""
    try:
        if not t_str: return 0.0
        parts = t_str.strip().split(':')
        sec = 0.0
        if len(parts) == 3: sec += float(parts[-3]) * 3600
        if len(parts) >= 2: sec += float(parts[-2]) * 60
        sec += float(parts[-1])
        return sec
    except: return 0.0

def format_time_str(seconds):
    """將秒數轉回 '0:01.2' 格式"""
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:04.1f}"

def enforce_min_duration(events_data):
    """
    🛡️ 強制合併邏輯：
    遍歷 LLM 生成的事件列表，如果某個區塊 < 2.0 秒，
    就將其與下一個區塊合併，直到滿足最小時長。
    """
    if not events_data: return []

    merged_events = []
    buffer = None

    for ev in events_data:
        start = parse_time_str(ev.get("start_time"))
        end = parse_time_str(ev.get("end_time"))
        
        # 確保 inner events 是列表
        if "events" not in ev: ev["events"] = []
        
        if buffer:
            # 合併進 Buffer
            buffer["end_time"] = ev["end_time"] # 延伸結束時間
            buffer["events"].extend(ev["events"]) # 合併原子動作
            
            # 重新計算 Buffer 時長
            buf_start = parse_time_str(buffer["start_time"])
            buf_end = parse_time_str(buffer["end_time"])
            
            if (buf_end - buf_start) >= MIN_CHUNK_DURATION:
                # 重新計算 time_range 字串
                buffer["time_range"] = format_time_str(buf_end - buf_start)
                merged_events.append(buffer)
                buffer = None
        else:
            duration = end - start
            if duration < MIN_CHUNK_DURATION:
                buffer = ev # 時長不足，放入 Buffer 等待下一個來救
            else:
                ev["time_range"] = format_time_str(duration)
                merged_events.append(ev)

    # 迴圈結束後，如果 Buffer 還有剩 (通常是最後一個片段)，就直接加入
    if buffer:
        # 重新計算 time_range
        b_s = parse_time_str(buffer["start_time"])
        b_e = parse_time_str(buffer["end_time"])
        buffer["time_range"] = format_time_str(b_e - b_s)
        merged_events.append(buffer)

    # 重新編號 EID (可選，讓資料更好看)
    for chunk in merged_events:
        for i, atom in enumerate(chunk["events"]):
            atom["eid"] = i + 1

    return merged_events

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

# ========== Prompt (Stage 1: Event Analysis) ==========
event_analysis_template = """ 
你是一位**客觀且極速的事件分析器**，你的任務是將影片內容分解成結構化 JSON 數據。

🎯 你的任務：分析影片中的所有動作，並輸出一個 JSON 陣列。

⛔️ **嚴格禁令 (Strict Grounding Rules) - 違者扣分：**
1.  **絕對不可腦補**：只描述**畫面中清晰可見**的動作。如果影片在球落地前就切斷了，**絕對不要**自己編造「得分」或「出界」的結果。
2.  **不可預測未來**：不要根據球員動作去猜測下一秒會發生什麼。只記錄已發生的事實。
3.  **不可添加不存在的球員**：只記錄畫面中出現的球員動作。
4.  **證據優先**：如果你不確定結果（例如球是否出界），請描述事實（如「球落地，裁判未判決」或「球落地」），不要強加結果。

📌 撰寫規則如下：
1. **輸出格式必須嚴格遵守 JSON 陣列結構**，不允許任何額外說明或文字。
2. **time_range 必須精確到毫秒**（即小數點後一位，例如：0:01.2），作為第二階段時間標記的依據。
3. 盡可能將多個同時發生的動作（例如：扣殺與防守）歸入同一個 `time_range` 內的 `events` 陣列中。
4. 輸出必須完整涵蓋該影片片段的所有動作。

必須包含以下欄位：
1. `start_time`: 開始時間,
2. `end_time`: 結束時間,
3. `time_range`: 持續時間 (結束時間 - 開始時間),
4. `events`: 該時間段的事件。
    - 每個事件必須包含以下子欄位：
    - `eid`: 該時間段的事件唯一識別碼 (從 1 開始遞增)。
    - `player`: 執行該動作的球員。
    - `action`: 具體的動作描述 (例如 "反手挑球"、"頭頂殺球")。
    - `result`: 該動作的直接後果 (例如 "球飛到底線"、"被擋回")。
    - `category`: **分類標準 (Category) - 請務必準確標記，供後端篩選：**
        1.  **Serve (發球/接發)**：包含發球動作與第一拍回球。
        2.  **Exchange (過渡/拉吊)**：普通的平抽擋、高遠球、吊球，沒有明顯得分機會的來回。
        3.  **Smash (殺球/進攻)**：具有威脅性的進攻動作（殺球、撲球）。
        4.  **Defend (防守/救球)**：面對進攻時的被動防守（挑球、魚躍救球）。
        5.  **Score (得分/結果)**：這一分的最後一球，包含得分方式（出界、掛網、落地）。
        6.  **Foul (犯規)**：觸網、發球違例等。
    - `is_crucial`: 若為 Score, Smash, Foul, Serve 則為 true，其餘為 false。

📽️ 影片背景資料如下：
{{ intro }}

JSON 輸出範例（請直接輸出 JSON 陣列）：
"segment_video_uri": "...",
{
    "start_time": "0:01.2",
    "end_time": "0:03.5",
    "time_range":"0:02.3",
    "events":[
        { 
            "eid": 1,
            "player": "Thinaah",
            "action": "發短球",
            "result": "對方上網",
            "category": "Serve",
            "is_crucial": true
        },
        { 
            "eid": 2,
            "player": "Miyau",
            "action": "網前推撲",
            "result": "球速加快",
            "category": "Exchange",
            "is_crucial": false
        },
    ],
    "start_time": "0:03.5",
    "end_time": "0:05.9",
    "time_range":"0:02.4",
    "events":[
        { 
            "eid": 1,
            "player": "Tan",
            "action": "平抽擋回",
            "result": "多拍僵持",
            "category": "Exchange",
            "is_crucial": false
        },
        { 
            "eid": 2,
            "player": "Sakuramoto",
            "action": "起跳重殺",
            "result": "球速極快",
            "category": "Smash",
            "is_crucial": true
        },
    ]
}
"""

prompt_builder_event = PromptBuilder(
    template=event_analysis_template,
    required_variables=["intro"]
)

# ========== Pipeline (Stage 1) ==========
# --- 組件實例化 ---
upload2gcs = Upload2GCS(bucket_name="ai_anchor")
add_video_2_prompt = AddVideo2Prompt()
gemini_generator = GeminiGenerator(
    project_id="ai-anchor-462506",
    location="us-central1",
    model="gemini-2.5-pro"
)

# --- Pipeline 1: Upload ---
pipeline_upload = Pipeline()
pipeline_upload.add_component(instance=upload2gcs, name="upload2gcs")

# --- Pipeline 2: Event Analysis ---
pipeline_event_analysis = Pipeline()
pipeline_event_analysis.add_component(instance=prompt_builder_event, name="prompt_builder") 
pipeline_event_analysis.add_component(instance=add_video_2_prompt, name="add_video")
pipeline_event_analysis.add_component(instance=gemini_generator, name="llm")

pipeline_event_analysis.connect("prompt_builder", "add_video")
pipeline_event_analysis.connect("add_video.prompt", "llm")

# ========== 主邏輯 ==========
def process_stage1_events(video_folder, output_folder, intro_text):
    os.makedirs(output_folder, exist_ok=True)
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    
    for file_name in tqdm(video_files, desc="[AI主播] Stage 1 事件分析"):
        segment_path = os.path.join(video_folder, file_name)
        json_str = ""
        video_uri = ""

        try:
            # --- Step 1: Upload ---
            upload_input = {"upload2gcs": {"file_path": segment_path}}
            upload_result = pipeline_upload.run(upload_input)
            video_uri = upload_result["upload2gcs"]["uri"]

            # --- Step 2: Analyze ---
            prompt_input_event = {
                "add_video": {"uri": video_uri},
                "prompt_builder": {"intro": intro_text}
            }
            event_result = pipeline_event_analysis.run(prompt_input_event)
            
            replies = event_result["llm"]["replies"]
            if not replies:
                print(f"\n⚠️ Stage 1 警告：LLM 未回傳任何內容。跳過：{file_name}")
                continue
            
            json_str = replies[0].strip()
            
            # Cleanup
            if json_str.startswith("```json"): json_str = json_str[7:].strip()
            if json_str.endswith("```"): json_str = json_str[:-3].strip()
            start_index = json_str.find('[')
            end_index = json_str.rfind(']')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                 json_str = json_str[start_index : end_index + 1]
            
            # Parse JSON
            event_data = json.loads(json_str) 
            
            # 💡【關鍵修改】呼叫強制合併函數，確保最小時長 2.0 秒
            processed_events = enforce_min_duration(event_data)
            
            # Save
            final_event_data = {
                "segment_video_uri": video_uri,
                "events": processed_events # 儲存處理過的資料
            }
            
            json_filename = f"{os.path.splitext(file_name)[0]}_event.json"
            output_path = os.path.join(output_folder, json_filename)
            with open(output_path, "w", encoding="utf-8") as f:
                 json.dump(final_event_data, f, ensure_ascii=False, indent=2)

            print(f"\n✅ Stage 1 成功！(已執行2秒合併) 檔案：{json_filename}")

        except exceptions.GoogleAPIError as e:
            print(f"\n❌ API 錯誤：{file_name}, {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON 錯誤：{file_name}, {e}")
            print(f"原始輸出: {json_str[:100]}...")
            continue
        except Exception as e:
            print(f"\n❌ 未知錯誤：{file_name}, {e}")
            continue

    print("\nStage 1 事件分析完成。")
    return {"status": "Stage 1 completed"}

if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    intro_text = input("請輸入影片背景介紹：")
    process_stage1_events(video_folder, output_folder, intro_text)