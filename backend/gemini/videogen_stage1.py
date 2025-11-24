import os
import json
import time
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from google.cloud import storage
from tqdm import tqdm
from google.api_core import exceptions

# ========== 1. 設定與憑證 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ========== 2. 關鍵參數 ==========
MIN_CHUNK_DURATION = 2.0 

# ========== 3. 工具函數 ==========
def parse_time_str(t_str):
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
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:04.1f}"

def enforce_min_duration(events_data):
    if not events_data: return []
    merged_events = []
    buffer = None
    for ev in events_data:
        if "events" not in ev: ev["events"] = []
        if buffer:
            buffer["end_time"] = ev["end_time"]
            buffer["events"].extend(ev["events"])
            b_start = parse_time_str(buffer["start_time"])
            b_end = parse_time_str(buffer["end_time"])
            if (b_end - b_start) >= MIN_CHUNK_DURATION:
                buffer["time_range"] = format_time_str(b_end - b_start)
                merged_events.append(buffer)
                buffer = None
        else:
            start = parse_time_str(ev.get("start_time"))
            end = parse_time_str(ev.get("end_time"))
            if (end - start) < MIN_CHUNK_DURATION:
                buffer = ev 
            else:
                ev["time_range"] = format_time_str(end - start)
                merged_events.append(ev)
    if buffer:
        b_s = parse_time_str(buffer["start_time"])
        b_e = parse_time_str(buffer["end_time"])
        buffer["time_range"] = format_time_str(b_e - b_s)
        merged_events.append(buffer)
    for chunk in merged_events:
        for i, atom in enumerate(chunk["events"]):
            atom["eid"] = i + 1
    return merged_events

# ========== 4. 組件與 Pipeline 初始化 (全域單次) ==========
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
        self.project_id, self.location, self.model = project_id, location, model
    @component.output_types(replies=list)
    def run(self, prompt: list):
        generator = VertexAIGeminiGenerator(project_id=self.project_id, location=self.location, model=self.model)
        return {"replies": generator.run(prompt)["replies"]}

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

prompt_builder_event = PromptBuilder(template=event_analysis_template, required_variables=["intro"])

# 初始化 Pipeline
upload2gcs = Upload2GCS(bucket_name="ai_anchor")
pipeline_upload = Pipeline()
pipeline_upload.add_component(instance=upload2gcs, name="upload2gcs")

add_video_2_prompt = AddVideo2Prompt()
gemini_generator = GeminiGenerator(project_id="ai-anchor-462506", location="us-central1", model="gemini-2.5-flash")
pipeline_event_analysis = Pipeline()
pipeline_event_analysis.add_component(instance=prompt_builder_event, name="prompt_builder") 
pipeline_event_analysis.add_component(instance=add_video_2_prompt, name="add_video")
pipeline_event_analysis.add_component(instance=gemini_generator, name="llm")
pipeline_event_analysis.connect("prompt_builder", "add_video")
pipeline_event_analysis.connect("add_video.prompt", "llm")

# ========== 5. 核心功能：處理單一影片 ==========
def process_single_video_stage1(video_path, output_folder, intro_text):
    """
    處理單一影片：上傳 -> 分析 -> 存檔
    回傳：成功生成的 JSON 路徑 (若失敗回傳 None)
    """
    os.makedirs(output_folder, exist_ok=True)
    file_name = os.path.basename(video_path)
    
    try:
        # Step 1: Upload
        upload_result = pipeline_upload.run({"upload2gcs": {"file_path": video_path}})
        video_uri = upload_result["upload2gcs"]["uri"]

        # Step 2: Analyze
        event_result = pipeline_event_analysis.run({
            "add_video": {"uri": video_uri},
            "prompt_builder": {"intro": intro_text}
        })
        
        replies = event_result["llm"]["replies"]
        if not replies:
            print(f"⚠️ [Stage 1] 無回傳: {file_name}")
            return None
        
        json_str = replies[0].strip()
        if json_str.startswith("```json"): json_str = json_str[7:].strip()
        if json_str.endswith("```"): json_str = json_str[:-3].strip()
        start_index = json_str.find('[')
        end_index = json_str.rfind(']')
        if start_index != -1 and end_index != -1:
             json_str = json_str[start_index : end_index + 1]
        
        event_data = json.loads(json_str) 
        
        # 強制合併邏輯
        processed_events = enforce_min_duration(event_data)
        
        final_event_data = {
            "segment_video_uri": video_uri,
            "events": processed_events
        }
        
        json_filename = f"{os.path.splitext(file_name)[0]}_event.json"
        output_path = os.path.join(output_folder, json_filename)
        with open(output_path, "w", encoding="utf-8") as f:
             json.dump(final_event_data, f, ensure_ascii=False, indent=2)

        return output_path

    except Exception as e:
        print(f"❌ [Stage 1 錯誤] {file_name}: {e}")
        return None

# ========== 6. 獨立運行模式 (批次處理資料夾) ==========
if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    intro_text = input("請輸入影片背景介紹：") or "羽球比賽"
    
    print(f"\n🚀 [獨立模式] Stage 1 批次啟動...")
    
    if os.path.exists(video_folder):
        files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
        for f in tqdm(files, desc="Processing"):
            path = os.path.join(video_folder, f)
            res = process_single_video_stage1(path, output_folder, intro_text)
            if res: print(f"  -> Saved: {os.path.basename(res)}")
    else:
        print("❌ 找不到影片資料夾")