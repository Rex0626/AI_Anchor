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
1. 角色 (Role)
你是一個**嚴格且不知疲倦的電腦視覺動作捕捉系統 (Computer Vision Motion Capture System)**。
你的任務是將影片內容轉換為「原子化」的動作日誌。你關注的是**毫秒級的像素變化**，絕不放過任何一秒。

2. 影片背景資料 (Video Context)
以下資訊包含球員身分與當前賽況，僅供辨識與標記使用：
{{ intro }}
*(注意：若畫面模糊或球員未出現，請使用「畫面遠端球員」或「對手」等客觀描述，勿強行填入人名。)*

3. 應該要做的事 (Tasks)
- **全程覆蓋 (Full Coverage)**：**必須分析到影片的最後一秒！** 即使發生了得分 (`Score`)，只要影片還沒結束，**必須繼續偵測下一球的發球 (`Serve`) 或準備動作**。
- **原子化紀錄**：不要聚合動作！每一次揮拍都是獨立事件。
- **填補時間軸**：請仔細檢查時間軸，**不要跳過中間的平抽擋 (`Exchange`)**。若兩個事件之間有超過 3 秒的空白，請再次確認是否有遺漏的過渡動作。
- **精確時間**：`time_range` 必須精確到毫秒。
- **完整覆蓋**：從影片開始到結束，所有可見的動作都必須被記錄。
- **動作描述**：直接在 `action` 中描述動作及其可見後果。

4. 禁止做的事 (Strict Prohibitions)
⛔️ **嚴格禁令 (違者導致系統錯誤)：**
- **禁止提前收工 (No Early Stop)**：絕對不可以只分析前 10 秒就停止！請務必檢查影片總長度，確保你的日誌覆蓋了整個片段。
- **禁止遺漏細節**：不要因為動作平淡就跳過。平抽擋和防守也是比賽的一部分，必須記錄。
- **絕對不可腦補**：如果影片在球落地前切斷（Cut），**絕對不要**標記為 `Score`。
- **不可預測未來**：不要因為球員慶祝就判定得分，除非親眼看到球落地。
- **禁止 Markdown**：直接輸出 JSON 陣列。

5. JSON 欄位定義 (Field Definitions)
必須包含：`start_time`, `end_time`, `time_range`, `events`。
`events` 中的每個物件需包含：
- `eid`: 事件序號 (1, 2...)
- `player`: 執行動作的球員。
- `action`: 具體動作描述。
    * 若動作有明確結果，請合併描述 (如 "正手殺球界外", "發短球掛網")。
    * 若影片中斷，僅描述動作本身 (如 "正手殺球", "飛身救球")。
- `category`: **分類標準 (請嚴格遵守以下通用分類)：**
    1. **Serve**: 發球/開球 (比賽開始)。
    2. **Exchange**: 球權交換/平抽擋/過渡 (雙方互有來回，未明顯進攻)。
    3. **Attack**: 進攻 (殺球、撲球、具有威脅性的擊球)。
    4. **Defend**: 防守 (挑球、救球、被動擋網)。
    5. **Score**: 得分/死球 (球落地、出界、掛網)。**若畫面中斷導致結果未知，禁止選此項。**
    6. **Foul**: 犯規 (觸網、發球違例)。
- `is_crucial`: **關鍵事件判定標準：**
    - `true`: 僅限 **Serve** (發球), **Score** (得分), **Foul** (犯規)。因為這些時刻比賽會停頓或重啟。
    - `false`: 其他所有分類 (Exchange, Attack, Defend)。

6. JSON 輸出範例 (Example)
[
    {
        "start_time": "0:01.2",
        "end_time": "0:02.0",
        "time_range": "0:00.8",
        "events": [
            { 
                "eid": 1,
                "player": "Thinaah",
                "action": "發短球",
                "category": "Serve",
                "is_crucial": true
            }
        ]
    },
    {
        "start_time": "0:02.0",
        "end_time": "0:02.8",
        "time_range": "0:00.8",
        "events": [
            { 
                "eid": 2,
                "player": "Miyau",
                "action": "反手挑球",
                "category": "Defend",
                "is_crucial": false
            }
        ]
    }
]

**請現在開始分析，務必堅持分析到影片最後一秒，直接輸出 JSON 陣列：**
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
        
        processed_events = event_data
        
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