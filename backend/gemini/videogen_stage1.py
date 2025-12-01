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
你不受限於單一運動規則，而是專注於捕捉畫面中的**關鍵動作**與**競技狀態變化**。

2. 影片背景資料 (Video Context)
以下資訊包含球員身分與當前賽況，僅供辨識與標記使用：
{{ intro }}
*(注意：若畫面模糊或球員未出現，請使用「畫面遠端球員」或「對手」等客觀描述，勿強行填入人名。)*

3. 應該要做的事 (Tasks)
- **時間標記**：每個事件都必須有明確的 `start_time` (動作開始) 和 `end_time` (動作結束/球落地)，如果是瞬間動作，`end_time` 設為 `start_time` 加 1 秒。
- **通用性**：無論是單人運動 (羽球) 還是團體運動 (籃球)，請依據比賽節奏記錄關鍵事件。
- **完整覆蓋**：請依時間順序記錄，從影片開始到結束，不要遺漏任何具備戰術意義的動作。
- **重複確認**：每一個影片片段開始前請務必確認{{intro}}的資料，確保你理解參賽者身份與賽況。
- **語言設定**：請使用我指定的語言填寫JSON的欄位。
- **正確性**：請說出實際發生的事件，不要自己加不存在的事件。

4. 禁止做的事 (Strict Prohibitions)
⛔️ **嚴格禁令 (違者導致系統錯誤)：**
- **禁止提前收工 (No Early Stop)**：絕對不可以只分析前 10 秒就停止！請務必檢查影片總長度，確保你的日誌覆蓋了整個片段。
- **禁止遺漏細節**：不要因為動作平淡就跳過。平抽擋和防守也是比賽的一部分，必須記錄，如果不確定細節就不用描述。
- **絕對不可腦補**：如果影片在球落地前切斷（Cut），**絕對不要**標記為 `Score`。
- **不可預測未來**：不要因為球員慶祝就判定得分，除非親眼看到球落地。
- **禁止 Markdown**：直接輸出 JSON 陣列。

5. JSON 欄位定義 (Field Definitions)
輸出一個 JSON 陣列，每個物件需包含：
- `start_time`: (String) 動作開始時間(時間格式：HH:MM:SS.s，只能到小數點後一位)。
- `end_time`: (String) 動作結束時間，若是瞬間擊球，設定為start_time+1(時間格式：HH:MM:SS.s，只能到小數點後一位)。
- `player`: (String) 執行動作的主體 (球員名、隊名)。
- `action`: (String) 具體動作名稱 (如: 殺球, 三分出手)。
- `detail`: (String, Optional) 動作細節描述。對於關鍵球或精彩動作，請務必描述軌跡或質量 (如: "貼網而過", "滑拍假動作")；對於普通來回可留空。
- `category`: (String) **分類標準 (請嚴格遵守以下通用分類)：**
    1. **Start**: 發球/開球/比賽開始。
    2. **Setup**: 組織/過渡 (如：籃球運球、足球傳導)。
    3. **Exchange**: 平抽/來回 (羽球/網球專用，雙方互有來回但未明顯進攻)。
    4. **Offense**: 進攻 (殺球、射門、具有威脅性的動作)。
    5. **Defense**: 防守 (挑球、救球、火鍋、撲救)。
    6. **Score**: 得分/死球/結果 (球落地、進球)。
    7. **Foul**: 犯規/中斷/出界。
    8. **End**: 比賽結束/局末。
- **注意**：若無法歸類，請選擇最接近的分類，切勿新增分類。
- `is_crucial`: (Boolean) 是否為高光時刻 (得分、精彩撲救、關鍵失誤為 true)。

6. JSON 輸出範例 (Example)
[
    {
      "start_time": "0:00.0",
      "end_time": "0:02.0",
      "player": "戴資穎",
      "action": "反手發短球",
      "detail": "貼網而過，質量極高",
      "category": "Serve",
      "is_crucial": true
    },
    {
      "start_time": "0:02.1",
      "end_time": "0:03.5",
      "player": "陳雨菲",
      "action": "正手挑高球",
      "detail": "被動防守至底線",
      "category": "Defense",
      "is_crucial": false
    },
    {
      "start_time": "0:03.6",
      "end_time": "0:04.2",
      "player": "戴資穎",
      "action": "直線殺球",
      "detail": "速度極快，落地得分",
      "category": "Offense",
      "is_crucial": true
    },
    {
      "start_time": "0:04.3",
      "end_time": "0:04.5",
      "player": "無",
      "action": "界內得分",
      "detail": "對手無法觸球",
      "category": "Score",
      "is_crucial": true
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
            "intro": intro_text,
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