import os
import json
import re
from datetime import timedelta
from moviepy.editor import VideoFileClip
from vertexai.generative_models import Part
from haystack_integrations.components.generators.google_vertex import VertexAIGeminiGenerator
from haystack import component, Pipeline
from haystack.components.builders import PromptBuilder
from tqdm import tqdm
from google.api_core import exceptions

# ========== 憑證載入 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ✅ 工具函數
def seconds_to_timecode(seconds):
    return str(timedelta(seconds=round(seconds, 3)))

def timecode_to_seconds(time_str):
    try:
        parts = time_str.split(':')
        seconds = 0.0
        if len(parts) == 3: seconds += float(parts[-3]) * 3600
        if len(parts) >= 2: seconds += float(parts[-2]) * 60
        seconds += float(parts[-1])
        return seconds
    except ValueError: return 0.0

# ========== 組件 ==========
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

# ========== Prompt (Stage 2: Pro Commentator - Summary & Highlight) ===========
narrative_template = """ 
你是一位專業的體育主播。你的任務是根據**影片畫面**和**結構化事件數據 (event_data)**，生成精準的旁白。

🎯 **核心目標：**
1. **關鍵必說：** 所有 `is_crucial: true` 或 `result: "得分"` 的事件**必須**播報。
2. **空白填補：** 在關鍵事件之間的空白時間段，請**自行決定**是否播報 `is_crucial: false` 的次要事件，或是進行戰況回顧，以保持解說的流暢度（不要讓觀眾覺得冷場）。
3. **嚴格限時：** 每個時間段**只能**有一句旁白，且必須在時間內說完。

📌 **撰寫規則 (絕對遵守)：**
1. **輸出格式：** 每一行必須以 **[MM:SS.f]** 結尾時間戳開頭。
   範例：`[00:04.5] 【平穩】馬來西亞隊的Thinaah發球。`

2. **【字數限制公式】：** - 請計算每個事件的持續時間 (`time_range` 的結束時間 - 開始時間)。
   - **最大字數 = 持續時間(秒) × 4**。
   - *例如：事件持續 2 秒，旁白絕不能超過 8 個字。*

3. **【內容策略】：**
   - **優先級 1 (最高)：** `is_crucial: true` 的事件。直接描述動作與結果。
   - **優先級 2 (填補)：** 若兩個關鍵事件之間間隔超過 3 秒，請從 `is_crucial: false` 事件中挑選一個來描述，或者簡單回顧上一球的精彩處。
   - **優先級 3 (忽略)：** 若時間段太短 (小於 1 秒) 且非關鍵，請直接忽略，保持沉默。

4. **數據優先：** 描述選手名字、動作、位置時，必須優先使用 `event_data` 中的資訊，不可隨意更改選手姓名。
5. **情緒標籤：** 必須加入【平穩】、【緊張】、【激動】。
6. **不用回覆**，直接輸出旁白句子。

📊 **結構化事件數據 (Stage 1 輸出)：**
{{ event_data }}

請根據影片和上述數據，輸出旁白：
"""

prompt_builder = PromptBuilder(template=narrative_template, required_variables=["event_data"])

# ========== Pipeline ==========
pipeline = Pipeline()
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=AddVideo2Prompt(), name="add_video")
pipeline.add_component(instance=GeminiGenerator(project_id="ai-anchor-462506", location="us-central1", model="gemini-2.5-flash"), name="llm")
pipeline.connect("prompt_builder.prompt", "add_video.prompt")
pipeline.connect("add_video.prompt", "llm.prompt")

# ========== 主邏輯 ==========
def process_stage2_narratives(video_folder, event_json_folder, output_folder, reaction_lag_sec=0.2):
    os.makedirs(output_folder, exist_ok=True)
    event_files = sorted([f for f in os.listdir(event_json_folder) if f.endswith("_event.json")])

    for file_name in tqdm(event_files, desc="[Stage 2] 敘事生成"):
        event_path = os.path.join(event_json_folder, file_name)
        base_name = file_name.replace("_event.json", "")
        video_path = os.path.join(video_folder, f"{base_name}.mp4")

        # 1. 讀取影片時長
        try:
            with VideoFileClip(video_path) as clip: duration = clip.duration
        except Exception as e:
            print(f"\n❌ [Stage 2] 無法讀取影片時長：{video_path}, {e}")
            continue

        # 2. 執行 LLM 生成
        try:
            with open(event_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            res = pipeline.run({
                "add_video": {"uri": data["segment_video_uri"]},
                "prompt_builder": {"event_data": json.dumps(data["events"], ensure_ascii=False)}
            })
            reply = res["llm"]["replies"][0].strip()
            
            # [DEBUG] 印出 LLM 回覆的前 200 字，確認它到底產生了什麼
            # print(f"\n🔍 [DEBUG] {file_name} LLM 原始回覆 (前200字): {reply[:200]}...") 

        except Exception as e:
            print(f"\n❌ [Stage 2] API/連線錯誤：{file_name}, {e}")
            continue

        # 3. 解析 (使用容錯率更高的 Regex)
        commentary = []
        last_end_time_sec = 0.0
        
        lines = reply.split("\n")

        for line in lines:
            line = line.strip()
            
            # <<<< 修改這裡：升級版 Regex，分離 [時間段]、【情緒】、內容 >>>>
            # 支援格式： [00:00.0-00:05.0] 【情緒】 內容
            m = re.search(r'\[(\d{1,2}:\d{2}(?:\.\d*)?)\s*[-~]\s*(\d{1,2}:\d{2}(?:\.\d*)?)\]\s*[【\[](.*?)[】\]]\s*(.*)', line)
            
            if m:
                s_str, e_str, emotion, text_content = m.groups()

                # 如果 LLM 忘記寫情緒，預設為 "平穩"
                if not emotion:
                    emotion = "平穩"
                
                s_sec = timecode_to_seconds(s_str)
                e_sec = timecode_to_seconds(e_str)
                
                # 時間校準 (加入反應延遲)
                final_start = max(s_sec + reaction_lag_sec, last_end_time_sec)
                final_end = min(e_sec + reaction_lag_sec + 0.5, duration)
                
                if final_end - final_start < 0.5: final_end = final_start + 0.5
                
                if final_start < duration:
                    # 儲存結果 (包含 emotion)
                    current_entry = {
                        "start_time": seconds_to_timecode(final_start),
                        "end_time": seconds_to_timecode(final_end),
                        "emotion": emotion,
                        "text": text_content.strip()
                    }
                    commentary.append(current_entry)
                    last_end_time_sec = final_start

        # 4. 儲存結果
        if commentary:
            with open(os.path.join(output_folder, f"{base_name}.json"), "w", encoding="utf-8") as f:
                json.dump({"segment": f"{base_name}.mp4", "commentary": commentary}, f, ensure_ascii=False, indent=2)
        else:
            # 如果還是失敗，印出完整回覆以供檢查
            print(f"\n⚠️ [Stage 2] 警告：{file_name} 無有效旁白。")
            print(f"🔴 [DEBUG] 完整的 LLM 回覆:\n{reply}\n" + "="*30)

# ✅ 後端單測模式
if __name__ == "__main__":
    # 原始影片資料夾 (用於獲取時長)
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    # Stage 1 產生的事件 JSON 資料夾
    event_json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    # Stage 2 最終旁白輸出的資料夾
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"

    process_stage2_narratives(video_folder, event_json_folder, output_folder)