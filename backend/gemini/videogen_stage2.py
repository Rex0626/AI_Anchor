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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

# ========== Prompt (智能合併版) ==========
narrative_template = """ 
你是一位專業體育主播。根據影片和事件數據生成旁白。

🎯 **核心原則：流暢敘事**
不要為每個微小動作配音。將連續動作組合成完整故事。

📌 **撰寫規則：**
1. **格式：** `[開始時間-結束時間]` 開頭。
2. **【智能合併】：** 連續且相關的短事件(如倒地->起身)必須合併成一句完整描述。
3. **【數據優先】：** 嚴禁修改 `event_data` 中的選手名字。
4. **【關鍵必說】：** `is_crucial: true` 事件必須包含。
5. **【字數】：** 合併後的句子，每1秒不超過4個字。
6. **情緒：** 加入【平穩】、【緊張】、【激動】。

📊 **事件數據：**
{{ event_data }}

請輸出合併優化後的旁白：
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

        try:
            with VideoFileClip(video_path) as clip: duration = clip.duration
            with open(event_path, 'r', encoding='utf-8') as f: data = json.load(f)
            
            res = pipeline.run({
                "add_video": {"uri": data["segment_video_uri"]},
                "prompt_builder": {"event_data": json.dumps(data["events"], ensure_ascii=False)}
            })
            reply = res["llm"]["replies"][0].strip()
        except Exception as e:
            print(f"\n❌ [Stage 2] 錯誤：{file_name}, {e}")
            continue

        # 解析
        commentary = []
        last_end = 0.0
        for line in reply.split("\n"):
            m = re.match(r'^\[(\d+:\d+\.?\d*)\s*-\s*(\d+:\d+\.?\d*)\]\s*(【.*?】.*)', line.strip())
            if m:
                s_sec = timecode_to_seconds(m.group(1))
                e_sec = timecode_to_seconds(m.group(2))
                
                final_s = max(s_sec + reaction_lag_sec, last_end)
                final_e = min(e_sec + reaction_lag_sec + 0.5, duration)
                
                if final_s < final_e:
                    commentary.append({
                        "start_time": seconds_to_timecode(final_s),
                        "end_time": seconds_to_timecode(final_e),
                        "text": m.group(3)
                    })
                    last_end = final_s # 允許些微重疊或緊接

        if commentary:
            with open(os.path.join(output_folder, f"{base_name}.json"), "w", encoding="utf-8") as f:
                json.dump({"segment": f"{base_name}.mp4", "commentary": commentary}, f, ensure_ascii=False, indent=2)
        else:
            print(f"⚠️ [Stage 2] 警告：{file_name} 無有效旁白。")

# ✅ 後端單測模式
if __name__ == "__main__":
    # 原始影片資料夾 (用於獲取時長)
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    # Stage 1 產生的事件 JSON 資料夾
    event_json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    # Stage 2 最終旁白輸出的資料夾
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"

    process_stage2_narratives(video_folder, event_json_folder, output_folder)