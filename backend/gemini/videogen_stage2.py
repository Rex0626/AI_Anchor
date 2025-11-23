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

# ========== 1. 設定與憑證 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cred_path = os.path.join(PROJECT_ROOT, "credentials", "ai-anchor-462506-7887b7105f6a.json")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path

# ========== 2. 關鍵參數 ==========
SYLLABLES_PER_SEC = 4.0       # 語速控制 (音節/秒)
MIN_EVENT_DURATION = 1.0      # 最小時長 (小於此秒數強制合併)
MAX_RALLY_DURATION = 4.5      # 最大合併時長 (超過此秒數強制切斷，避免冷場)

# ========== 3. 工具函數 ==========
def seconds_to_timecode(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}"

def format_duration(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}:{s:04.1f}"

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

def estimate_speech_time(text):
    if not text: return 0.0
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
    text_no_zh = re.sub(r'[\u4e00-\u9fff]', ' ', text)
    text_clean_en = re.sub(r'[^\w\s]', '', text_no_zh)
    english_words = text_clean_en.split()
    count_punc = len(re.findall(r'[，。！,.]', text))
    total_units = (len(chinese_chars) * 1.0) + (len(english_words) * 1.3) + (count_punc * 0.4)
    return total_units / SYLLABLES_PER_SEC

# ========== 4. Haystack 組件 ==========
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

# ========== 5. Prompt 模板 (新增人名強調規則) ==========
# ========== 5. Prompt 模板 (數據 + 視覺雙重驅動) ==========
narrative_template = """ 
你是一位**資深、節奏明快**的羽球賽事即時主播。
現在你有兩個資訊來源：
1.  **比賽影片**：請觀察畫面中的精彩細節、球員情緒與擊球力道。
2.  **時間區塊列表 (JSON)**：這是精準的動作紀錄與音節限制。

🎯 **你的任務：視覺與數據的完美結合**
請根據 JSON 的指引鎖定時間段，並**觀看影片**來豐富你的解說。

**八大黃金規則 (請嚴格遵守)：**
1.  **極簡風格**：使用「球員+動作」的短語模式 (如：Miyau挑高、Tan殺球)。
2.  **資料正確性 (重要)**：**人名**與**動作類型**必須直接使用輸入資料中的詞彙！(JSON 是事實基準)。
3.  **人名重述 (重要)**：每隔 1-2 個短句，或者在攻防轉換時，**務必帶上球員名字**。
    * ❌ 不好：殺球！挑高！又殺球！
    * ✅ 完美：Sakura殺球！Tan挑高！Miyau再扣殺！
4.  **嚴格限長**：若限制很短 (如 < 6)，絕不能寫長句，請用單詞 (如：得分！)。
5.  **重複即總結**：如果包含重複動作 (如：殺球->挑球->殺球)，請改用**總結說明** (如：「雙方激烈攻防！」)。
6.  **不重複**：上下時段內容若相似，請換個說法或加語氣詞。
7.  **完整性**：句子必須是完整的「球員+動作」結構，不要留下只有名字的斷句。
    * ❌ 錯誤：櫻本再殺！陳康
    * ✅ 正確：櫻本再殺！陳康擋網！
8.  **視覺加分 (Visuals)**：請觀察影片細節，加入**形容詞**或**情緒**，但不要改變動作本質。
    * ❌ 平淡：Tan殺球。
    * ✅ 生動：Tan**躍起重扣**！ (觀察到跳很高)
    * ✅ 生動：Miyau**極限**救球！ (觀察到動作很勉強)

📌 **輸入資料範例：**
* ID: 0 | 限制: 20 | 內容: 殺球 -> 挑球 -> 殺球 -> 擋網
* ID: 1 | 限制: 4 | 內容: 殺球 (得分)

📌 **輸出格式 (JSON 陣列)：**
注意：只需要回傳 ID 和 Text。
[
  {"id": 0, "text": "Sakuramoto連續猛攻，Tan頑強擋下！"},
  {"id": 1, "text": "殺球得分！"}
]

📊 **待處理列表 (請依此為準)：**
{{ event_data }}

請輸出 JSON：
"""

prompt_builder = PromptBuilder(template=narrative_template, required_variables=["event_data"])

# ========== 6. Pipeline ==========
pipeline = Pipeline()
pipeline.add_component(instance=prompt_builder, name="prompt_builder")
pipeline.add_component(instance=AddVideo2Prompt(), name="add_video")
pipeline.add_component(instance=GeminiGenerator(project_id="ai-anchor-462506", location="us-central1", model="gemini-2.5-pro"), name="llm")
pipeline.connect("prompt_builder.prompt", "add_video.prompt")
pipeline.connect("add_video.prompt", "llm.prompt")

# ========== 7. 主邏輯 ==========
def process_stage2_narratives(video_folder, event_json_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    event_files = sorted([f for f in os.listdir(event_json_folder) if f.endswith("_event.json")])

    for file_name in tqdm(event_files, desc="[Stage 2] 敘事生成"):
        event_path = os.path.join(event_json_folder, file_name)
        base_name = file_name.replace("_event.json", "")
        video_path = os.path.join(video_folder, f"{base_name}.mp4")
        
        try:
            with VideoFileClip(video_path) as clip: total_duration = clip.duration
        except: total_duration = 30.0 

        try:
            with open(event_path, 'r', encoding='utf-8') as f: 
                data = json.load(f)
                nested_events = data.get("events", []) if isinstance(data, dict) else data
                video_uri = data.get("segment_video_uri", "") if isinstance(data, dict) else ""
        except: continue

        if not nested_events: continue

        # --- A. 數據準備 & 聚合 (Aggregation) ---
        RALLY_TYPES = ["Exchange", "Smash", "Defend"]
        chunk_events = [] 
        global_id_counter = 0

        # 1. INTRO
        first_chunk_start = parse_time_str(nested_events[0].get("start_time", "0:00.0"))
        intro_limit = int(first_chunk_start * SYLLABLES_PER_SEC)
        if intro_limit >= 8:
             chunk_events.append({
                "global_id": "INTRO",
                "start_sec": 0.0,
                "end_sec": first_chunk_start,
                "limit": intro_limit,
                "info": "開場空白"
            })

        last_event_end = 0.0
        
        # 2. 處理與聚合
        buffer_chunk = None

        for chunk in nested_events:
            chunk_start = parse_time_str(chunk.get("start_time", "0:00.0"))
            chunk_end = parse_time_str(chunk.get("end_time", "0:00.0"))
            inner_list = chunk.get("events", [])
            if not inner_list: continue
            
            actions_str = " -> ".join([f"{ev.get('player')}{ev.get('action')}" for ev in inner_list])
            
            is_pure_rally = all(ev.get('category') in RALLY_TYPES for ev in inner_list) and \
                            not any(ev.get('category') == 'Score' for ev in inner_list)

            current_chunk = {
                "start": chunk_start,
                "end": chunk_end,
                "info": actions_str,
                "is_rally": is_pure_rally
            }

            # --- 聚合核心邏輯 (含時長限制) ---
            if buffer_chunk:
                # 計算若合併後的預估總時長
                potential_dur = current_chunk["end"] - buffer_chunk["start"]
                
                # 條件：類型相同 且 合併後不超過最大時長 (例如 6秒)
                is_mergeable = (buffer_chunk["is_rally"] and 
                                current_chunk["is_rally"] and 
                                potential_dur <= MAX_RALLY_DURATION)

                if is_mergeable:
                    # 合併
                    buffer_chunk["end"] = current_chunk["end"]
                    buffer_chunk["info"] += f" -> {current_chunk['info']}"
                else:
                    # 不合併，先輸出 Buffer
                    dur = buffer_chunk["end"] - buffer_chunk["start"]
                    limit = int(dur * SYLLABLES_PER_SEC)
                    
                    # 檢查最小時長 (避免碎嘴)
                    if limit > 3:
                        chunk_events.append({
                            "global_id": global_id_counter,
                            "start_sec": buffer_chunk["start"],
                            "end_sec": buffer_chunk["end"],
                            "limit": limit,
                            "info": buffer_chunk["info"]
                        })
                        global_id_counter += 1
                    
                    buffer_chunk = current_chunk
            else:
                buffer_chunk = current_chunk

            last_event_end = max(last_event_end, chunk_end)

        # 迴圈結束清空
        if buffer_chunk:
            dur = buffer_chunk["end"] - buffer_chunk["start"]
            limit = int(dur * SYLLABLES_PER_SEC)
            if limit > 3:
                chunk_events.append({
                    "global_id": global_id_counter,
                    "start_sec": buffer_chunk["start"],
                    "end_sec": buffer_chunk["end"],
                    "limit": limit,
                    "info": buffer_chunk["info"]
                })
                global_id_counter += 1

        # 3. OUTRO
        outro_dur = total_duration - last_event_end
        outro_limit = int(outro_dur * SYLLABLES_PER_SEC)
        if outro_limit >= 8:
            chunk_events.append({
                "global_id": "OUTRO",
                "start_sec": last_event_end,
                "end_sec": total_duration,
                "limit": outro_limit,
                "info": "結尾空白"
            })

        # --- B. 呼叫 LLM ---
        llm_input_data = []
        for e in chunk_events:
            llm_input_data.append({
                "id": e["global_id"], 
                "constraint": f"限 {e['limit']} 音節", 
                "content": e["info"]
            })
        
        try:
            res = pipeline.run({
                "add_video": {"uri": video_uri},
                "prompt_builder": {"event_data": json.dumps(llm_input_data, ensure_ascii=False, indent=2)}
            })
            reply = res["llm"]["replies"][0].strip()
            
            if reply.startswith("```"):
                reply = reply.split("\n", 1)[1]
                if reply.endswith("```"):
                    reply = reply.rsplit("\n", 1)[0]
            
            generated_list = json.loads(reply)
            generated_map = {str(item["id"]): item["text"] for item in generated_list}

        except Exception as e:
            print(f"❌ 生成失敗: {e}")
            continue

        # --- C. 輸出結果 (含去重) ---
        commentary = []

        for chunk in chunk_events:
            gid = str(chunk["global_id"])
            text_content = generated_map.get(gid)

            if not text_content: continue 

            duration = chunk["end_sec"] - chunk["start_sec"]
            
            # ⚡️⚡️⚡️ [新增] 寬容截斷邏輯 ⚡️⚡️⚡️
            estimated_dur = estimate_speech_time(text_content)
            
            # 設定寬容上限：允許 TTS 加速到 1.2 倍
            # 如果預估時間 > (實際時間 * 1.2)，代表加速也救不回來，必須砍字
            max_acceptable_dur = duration * 1.2 
            
            if estimated_dur > max_acceptable_dur:
                # 計算安全長度 (砍到剛好可以用 1.2 倍速念完的長度)
                ratio = max_acceptable_dur / estimated_dur
                safe_length = int(len(text_content) * ratio)
                text_content = text_content[:safe_length]
                
                # 修飾：避免斷在逗號或怪字符，並加上驚嘆號表示語氣
                text_content = text_content.rstrip("，,、") 
                if not text_content.endswith(("！", "。", "!")):
                    text_content += "！"

            emotion = "激動" if "殺球" in chunk["info"] or "得分" in chunk["info"] else "平穩"

            # 去重合併邏輯
            if commentary and len(text_content) >= 2 and len(commentary[-1]["text"]) >= 2:
                check_len = min(5, len(text_content), len(commentary[-1]["text"]))
                if text_content[:check_len] == commentary[-1]["text"][:check_len]:
                    # 合併
                    commentary[-1]["end_time"] = seconds_to_timecode(chunk["end_sec"])
                    prev_start = parse_time_str(commentary[-1]["start_time"])
                    new_dur = chunk["end_sec"] - prev_start
                    commentary[-1]["time_range"] = format_duration(new_dur)
                    continue

            commentary.append({
                "start_time": seconds_to_timecode(chunk["start_sec"]),
                "end_time": seconds_to_timecode(chunk["end_sec"]),
                "time_range": format_duration(duration),
                "emotion": emotion,
                "text": text_content
            })

        # 4. 儲存
        if commentary:
            with open(os.path.join(output_folder, f"{base_name}.json"), "w", encoding="utf-8") as f:
                json.dump({"segment": f"{base_name}.mp4", "commentary": commentary}, f, ensure_ascii=False, indent=2)
            print(f"✅ {base_name} 完成！")
        else:
            print(f"⚠️ {base_name} 無適合旁白。")

if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    event_json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"

    process_stage2_narratives(video_folder, event_json_folder, output_folder)