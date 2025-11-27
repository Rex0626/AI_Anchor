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
SYLLABLES_PER_SEC = 4.8       
MIN_EVENT_DURATION = 1.0      
MAX_RALLY_DURATION = 6.0
MIN_GAP_DURATION = 3.0        
MAX_INTRO_OUTRO_SYLLABLES = 30 
MERGE_THRESHOLD = 1.2 # [新增] 強制合併閾值：若片段短於 1.2 秒，強制合併到下一段

# 全域歷史紀錄
NARRATIVE_HISTORY = [] 
HISTORY_WINDOW_SIZE = 3 

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

# ========== 4. Pipeline 初始化 ==========
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

# ========== 5. Prompt 模板 ==========
narrative_template = """ 
1. 角色 (Role)
你是一位**資深、熱血且節奏明快**的羽球賽事即時主播。
你的聲音充滿激情，能精準捕捉賽場上的每一個精彩瞬間。

2. 前情提要 (Context)
- **歷史戰況回顧**：
{{ prev_context }}
*(請參考上述歷史紀錄，掌握比賽氣勢流向)*

- **雙模態資訊**：請結合 **JSON 數據 (骨架)** 與 **影片畫面 (血肉)** 進行解說。

3. 應該要做的事 (Tasks)
- **區分場景與節奏 (Pacing)**：
    - **🟢 INTRO**: 暖場，帶入氣氛。
    - **🟡 RALLY (激動)**: 語速快！緊跟球路。若有連續攻防，請用流暢語句串聯。
    - **🔵 GAP (舒緩)**: 當內容標註為「中場間隙」時，請放慢語速。填補內容僅限：**評論上一球得失、描述球員狀態、或分析心理**。
    - **🔴 OUTRO**: 總結本段落結果。
- **人名重述**：務必帶上球員名字，特別是在攻防轉換時。
- **視覺細節**：描述殺球的「聲音」、救球的「狼狽」、慶祝的「動作」。

4. 禁止做的事 (Strict Prohibitions)
⛔️ **嚴格禁令 (違者導致播報事故)：**
- **🈲 間隙幻覺 (No Action in Gap)**：在 `GAP` 時段，**絕對禁止**描述任何擊球動作（如發球、殺球）。這是死球時間，只能講靜態內容。
- **🈲 禁止腦補結果**：若輸入內容提到「畫面中斷」或「球未落地」，**絕對不可**宣告得分或界外。
- **🈲 禁止遺漏 (No Skipping)**：輸入列表中的每一個 ID 都必須對應一句解說，不可跳過任何一個動作區塊。
- **🈲 絕對不可超時**：嚴格遵守 `constraint` 音節限制。

5. JSON 欄位定義
輸出純 JSON 陣列，包含 `id` 和 `text`。

6. JSON 輸出範例
**輸入:**
[
    {"id": 0, "constraint": "限 15 音節", "content": "Sakuramoto殺球 -> Tan擋網"},
    {"id": 1, "constraint": "限 10 音節", "content": "中場間隙 (Gap)"},
    {"id": 2, "constraint": "限 8 音節", "content": "殺球 -> 畫面中斷"}
]
**輸出:**
[
    {"id": 0, "text": "Sakuramoto起跳重殺，但Tan防守得非常穩健！"},
    {"id": 1, "text": "這球雙方節奏都很快，稍微喘口氣。"},
    {"id": 2, "text": "這球殺得非常兇！"}
]

📊 **本段待處理列表：**
{{ event_data }}

請輸出 JSON：
"""

prompt_builder = PromptBuilder(template=narrative_template, required_variables=["event_data","prev_context"])

add_video_s2 = AddVideo2Prompt()
gemini_s2 = GeminiGenerator(project_id="ai-anchor-462506", location="us-central1", model="gemini-2.5-flash")

pipeline_s2 = Pipeline()
pipeline_s2.add_component(instance=prompt_builder, name="prompt_builder")
pipeline_s2.add_component(instance=add_video_s2, name="add_video")
pipeline_s2.add_component(instance=gemini_s2, name="llm")
pipeline_s2.connect("prompt_builder.prompt", "add_video.prompt")
pipeline_s2.connect("add_video.prompt", "llm.prompt")

# ========== 6. 輔助函式 (含最終防線) ==========
def _flush_chunk(results_list, chunk_data, global_counter_ref):
    dur = chunk_data["end"] - chunk_data["start"]
    limit = int(dur * SYLLABLES_PER_SEC)
    should_keep = False
    
    # 判斷類型
    is_gap = "間隙" in chunk_data.get("info", "") or "Gap" in chunk_data.get("info", "")
    is_crucial = chunk_data.get("is_crucial", False)

    # 🔥 最終防線：若通過了迴圈的篩選，這裡做最後的格式保障
    if is_crucial or is_gap:
        # 關鍵時刻/間隙：強制保留並給予最小字數空間
        limit = max(limit, 6) 
        should_keep = True
    else:
        # 普通 Rally：若還是太短且沒被合併，丟棄 (這是最後一道濾網)
        if limit >= 4:
            should_keep = True

    if should_keep:
        results_list.append({
            "global_id": global_counter_ref[0], 
            "start_sec": chunk_data["start"], 
            "end_sec": chunk_data["end"], 
            "limit": limit, 
            "info": chunk_data["info"]
        })
        global_counter_ref[0] += 1


# ========== 7. 核心功能：處理單一影片 ==========
def process_single_video_stage2(video_path, event_json_path, output_folder):
    global NARRATIVE_HISTORY

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    try:
        with VideoFileClip(video_path) as clip: total_duration = clip.duration
    except: total_duration = 30.0 

    try:
        with open(event_json_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            nested_events = data.get("events", []) if isinstance(data, dict) else data
            video_uri = data.get("segment_video_uri", "") if isinstance(data, dict) else ""
    except: return None

    if not nested_events: return None

    # --- A. 數據聚合邏輯 ---
    RALLY_TYPES = ["Exchange", "Attack", "Defend"] 
    chunk_events = [] 
    global_id_counter = [0] 
    last_committed_time = 0.0

    # 1. INTRO
    first_chunk_start = parse_time_str(nested_events[0].get("start_time", "0:00.0"))
    intro_limit = int(first_chunk_start * SYLLABLES_PER_SEC)
    if intro_limit >= 8 and intro_limit <= MAX_INTRO_OUTRO_SYLLABLES:
        chunk_events.append({
            "global_id": "INTRO",
            "start_sec": 0.0,
            "end_sec": first_chunk_start,
            "limit": intro_limit,
            "info": "開場空白"
        })
        last_committed_time = first_chunk_start
    else:
        last_committed_time = 0.0
    
    # 2. 聚合迴圈 (含向後合併邏輯)
    buffer_chunk = None

    for chunk in nested_events:
        chunk_start = parse_time_str(chunk.get("start_time", "0:00.0"))
        chunk_end = parse_time_str(chunk.get("end_time", "0:00.0"))
        inner_list = chunk.get("events", [])
        
        if not inner_list: continue

        actions_str = " -> ".join([f"{ev.get('player')}{ev.get('action')}" for ev in inner_list])
        is_crucial = any(ev.get('is_crucial') is True for ev in inner_list)
        is_pure_rally = all(ev.get('category') in RALLY_TYPES for ev in inner_list) and not is_crucial

        current_chunk = {
            "start": chunk_start, 
            "end": chunk_end, 
            "info": actions_str,
            "is_rally": is_pure_rally,
            "is_crucial": is_crucial 
        }

        # --- Gap Detection ---
        prev_end_candidate = buffer_chunk["end"] if buffer_chunk else last_committed_time
        gap_duration = chunk_start - prev_end_candidate
        
        if gap_duration > MIN_GAP_DURATION:
            # 發現大間隙
            if buffer_chunk:
                # 檢查 buffer 是否太短？若是，直接被 Gap 吞噬 (刪除 buffer)
                # 這避免產生 "Action (0.2s) -> Gap" 的怪異結構
                buf_dur = buffer_chunk["end"] - buffer_chunk["start"]
                if buf_dur < MERGE_THRESHOLD:
                    # 吞噬：Gap 起點提前到 buffer 起點
                    prev_end_candidate = buffer_chunk["start"]
                    buffer_chunk = None # 丟棄 buffer
                else:
                    # 正常結算
                    _flush_chunk(chunk_events, buffer_chunk, global_id_counter)
                    last_committed_time = buffer_chunk["end"]
                    buffer_chunk = None
                    prev_end_candidate = last_committed_time
            
            # 插入間隙事件
            gap_chunk = {
                "start": prev_end_candidate,
                "end": chunk_start,
                "info": "中場間隙 (Gap)",
                "is_crucial": False
            }
            _flush_chunk(chunk_events, gap_chunk, global_id_counter)
            last_committed_time = chunk_start
            buffer_chunk = current_chunk

        else:
            # --- 正常合併邏輯 ---
            if buffer_chunk:
                potential_dur = current_chunk["end"] - buffer_chunk["start"]
                is_mergeable = (
                    buffer_chunk["is_rally"] and 
                    current_chunk["is_rally"] and 
                    potential_dur <= MAX_RALLY_DURATION
                )
                
                if is_mergeable:
                    # 標準合併：向後延伸
                    buffer_chunk["end"] = current_chunk["end"] 
                    buffer_chunk["info"] += f" -> {current_chunk['info']}"
                else:
                    # 衝突：無法標準合併
                    # 🔥 [新增] 強制向後合併檢查 (Force Merge Forward)
                    # 如果 buffer 實在太短 (例如 0.2s)，為了不浪費，強制塞給 current
                    buf_dur = buffer_chunk["end"] - buffer_chunk["start"]
                    
                    if buf_dur < MERGE_THRESHOLD:
                        # 執行向後合併：Current 吸收 Buffer
                        current_chunk["start"] = buffer_chunk["start"] # 時間前推
                        current_chunk["info"] = f"{buffer_chunk['info']} -> {current_chunk['info']}" # 內容前置
                        
                        # 屬性繼承：若 buffer 是關鍵，合併後也視為關鍵 (避免漏報)
                        if buffer_chunk["is_crucial"]:
                            current_chunk["is_crucial"] = True
                        
                        # Buffer 被吸收，現在 Current 變成新的 Buffer
                        buffer_chunk = current_chunk
                    else:
                        # Buffer 夠長，可以獨立生存
                        _flush_chunk(chunk_events, buffer_chunk, global_id_counter)
                        last_committed_time = buffer_chunk["end"]
                        buffer_chunk = current_chunk
            else:
                buffer_chunk = current_chunk
        
    # 結算最後的 buffer
    if buffer_chunk:
        # 最後一段無法向後合併，只能依靠 _flush_chunk 的 padding 保護
        _flush_chunk(chunk_events, buffer_chunk, global_id_counter)
        last_committed_time = buffer_chunk["end"]

    # 3. OUTRO
    outro_dur = total_duration - last_committed_time
    outro_limit = int(outro_dur * SYLLABLES_PER_SEC)
    
    if outro_limit >= 8 and outro_limit <= MAX_INTRO_OUTRO_SYLLABLES:
        chunk_events.append({
            "global_id": "OUTRO", 
            "start_sec": last_committed_time, 
            "end_sec": total_duration, 
            "limit": outro_limit, 
            "info": "結尾空白"
        })

    # --- B. 呼叫 LLM ---
    if NARRATIVE_HISTORY:
        recent_history = NARRATIVE_HISTORY[-HISTORY_WINDOW_SIZE:]
        history_str = "\n".join([f"- {h}" for h in recent_history])
    else:
        history_str = "這是比賽的第一個片段，請開始精彩的解說。"

    llm_input_data = []
    for e in chunk_events:
        llm_input_data.append({"id": e["global_id"], "constraint": f"限 {e['limit']} 音節", "content": e["info"]})
    
    try:
        res = pipeline_s2.run({
            "add_video": {"uri": video_uri},
            "prompt_builder": {
                "event_data": json.dumps(llm_input_data, ensure_ascii=False, indent=2),
                "prev_context": history_str 
                }
        })
        reply = res["llm"]["replies"][0].strip()
        if reply.startswith("```"): reply = reply.split("\n", 1)[1].rsplit("\n", 1)[0]
        generated_list = json.loads(reply)
        generated_map = {str(item["id"]): item["text"] for item in generated_list}
    except Exception as e:
        print(f"❌ [Stage 2 錯誤] {e}")
        return None

    # --- C. 輸出結果 ---
    commentary = []
    current_segment_narrative = [] 
    
    for chunk in chunk_events:
        gid = str(chunk["global_id"])
        text_content = generated_map.get(gid)
        if not text_content: continue 

        duration = chunk["end_sec"] - chunk["start_sec"]
        
        validation_duration = duration
        if gid in ["INTRO", "OUTRO"]: validation_duration = min(duration, 5.0)
        
        estimated_dur = estimate_speech_time(text_content)
        if estimated_dur > (validation_duration * 1.2):
            ratio = (validation_duration * 1.2) / estimated_dur
            safe_length = int(len(text_content) * ratio)
            text_content = text_content[:safe_length].rstrip("，,")

        chunk_info_lower = chunk["info"].lower()
        if "gap" in chunk_info_lower:
            emotion = "平穩"
        else:
            emotion = "激動" if "殺球" in chunk_info_lower or "得分" in chunk_info_lower or "attack" in chunk_info_lower else "平穩"

        if commentary and len(text_content) >= 2 and len(commentary[-1]["text"]) >= 2:
            check_len = min(5, len(text_content), len(commentary[-1]["text"]))
            if text_content[:check_len] == commentary[-1]["text"][:check_len]:
                commentary[-1]["end_time"] = seconds_to_timecode(chunk["end_sec"])
                prev_start = parse_time_str(commentary[-1]["start_time"])
                new_dur = chunk["end_sec"] - prev_start
                commentary[-1]["time_range"] = format_duration(new_dur)
                continue
        
        if text_content:
            current_segment_narrative.append(text_content)

        commentary.append({
            "start_time": seconds_to_timecode(chunk["start_sec"]),
            "end_time": seconds_to_timecode(chunk["end_sec"]),
            "time_range": format_duration(duration),
            "emotion": emotion,
            "text": text_content
        })

    if current_segment_narrative:
        full_segment_text = " ".join(current_segment_narrative)
        NARRATIVE_HISTORY.append(full_segment_text)
        if len(NARRATIVE_HISTORY) > 20: 
            NARRATIVE_HISTORY.pop(0)

    output_filename = f"{base_name}.json"
    output_path = os.path.join(output_folder, output_filename)
    if commentary:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"segment": os.path.basename(video_path), "commentary": commentary}, f, ensure_ascii=False, indent=2)
        return output_path
    else:
        return None

# ========== 8. 獨立運行模式 ==========
if __name__ == "__main__":
    video_folder = "D:/Vs.code/AI_Anchor/backend/video_splitter/badminton_segments"
    event_json_folder = "D:/Vs.code/AI_Anchor/backend/gemini/event_analysis_output"
    output_folder = "D:/Vs.code/AI_Anchor/backend/gemini/final_narratives"
    
    NARRATIVE_HISTORY = [] 
    
    print(f"\n🚀 [獨立模式] Stage 2 (向後合併增強版) 批次啟動...")
    if os.path.exists(event_json_folder):
        files = sorted([f for f in os.listdir(event_json_folder) if f.endswith("_event.json")])
        for f in tqdm(files, desc="Processing"):
            base = f.replace("_event.json", "")
            vid_path = os.path.join(video_folder, f"{base}.mp4")
            json_path = os.path.join(event_json_folder, f)
            if os.path.exists(vid_path):
                res = process_single_video_stage2(vid_path, json_path, output_folder)
                if res: print(f"  -> Saved: {os.path.basename(res)}")
    else:
        print("❌ 找不到 JSON 資料夾")