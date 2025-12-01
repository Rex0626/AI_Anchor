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
SYLLABLES_PER_SEC = 4.5      
MIN_EVENT_DURATION = 1.0      
MAX_RALLY_DURATION = 4.5
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
1. 角色設定 (Role)
你是一位**資深、熱血且具備戰術洞察力**的頂級賽事主播。
你的目標是透過聲音將觀眾帶入賽場。你的解說風格：
- **拒絕平鋪直敘**：不要當「報幕員」，要當「說書人」。
- **強調因果關係**：解釋動作背後的意圖與結果（例如：不只是說「他殺球」，要說「這記殺球破壞了對手重心」）。
- **口語化 (TTS Friendly)**：使用適合朗讀的短句，避免生硬的書面用語。
- **注意細節**：請善用detail來豐富文本解說。

2. 上下文資訊 (Context)
- **歷史戰況 (Flow)**：
{{ prev_context }}
*(請繼承上述的語氣與情緒，確保解說流暢不斷層)*

- **輸入來源**：結合 **JSON 事件鏈** 與 **視覺畫面** 進行解說。

3. 任務執行 (Tasks)
你的工作是要將一系列的事件轉化為生動的解說文本：

- **特殊任務指引 (Special Tasks)**：
    - **[Intro]**: 影片剛開始。請簡單開場，介紹選手或當前比分局勢。
    - **[Gap]**: 比賽間隙。請描述球員的心理狀態、擦汗、換球、調整呼吸。
    - **[Outro]**: 影片結束。請快速總結剛剛這球的結果、得分者。
    - **[Replay]**: 精彩回放/慢動作。請**深入分析**剛才動作的技術細節（如：手腕變化、假動作、腳步移動），語氣專業且帶有讚嘆。

4. 嚴格禁令 (Strict Prohibitions)
⛔️ **違規將導致系統錯誤：**
- **🈲 禁止流水帳**：絕對不要使用「然後...接著...」這種連接詞。請用**因果關係**串聯（「逼得對手...」、「導致...」）。
- **🈲 禁止間隙幻覺**：在 `[Gap]` 絕對不能描述新的擊球動作（殺球/發球）。只能講評論。
- **🈲 禁止未卜先知**：若 `[Score]` 未出現，不可提前宣告得分。
- **🈲 嚴格字數控制**：`constraint` 是物理限制。**寧可話少精簡，絕不超時爆音。**

5. 輸出格式
輸出純 JSON 陣列，包含 `id` 和 `text` 兩個欄位。

6. 範例 (Example)
**輸入:**
[
    {"id": 0, "constraint": "限 14 音節", "content": "[Serve] 戴資穎 - 發球 (過高) -> [Offense] 陳雨菲 - 撲球 (下壓)"},
    {"id": 1, "constraint": "限 8 音節", "content": "[Score] 無 - 界內得分"},
    {"id": 2, "constraint": "限 12 音節", "content": "[Gap] 對手懊惱"}
]
**輸出:**
[
    {"id": 0, "text": "小戴這球發高了！陳雨菲沒放過機會直接下壓！"},
    {"id": 1, "text": "落地得分！這球抓得太準了！"},
    {"id": 2, "text": "小戴臉上露出了懊惱的表情。"}
]

📊 **待處理數據：**
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


# ========== 7. 核心功能：處理單一影片 (已修正時間軸排軸邏輯) ==========
def process_single_video_stage2(video_path, event_json_path, output_folder):
    global NARRATIVE_HISTORY

    os.makedirs(output_folder, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 0. 基礎資訊讀取
    try:
        with VideoFileClip(video_path) as clip: total_duration = clip.duration
    except: total_duration = 30.0 

    try:
        with open(event_json_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
            events = data.get("events", [])
            video_uri = data.get("video_uri", "") or data.get("segment_video_uri", "")
    except Exception as e:
        print(f"❌ 讀取 JSON 失敗: {e}")
        return None

    if not events: return None

    # ==========================================
    # Phase 1: 事件聚合 (Aggregation) - 強制切片版
    # ==========================================
    narrative_blocks = []
    current_block_events = []
    block_start_raw = 0.0
    
    events.sort(key=lambda x: parse_time_str(x.get("start_time", "0:00")))
    last_event_end = 0.0
    
    for i, event in enumerate(events):
        start = parse_time_str(event.get("start_time"))
        if start > total_duration - 0.5: continue

        end = parse_time_str(event.get("end_time"))
        if end == 0.0: end = start + 1.0
        
        cat = event.get("category", "General")
        sub = event.get("subject") or event.get("player", "球員")
        act = event.get("action", "")
        det = event.get("detail", "")
        event_str = f"[{cat}] {sub} - {act} ({det})"

        # --- 🔥 修改點：強制切片邏輯 ---
        current_block_duration = end - block_start_raw
        gap_from_last = start - last_event_end
        
        is_new_block = False
        if not current_block_events:
            is_new_block = True
        elif gap_from_last > 1.2:  # 間隔 > 1.2s 斷句
            is_new_block = True
        elif len(current_block_events) >= 3: # 🔥 動作數量 >= 3 強制換句
            is_new_block = True
        elif current_block_duration > 3.5:   # 🔥 時間長度 > 3.5s 強制換句
            is_new_block = True
            
        if is_new_block:
            if current_block_events:
                narrative_blocks.append({
                    "type": "RALLY",
                    "raw_start": block_start_raw,
                    "raw_end": last_event_end,
                    "content": " -> ".join(current_block_events)
                })
            current_block_events = [event_str]
            block_start_raw = start
        else:
            current_block_events.append(event_str)
        
        last_event_end = end

    if current_block_events:
        narrative_blocks.append({
            "type": "RALLY",
            "raw_start": block_start_raw,
            "raw_end": last_event_end,
            "content": " -> ".join(current_block_events)
        })

    # ==========================================
    # Phase 2: 預先排程 (Pre-Scheduling) - 膨脹與填空
    # ==========================================
    scheduled_tasks = [] 
    audio_cursor = 0.0 
    
    DELAY_MAP = {
        "setup": 2.0, "serve": 2.0, "offense": 0.6, "smash": 0.5,    
        "defense": 0.7, "score": 0.1, "gap": 0.0, "intro": 0.0, "outro": 0.0, "default": 0.8   
    }

    # --- 🔥 修改點 1: Intro 填空 ---
    first_block_start = narrative_blocks[0]["raw_start"] if narrative_blocks else total_duration
    if first_block_start > 3.0:
        intro_dur = min(first_block_start - 0.5, 6.0)
        scheduled_tasks.append({
            "id": "intro",
            "final_start": 0.0,
            "final_end": intro_dur,
            "duration": intro_dur,
            "type": "INTRO",
            "raw_content": "比賽開始",
            "prompt_constraint": f"限 {int(intro_dur * SYLLABLES_PER_SEC)} 音節",
            "prompt_content": "[Intro] 這是比賽開始，請做簡單開場介紹。"
        })
        audio_cursor = intro_dur

    # --- 迴圈排程 ---
    for idx, block in enumerate(narrative_blocks):
        # A. 理想時間
        delay = 0.8
        content_lower = block["content"].lower()
        for k, v in DELAY_MAP.items():
            if k in content_lower: delay = v; break
        ideal_start = block["raw_start"] + delay
        
        # --- 🔥 修改點 2: Gap 填空 ---
        gap_duration = ideal_start - audio_cursor
        if gap_duration > 4.0:
            fill_dur = min(gap_duration - 0.5, 5.0)
            gap_start = audio_cursor + 0.2
            scheduled_tasks.append({
                "id": f"gap_{idx}",
                "final_start": gap_start,
                "final_end": gap_start + fill_dur,
                "duration": fill_dur,
                "type": "GAP",
                "raw_content": "間隙",
                "prompt_constraint": f"限 {int(fill_dur * SYLLABLES_PER_SEC)} 音節",
                "prompt_content": "[Gap] 雙方調整節奏/球員心理/準備下一球"
            })
            audio_cursor = gap_start + fill_dur

        # C. 排程當前 Block
        start_time = max(ideal_start, audio_cursor + 0.2)
        
        # --- 🔥 修改點 3: 時間膨脹計算 ---
        raw_span = block["raw_end"] - block["raw_start"]
        base_min_duration = 3.5 # 保底 3.5 秒
        target_dur = min(raw_span + 2.0, 6.0) # 原始+2秒，上限6秒
        target_dur = max(target_dur, base_min_duration) # 應用保底

        # Lookahead: 只有遇到關鍵球才稍微讓路
        if idx < len(narrative_blocks) - 1:
            next_content = narrative_blocks[idx+1]["content"].lower()
            next_raw_start = narrative_blocks[idx+1]["raw_start"]
            if "score" in next_content or "smash" in next_content:
                next_ideal = next_raw_start + 0.5
                if next_ideal < start_time + target_dur:
                    compressed = next_ideal - start_time
                    target_dur = max(compressed, 2.5)

        end_time = start_time + target_dur
        if end_time > total_duration: end_time = total_duration
        
        final_duration = end_time - start_time
        if final_duration < 0.8: continue

        syllable_count = int(final_duration * SYLLABLES_PER_SEC)
        syllable_count = max(syllable_count, 5)

        scheduled_tasks.append({
            "id": idx,
            "final_start": start_time,
            "final_end": end_time,
            "duration": final_duration,
            "type": block["type"],
            "raw_content": block["content"],
            "prompt_constraint": f"限 {syllable_count} 音節",
            "prompt_content": block["content"]
        })
        audio_cursor = end_time

    # 目的：避免最後剩餘時間太長(如17秒)導致AI寫作文。將其拆解為「總結」+「回放分析」。
    
    remaining_time = total_duration - audio_cursor
    
    if remaining_time > 12.0:
        # 情況 A: 剩餘時間充裕 -> 拆分為 [Outro] + [Replay]
        
        # 1. 快速總結 (Outro) - 固定給 5 秒
        outro_dur = 5.0
        scheduled_tasks.append({
            "id": "outro_summary",
            "final_start": audio_cursor + 0.2,
            "final_end": audio_cursor + 0.2 + outro_dur,
            "duration": outro_dur,
            "type": "OUTRO",
            "raw_content": "結尾總結",
            "prompt_constraint": f"限 {int(outro_dur * SYLLABLES_PER_SEC)} 音節",
            "prompt_content": "[Outro] 本回合結束，快速總結得分關鍵。"
        })
        # 更新指針，為下一段做準備
        audio_cursor += (0.2 + outro_dur)

        # 2. 回放分析 (Replay) - 填補剩餘時間
        # 計算可用時間：剩餘時間 - 緩衝 1.0秒
        # 設定上限 8.0 秒 (避免講太久)
        replay_dur = min(remaining_time - outro_dur - 1.0, 8.0) 
        
        if replay_dur > 3.0:
            scheduled_tasks.append({
                "id": "outro_replay",
                "final_start": audio_cursor + 0.5,
                "final_end": audio_cursor + 0.5 + replay_dur,
                "duration": replay_dur,
                "type": "REPLAY",
                "raw_content": "慢動作分析",
                "prompt_constraint": f"限 {int(replay_dur * SYLLABLES_PER_SEC)} 音節",
                "prompt_content": "[Replay] 這是精彩重播畫面，請深入分析剛才動作的技術細節(如假動作或落點)。"
            })

    elif remaining_time > 3.0:
        # 情況 B: 剩餘時間正常 -> 只有 [Outro]
        # 設定上限 6.0 秒
        outro_dur = min(remaining_time - 0.5, 6.0)
        scheduled_tasks.append({
            "id": "outro",
            "final_start": audio_cursor + 0.2,
            "final_end": audio_cursor + 0.2 + outro_dur,
            "duration": outro_dur,
            "type": "OUTRO",
            "raw_content": "結尾",
            "prompt_constraint": f"限 {int(outro_dur * SYLLABLES_PER_SEC)} 音節",
            "prompt_content": "[Outro] 本回合結束，總結剛才的精彩表現。"
        })

    if not scheduled_tasks:
        print(f"⚠️ [Skip] {base_name}: 無有效任務")
        return None

    # ==========================================
    # Phase 3: 生成解說 (Generation) - 帶 Context 版
    # ==========================================
    
    # 1. 準備任務數據
    llm_input_data = []
    for task in scheduled_tasks:
        llm_input_data.append({
            "id": task["id"],
            "constraint": task["prompt_constraint"], 
            "content": task["prompt_content"]
        })
        
    # 2. 準備歷史紀錄 (Context)
    if NARRATIVE_HISTORY:
        recent_history = NARRATIVE_HISTORY[-HISTORY_WINDOW_SIZE:]
        history_str = "\n".join([f"- {h}" for h in recent_history])
    else:
        history_str = "這是比賽的第一個片段，請直接開始解說。"

    try:
        # 3. 執行 Pipeline (🔥 關鍵：必須傳入 prev_context)
        res = pipeline_s2.run({
            "add_video": {"uri": video_uri},
            "prompt_builder": {
                "event_data": json.dumps(llm_input_data, ensure_ascii=False, indent=2),
                "prev_context": history_str  # <--- 這行就是解決 Missing input 的關鍵
            }
        })

        reply = res["llm"]["replies"][0].strip()
        
        # 清洗 JSON
        if "```" in reply:
            match = re.search(r'\[.*\]', reply, re.DOTALL)
            if match: reply = match.group()
        
        generated_list = json.loads(reply)
        generated_map = {str(item["id"]): item["text"] for item in generated_list}
        
    except Exception as e:
        print(f"❌ [Stage 2 LLM 錯誤] {e}")
        return None

    # ==========================================
    # Phase 4: 輸出組裝 (Assembly)
    # ==========================================
    
    commentary = []
    segment_texts = []
    
    for task in scheduled_tasks:
        tid = str(task["id"]) # 轉字串
        text = generated_map.get(tid, "")
        if not text: continue
        
        # 🔥 修改：增強版情緒判斷
        emotion = "平穩" 
        content_lower = task["raw_content"].lower()
        task_type = task["type"]
        
        if task_type == "INTRO":
            emotion = "舒緩"
        elif task_type == "OUTRO":
            emotion = "激動" 
        elif task_type == "REPLAY":  # 🔥 新增這行
            emotion = "專業"       # 回放分析時使用專業/分析語氣
        elif task_type == "GAP":
            emotion = "舒緩"
        elif any(k in content_lower for k in ["score", "smash", "kill", "won", "winner"]):
            emotion = "激動"
        elif any(k in content_lower for k in ["defense", "save", "foul", "out", "mistake"]):
            emotion = "緊張"
        elif any(k in content_lower for k in ["serve", "prepare"]):
            emotion = "舒緩"
        elif any(k in content_lower for k in ["miss", "error", "fail"]):
            emotion = "遺憾"

        commentary.append({
            "start_time": seconds_to_timecode(task["final_start"]),
            "end_time": seconds_to_timecode(task["final_end"]),
            "time_range": format_duration(task["duration"]),
            "emotion": emotion,
            "text": text
        })
        segment_texts.append(text)

    # (後續存檔部分保持不變)
    if segment_texts:
        NARRATIVE_HISTORY.append(" ".join(segment_texts))
        if len(NARRATIVE_HISTORY) > 10: NARRATIVE_HISTORY.pop(0)

    output_path = os.path.join(output_folder, f"{base_name}.json")
    if commentary:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"segment": base_name, "commentary": commentary}, f, ensure_ascii=False, indent=2)
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