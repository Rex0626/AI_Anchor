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
SYLLABLES_PER_SEC = 4.0      
MIN_EVENT_DURATION = 1.0      
MAX_RALLY_DURATION = 4.5
MIN_GAP_DURATION = 3.0        
MAX_INTRO_OUTRO_SYLLABLES = 30 
MERGE_THRESHOLD = 1.2 

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
- **強調因果關係**：解釋動作背後的意圖與結果。
- **口語化 (TTS Friendly)**：使用適合朗讀的短句，避免生硬的書面用語。
- **注意細節**：請善用detail來豐富文本解說。

2. 上下文資訊 (Context)
- **比賽背景 (Static Info)**：
{{ intro }}
*(⚠️ 這是本場比賽的核心資訊，請務必根據此設定來稱呼球員與辨識身分)*

- **歷史戰況 (Flow)**：
{{ prev_context }}
*(請繼承上述的語氣與情緒)*

- **輸入來源**：結合 **JSON 事件鏈** 與 **視覺畫面** 進行解說。

3. 任務執行 (Tasks)
你的工作是要將一系列的事件轉化為生動的解說文本：

- **特殊任務指引 (Special Tasks)**：
    - **[Summary]**: 當看到此標記（如 `[Summary] 雙方連續 6 拍平抽`），**請勿逐一描述動作**。請直接用一句話總結戰況，例如：「雙方在網前展開了令人窒息的快速對攻！」
    - **[Intro]**: 影片剛開始。請簡單開場，介紹選手或當前比分局勢。
    - **[Gap]**: 比賽間隙。請描述球員的心理狀態、擦汗、換球、調整呼吸。
    - **[Outro]**: 影片結束。請快速總結剛剛這球的結果、得分者。
    - **[Replay]**: 精彩回放/慢動作。請**深入分析**剛才動作的技術細節，語氣專業且帶有讚嘆。

4. 嚴格禁令 (Strict Prohibitions)
⛔️ **違規將導致系統錯誤：**
- **🈲 禁止流水帳**：絕對不要使用「然後...接著...」。請用**因果關係**串聯。
- **🈲 禁止間隙幻覺**：在 `[Gap]` 絕對不能描述新的擊球動作。只能講評論。
- **🈲 禁止未卜先知**：若 `[Score]` 未出現，不可提前宣告得分。
- **🈲 嚴格字數控制**：`constraint` 是物理限制。**寧可話少精簡，絕不超時爆音。**

5. 輸出格式
輸出純 JSON 陣列，包含 `id` 和 `text` 兩個欄位。

6. 範例 (Example)
**輸入:**
[
    {"id": 0, "constraint": "限 14 音節", "content": "[Serve] 戴資穎 - 發球 (過高) -> [Offense] 陳雨菲 - 撲球 (下壓)"},
    {"id": 1, "constraint": "限 8 音節", "content": "[Score] 無 - 界內得分"}
]
**輸出:**
[
    {"id": 0, "text": "小戴這球發高了！陳雨菲沒放過機會直接下壓！"},
    {"id": 1, "text": "落地得分！這球抓得太準了！"}
]

📊 **待處理數據：**
{{ event_data }}

請輸出 JSON：
"""

# 🔥 [修正] 必須包含 "intro"，否則會報 Input not found 錯誤
prompt_builder = PromptBuilder(template=narrative_template, required_variables=["event_data", "prev_context", "intro"])

add_video_s2 = AddVideo2Prompt()
gemini_s2 = GeminiGenerator(project_id="ai-anchor-462506", location="us-central1", model="gemini-2.5-flash")

pipeline_s2 = Pipeline()
pipeline_s2.add_component(instance=prompt_builder, name="prompt_builder")
pipeline_s2.add_component(instance=add_video_s2, name="add_video")
pipeline_s2.add_component(instance=gemini_s2, name="llm")
pipeline_s2.connect("prompt_builder.prompt", "add_video.prompt")
pipeline_s2.connect("add_video.prompt", "llm.prompt")


# ========== 7. 核心功能：處理單一影片 (最終完整版) ==========
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
            # 🔥 讀取 intro，解決身分失憶問題
            current_intro = data.get("intro", "這是一場精彩的羽球比賽，請根據畫面解說。")
    except Exception as e:
        print(f"❌ 讀取 JSON 失敗: {e}")
        return None

    if not events: return None

    # ==========================================
    # Phase 1: 事件聚合 (語意感知 + 摘要增強版)
    # ==========================================
    narrative_blocks = []
    current_block_events = []
    current_block_cats = [] 
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

        # --- 切分邏輯 ---
        should_split = False
        gap_from_last = start - last_event_end
        current_dur = end - block_start_raw
        
        # 硬性條件
        if not current_block_events:
            should_split = False
        elif cat == "Serve" or cat == "Start":
            should_split = True # 發球必斷
        elif "Score" in current_block_cats: 
            should_split = True # 得分後必斷
        elif gap_from_last > 2.0:
            should_split = True # 間隔過長必斷
        
        # 軟性條件
        elif not should_split:
            is_combo = (current_block_cats and current_block_cats[-1] == "Offense" and cat == "Defense")
            if not is_combo and current_dur > 4.5:
                should_split = True
            elif len(current_block_events) >= 5:
                should_split = True
        
        if should_split:
            # 🔥 [摘要機制] 檢查是否需要 Summary
            final_content = " -> ".join(current_block_events)
            
            exch_count = current_block_cats.count("Exchange")
            drive_count = sum(1 for s in current_block_events if "平抽" in s or "擋" in s)
            total_count = len(current_block_events)
            
            if total_count >= 3 and (exch_count + drive_count) >= (total_count * 0.7):
                final_content = f"[Summary] 雙方進行了 {total_count} 拍的快速平抽擋/來回對峙"
            
            narrative_blocks.append({
                "type": "RALLY",
                "raw_start": block_start_raw,
                "raw_end": last_event_end,
                "content": final_content
            })
            
            current_block_events = [event_str]
            current_block_cats = [cat]
            block_start_raw = start
        else:
            current_block_events.append(event_str)
            current_block_cats.append(cat)
            if len(current_block_events) == 1: block_start_raw = start
        
        last_event_end = max(last_event_end, end)

    # 處理殘留 Block
    if current_block_events:
        final_content = " -> ".join(current_block_events)
        exch_count = current_block_cats.count("Exchange")
        if len(current_block_events) >= 3 and exch_count >= len(current_block_events)*0.7:
             final_content = f"[Summary] 雙方進行了連續的來回對峙"

        narrative_blocks.append({
            "type": "RALLY",
            "raw_start": block_start_raw,
            "raw_end": last_event_end,
            "content": final_content
        })

    # ==========================================
    # Phase 2: 預先排程 (含反應延遲 + 智慧音節)
    # ==========================================
    scheduled_tasks = [] 
    audio_cursor = 0.0 
    
    # 🔥 反應延遲設定
    DELAY_MAP = {
        "setup": 2.0, "serve": 2.0, "offense": 0.6, "smash": 0.5,    
        "defense": 0.7, "score": 0.1, "gap": 0.0, "intro": 0.0, "outro": 0.0, "default": 0.8   
    }

    # 1. Intro 填空
    first_block_start = narrative_blocks[0]["raw_start"] if narrative_blocks else total_duration
    if first_block_start > 3.0:
        intro_dur = min(first_block_start - 0.5, 6.0)
        intro_limit = max(int(intro_dur * SYLLABLES_PER_SEC), 10)
        
        scheduled_tasks.append({
            "id": "intro",
            "final_start": 0.0,
            "final_end": intro_dur,
            "duration": intro_dur,
            "type": "INTRO",
            "raw_content": "開場",
            "prompt_constraint": f"限 {intro_limit} 音節",
            "prompt_content": "[Intro] 這是比賽開始，請做簡單開場介紹。"
        })
        audio_cursor = intro_dur

    # 2. 迴圈排程
    for idx, block in enumerate(narrative_blocks):
        delay = 0.8
        content_lower = block["content"].lower()
        for k, v in DELAY_MAP.items():
            if k in content_lower: delay = v; break
        ideal_start = block["raw_start"] + delay
        
        # Gap 填空
        gap_duration = ideal_start - audio_cursor
        if gap_duration > 4.0:
            fill_dur = min(gap_duration - 0.5, 5.0)
            gap_start = audio_cursor + 0.2
            gap_limit = max(int(fill_dur * SYLLABLES_PER_SEC), 8)

            scheduled_tasks.append({
                "id": f"gap_{idx}",
                "final_start": gap_start,
                "final_end": gap_start + fill_dur,
                "duration": fill_dur,
                "type": "GAP",
                "raw_content": "間隙",
                "prompt_constraint": f"限 {gap_limit} 音節",
                "prompt_content": "[Gap] 填補空白，描述球員狀態或心理。"
            })
            audio_cursor = gap_start + fill_dur

        # 排程當前 Block
        start_time = max(ideal_start, audio_cursor + 0.2)
        
        raw_span = block["raw_end"] - block["raw_start"]
        base_min_duration = 3.5 
        target_dur = min(raw_span + 2.0, 6.0) 
        target_dur = max(target_dur, base_min_duration) 

        # Lookahead
        if idx < len(narrative_blocks) - 1:
            next_raw_start = narrative_blocks[idx+1]["raw_start"]
            deadline = next_raw_start + 1.0
            max_allowed_dur = max(1.5, deadline - start_time)
            target_dur = min(target_dur, max_allowed_dur)

        end_time = start_time + target_dur
        if end_time > total_duration: end_time = total_duration
        
        final_duration = end_time - start_time
        if final_duration < 0.8: continue

        # 🔥 智慧音節計算
        syllable_count = int(final_duration * SYLLABLES_PER_SEC)
        
        is_crucial = any(k in content_lower for k in ["score", "smash", "kill", "won"])
        is_summary = "[summary]" in content_lower
        
        if is_crucial or is_summary:
            syllable_count = max(syllable_count, 12) 
        else:
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

    # 3. Outro/Replay 處理
    remaining_time = total_duration - audio_cursor
    if remaining_time > 12.0:
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
        audio_cursor += (0.2 + outro_dur)
        replay_dur = min(remaining_time - outro_dur - 1.0, 8.0) 
        if replay_dur > 3.0:
            scheduled_tasks.append({
                "id": "outro_replay",
                "final_start": audio_cursor + 0.5,
                "final_end": audio_cursor + 0.5 + replay_dur,
                "duration": replay_dur,
                "type": "REPLAY",
                "raw_content": "慢動作分析",
                "prompt_constraint": f"限 {max(int(replay_dur * SYLLABLES_PER_SEC), 10)} 音節",
                "prompt_content": "[Replay] 這是精彩重播畫面，請深入分析技術細節。"
            })
    elif remaining_time > 3.0:
        outro_dur = min(remaining_time - 0.5, 6.0)
        scheduled_tasks.append({
            "id": "outro",
            "final_start": audio_cursor + 0.2,
            "final_end": audio_cursor + 0.2 + outro_dur,
            "duration": outro_dur,
            "type": "OUTRO",
            "raw_content": "結尾",
            "prompt_constraint": f"限 {max(int(outro_dur * SYLLABLES_PER_SEC), 8)} 音節",
            "prompt_content": "[Outro] 本回合結束，總結剛才的精彩表現。"
        })

    if not scheduled_tasks: return None

    # ==========================================
    # Phase 3: 生成解說 (Generation) - 帶 Context & Intro
    # ==========================================
    llm_input_data = []
    for task in scheduled_tasks:
        llm_input_data.append({
            "id": task["id"],
            "constraint": task["prompt_constraint"], 
            "content": task["prompt_content"]
        })
        
    if NARRATIVE_HISTORY:
        recent_history = NARRATIVE_HISTORY[-HISTORY_WINDOW_SIZE:]
        history_str = "\n".join([f"- {h}" for h in recent_history])
    else:
        history_str = "這是比賽的第一個片段，請直接開始解說。"

    try:
        # 🔥 傳入 intro 到 Pipeline
        res = pipeline_s2.run({
            "add_video": {"uri": video_uri},
            "prompt_builder": {
                "event_data": json.dumps(llm_input_data, ensure_ascii=False, indent=2),
                "prev_context": history_str,
                "intro": current_intro 
            }
        })
        reply = res["llm"]["replies"][0].strip()
        if "```" in reply:
            match = re.search(r'\[.*\]', reply, re.DOTALL)
            if match: reply = match.group()
        generated_list = json.loads(reply)
        generated_map = {str(item["id"]): item["text"] for item in generated_list}
    except Exception as e:
        print(f"❌ [Stage 2 LLM 錯誤] {e}")
        return None

    # ==========================================
    # Phase 4: 輸出組裝 (Assembly) - 嚴格排軸 + 雙重檢查
    # ==========================================
    commentary = []
    segment_texts = []
    num_tasks = len(scheduled_tasks)

    for i, task in enumerate(scheduled_tasks):
        tid = str(task["id"])
        text = generated_map.get(tid, "")
        if not text: continue
        
        final_start = task["final_start"]
        
        # 硬性截止
        if i < num_tasks - 1:
            next_start = scheduled_tasks[i+1]["final_start"]
            hard_limit_end = next_start - 0.2 
        else:
            hard_limit_end = total_duration
            
        # 🔥 雙重檢查
        estimated_dur = estimate_speech_time(text)
        calculated_end = final_start + estimated_dur
        final_end = min(calculated_end, hard_limit_end)
        
        if final_end <= final_start: final_end = final_start + 0.5 
        final_dur = final_end - final_start

        if final_dur > 0.1:
            speed_val = estimated_dur / final_dur
        else:
            speed_val = 1.0

        # 限制範圍：最慢 1.0倍，最快 2.0倍
        speed_val = round(max(1.0,min(speed_val,2.0)),2)

        # 情緒判斷
        emotion = "平穩" 
        content_lower = task["raw_content"].lower()
        task_type = task["type"]
        
        if task_type == "INTRO": emotion = "舒緩"
        elif task_type == "OUTRO": emotion = "激動" 
        elif task_type == "REPLAY": emotion = "專業"       
        elif task_type == "GAP": emotion = "舒緩"
        elif any(k in content_lower for k in ["score", "smash", "kill", "won", "winner"]): emotion = "激動"
        elif any(k in content_lower for k in ["defense", "save", "foul", "out", "mistake"]): emotion = "緊張"
        elif any(k in content_lower for k in ["serve", "prepare"]): emotion = "舒緩"
        elif any(k in content_lower for k in ["miss", "error", "fail"]): emotion = "遺憾"

        commentary.append({
            "start_time": seconds_to_timecode(final_start),
            "end_time": seconds_to_timecode(final_end),
            "time_range": format_duration(final_dur),
            "speed": speed_val,
            "emotion": emotion,
            "text": text
        })
        segment_texts.append(text)

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
    
    print(f"\n🚀 [獨立模式] Stage 2 (最終完美版) 批次啟動...")
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