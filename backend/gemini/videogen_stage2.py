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

- **解讀規則**：[分類] player - action (detail)
    *範例：`[Offense] 戴資穎 - 殺球 (貼網)`*

- **語氣與節奏指引 (Tone & Pacing)**：
    - **🟢 [Setup] / [Exchange] (戰術分析)**：
        * **語氣**：冷靜、清晰。
        * **重點**：描述球路佈局。例如：「雙方還在互相試探網前手感...」
    - **🟡 [Offense] / [Defense] (攻防張力)**：
        * **語氣**：**急促、緊湊！**
        * **重點**：使用「動作-反應」邏輯。例如：「小戴突然起跳重殺！雨菲反應很快直接擋回！」
    - **🔴 [Score] / [Result] (情緒釋放)**：
        * **語氣**：**高昂、激動！**
        * **重點**：讚嘆得分手段或惋惜失誤。例如：「哇！這球殺得太刁鑽了！完全沒機會！」
    - **🔵 [Gap] / [Intro] / [Outro] (呼吸留白)**：
        * **語氣**：舒緩、感性。
        * **重點**：填補空白，但不要填滿。評論上一球的心理博弈，或預告下一球。

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

    # --- A. 智慧聚合邏輯 (Smart Aggregation) ---
    narrative_blocks = []
    current_block = []
    block_start_time = 0.0
    
    events.sort(key=lambda x: parse_time_str(x.get("start_time", "0:00")))

    # 1. 處理開場
    first_event_start = parse_time_str(events[0].get("start_time", "0:00"))
    if first_event_start > 1.5:
        narrative_blocks.append({
            "type": "INTRO",
            "start": 0.0,
            "end": first_event_start,
            "content": "開場/準備動作"
        })

    # 2. 遍歷事件並分組
    last_event_end = 0.0
    
    for i, event in enumerate(events):
        start = parse_time_str(event.get("start_time"))
        end = parse_time_str(event.get("end_time"))
        if end == 0.0: end = start + 1.0 
        
        cat = event.get("category", "General")
        sub = event.get("subject") or event.get("player", "球員")
        act = event.get("action", "")
        det = event.get("detail", "")
        
        event_str = f"[{cat}] {sub} - {act}"
        if det: event_str += f" ({det})"

        gap_from_prev = start - last_event_end
        should_start_new_block = False
        
        if not current_block:
            should_start_new_block = True
        elif gap_from_prev > 2.0: 
            should_start_new_block = True
        else:
            current_block_dur = end - block_start_time
            if current_block_dur > MAX_RALLY_DURATION:
                should_start_new_block = True
        
        if should_start_new_block:
            if current_block:
                narrative_blocks.append({
                    "type": "RALLY",
                    "start": block_start_time,
                    "end": last_event_end,
                    "content": " -> ".join(current_block)
                })
                if gap_from_prev > 2.0:
                    narrative_blocks.append({
                        "type": "GAP",
                        "start": last_event_end,
                        "end": start,
                        "content": "中場間隙/調整"
                    })

            current_block = [event_str]
            block_start_time = start
        else:
            current_block.append(event_str)
        
        last_event_end = end

    if current_block:
        narrative_blocks.append({
            "type": "RALLY",
            "start": block_start_time,
            "end": last_event_end,
            "content": " -> ".join(current_block)
        })

    # 3. 處理結尾
    if total_duration - last_event_end > 2.0:
        narrative_blocks.append({
            "type": "OUTRO",
            "start": last_event_end,
            "end": total_duration,
            "content": "本段結束/重播畫面"
        })

    # --- B. 準備 LLM 輸入資料 (已修改：保存 raw_content) ---
    llm_input_data = []
    final_blocks_map = [] 

    for idx, block in enumerate(narrative_blocks):
        duration = block["end"] - block["start"]
        if duration < 0.5: continue 

        syllable_limit = int(duration * SYLLABLES_PER_SEC)
        syllable_limit = max(syllable_limit, 6) 
        
        info_text = block["content"]
        if block["type"] == "GAP": info_text = "[Gap] 中場休息/球員特寫"
        if block["type"] == "INTRO": info_text = "[Intro] 比賽開始"

        llm_input_data.append({
            "id": idx,
            "constraint": f"限 {syllable_limit} 音節",
            "content": info_text
        })
        
        # 🔥 修改處 1：保存原始內容以便後續判斷類型
        final_blocks_map.append({
            "id": idx,
            "start": block["start"],
            "end": block["end"],
            "type": block["type"],
            "raw_content": info_text.lower() # 轉小寫存起來
        })

    # --- C. 呼叫 LLM ---
    if NARRATIVE_HISTORY:
        recent_history = NARRATIVE_HISTORY[-HISTORY_WINDOW_SIZE:]
        history_str = "\n".join([f"- {h}" for h in recent_history])
    else:
        history_str = "這是比賽的第一個片段。"

    try:
        res = pipeline_s2.run({
            "add_video": {"uri": video_uri},
            "prompt_builder": {
                "event_data": json.dumps(llm_input_data, ensure_ascii=False, indent=2),
                "prev_context": history_str 
                }
        })
        reply = res["llm"]["replies"][0].strip()
        if "```" in reply:
            reply = re.search(r'\[.*\]', reply, re.DOTALL).group()
        
        generated_list = json.loads(reply)
        generated_map = {item["id"]: item["text"] for item in generated_list}
    except Exception as e:
        print(f"❌ [Stage 2 LLM 錯誤] {e}")
        return None

    # --- D. 輸出結果 (已修改：動態排軸優化) ---
    
    # 🔥 定義動作延遲表 (單位：秒)
    DELAY_MAP = {
        "setup": 2.2,    # 發球/準備：動作長，往後推 2.2 秒再講
        "serve": 2.2,    
        "offense": 0.6,  # 殺球/進攻：模擬反應時間 0.6 秒
        "smash": 0.6,    
        "defense": 0.8,  # 防守
        "score": 0.1,    # 得分：球落地馬上喊
        "gap": 0.5,      # 間隙：稍微留白
        "intro": 0.0,    
        "default": 0.8   
    }
    
    MIN_BLOCK_DURATION = 1.3 
    commentary = []
    segment_narrative_text = []
    
    # 指針：記錄上一句話結束時間，防止重疊
    last_speech_end_time = 0.0

    for block_meta in final_blocks_map:
        bid = block_meta["id"]
        text = generated_map.get(bid, "")
        if not text: continue

        # 1. 取出原始資料
        raw_start = block_meta["start"]
        raw_content = block_meta.get("raw_content", "")
        block_type = block_meta["type"]
        
        # 2. 判斷延遲時間
        adjusted_start = raw_start

        # 4. 🔥 防重疊機制
        if adjusted_start < last_speech_end_time + 0.15:
            adjusted_start = last_speech_end_time + 0.15
            
        # 5. 計算結束時間 (基於文字長度動態估算)
        estimated_speech_dur = len(text) / SYLLABLES_PER_SEC
        target_duration = max(estimated_speech_dur, MIN_BLOCK_DURATION)
        
        adjusted_end = adjusted_start + target_duration
        
        # 6. 邊界檢查
        if adjusted_end > total_duration:
            adjusted_end = total_duration
            if adjusted_end - adjusted_start < 1.0:
                adjusted_start = max(0, adjusted_end - 1.0)

        # 7. 更新指針
        last_speech_end_time = adjusted_end

        # 8. 情緒標籤
        emotion = "平穩"
        if block_type == "RALLY":
            if any(k in raw_content for k in ["offense", "score", "smash", "kill"]):
                emotion = "激動"
        elif block_type == "GAP":
            emotion = "舒緩"

        commentary.append({
            "start_time": seconds_to_timecode(adjusted_start),
            "end_time": seconds_to_timecode(adjusted_end),
            "time_range": format_duration(adjusted_end - adjusted_start),
            "emotion": emotion,
            "text": text
        })
        segment_narrative_text.append(text)

    # 更新歷史紀錄
    if segment_narrative_text:
        NARRATIVE_HISTORY.append(" ".join(segment_narrative_text))
        if len(NARRATIVE_HISTORY) > 10: NARRATIVE_HISTORY.pop(0)

    # 存檔
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