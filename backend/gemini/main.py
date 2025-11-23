import os
import time
from videogen_stage1 import process_stage1_events
from videogen_stage2 import process_stage2_narratives

def count_files(folder, extension):
    """輔助函式：計算檔案數量"""
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(extension)])

def format_seconds(seconds):
    """將秒數轉為 分:秒 格式，方便閱讀"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}分 {s}秒 ({seconds:.2f}s)"

def main():
    # ========== 設定路徑 ==========
    base_dir = "D:/Vs.code/AI_Anchor"
    video_folder = os.path.join(base_dir, "backend/video_splitter/badminton_segments")
    event_json_folder = os.path.join(base_dir, "backend/gemini/event_analysis_output")
    final_output_folder = os.path.join(base_dir, "backend/gemini/final_narratives")

    # ========== 初始化檢查 ==========
    total_videos = count_files(video_folder, ".mp4")
    if total_videos == 0:
        print("❌ 錯誤：找不到影片檔案。")
        return

    print(f"\n📊 任務隊列：共 {total_videos} 支影片待處理")
    intro_text = input("請輸入影片背景介紹 (Enter 跳過)：") or "羽球比賽精彩片段"

    print("\n🎬 [主程式] 計時開始...")
    
    # 記錄總開始時間
    global_start_time = time.time()

    # ==========================================
    # ⏱️ 執行 Stage 1 並計時
    # ==========================================
    print("\n🔄 [Stage 1] 啟動：事件分析...")
    s1_start = time.time()
    
    try:
        process_stage1_events(video_folder, event_json_folder, intro_text)
    except Exception as e:
        print(f"❌ Stage 1 中斷: {e}")
        return

    s1_end = time.time()
    s1_duration = s1_end - s1_start
    
    # 檢查產出
    json_count = count_files(event_json_folder, "_event.json")
    if json_count == 0:
        print("⚠️ Stage 1 未產出任何檔案，流程終止。")
        return

    # ==========================================
    # ⏱️ 執行 Stage 2 並計時
    # ==========================================
    print("\n🔄 [Stage 2] 啟動：敘事生成...")
    s2_start = time.time()

    try:
        process_stage2_narratives(video_folder, event_json_folder, final_output_folder)
    except Exception as e:
        print(f"❌ Stage 2 中斷: {e}")
        return

    s2_end = time.time()
    s2_duration = s2_end - s2_start

    # ==========================================
    # 📊 最終效能報告
    # ==========================================
    global_end_time = time.time()
    total_duration = global_end_time - global_start_time
    
    # 計算平均效率
    avg_time_per_video = total_duration / total_videos if total_videos > 0 else 0

    print("\n" + "="*50)
    print(f"✅ [執行完成] 效能統計報告")
    print("="*50)
    print(f"📂 處理影片數 ： {total_videos} 支")
    print("-" * 30)
    print(f"1️⃣ Stage 1 耗時： {format_seconds(s1_duration)}")
    print(f"2️⃣ Stage 2 耗時： {format_seconds(s2_duration)}")
    print("-" * 30)
    print(f"⏱️ 總執行時間  ： {format_seconds(total_duration)}")
    print(f"⚡ 平均速度    ： 每支影片約需 {avg_time_per_video:.2f} 秒")
    print("="*50)
    print(f"💾 最終檔案位置： {final_output_folder}")

if __name__ == "__main__":
    main()