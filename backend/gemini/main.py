import os
import time
import threading
import queue
from tqdm import tqdm

# 引入我們之前改好的單檔處理函式
from videogen_stage1 import process_single_video_stage1
from videogen_stage2 import process_single_video_stage2

# 建立一個無限大小的佇列，用來傳遞 Stage 1 完成的任務給 Stage 2
task_queue = queue.Queue()

def count_files(folder, extension):
    if not os.path.exists(folder): return 0
    return len([f for f in os.listdir(folder) if f.endswith(extension)])

def format_seconds(seconds):
    return f"{int(seconds // 60)}分 {int(seconds % 60)}秒"

# ========== 執行緒 1：生產者 (負責跑 Stage 1) ==========
def stage1_producer(video_files, video_folder, event_json_folder, intro_text):
    print("👁️ [Stage 1 執行緒] 啟動，開始分析影像...")
    
    for i, file_name in enumerate(video_files):
        video_path = os.path.join(video_folder, file_name)
        print(f"\n[Stage 1] 正在分析第 {i+1} 支: {file_name}")
        
        # 執行 Stage 1
        json_path = process_single_video_stage1(video_path, event_json_folder, intro_text)
        
        if json_path and os.path.exists(json_path):
            # 成功！將任務打包放入佇列，讓 Stage 2 去撿
            # 我們傳遞一個 tuple: (影片路徑, JSON路徑)
            task_queue.put((video_path, json_path))
            print(f"✅ [Stage 1] {file_name} 完成 -> 已加入 Stage 2 佇列")
        else:
            print(f"❌ [Stage 1] {file_name} 失敗，不進行後續處理")

    # 全部影片都處理完了，放入一個 "毒藥丸 (Poison Pill)" 告訴 Stage 2 可以下班了
    task_queue.put(None)
    print("🏁 [Stage 1 執行緒] 所有影片分析完畢，準備結束。")

# ========== 執行緒 2：消費者 (負責跑 Stage 2) ==========
def stage2_consumer(final_output_folder):
    print("✍️ [Stage 2 執行緒] 待命，等待 Stage 1 的產出...")
    
    success_count = 0
    
    while True:
        # 從佇列中拿取任務 (如果佇列是空的，這裡會自動等待，直到有東西進來)
        task = task_queue.get()
        
        # 檢查是否收到結束訊號 (毒藥丸)
        if task is None:
            task_queue.task_done()
            break
        
        video_path, json_path = task
        file_name = os.path.basename(video_path)
        
        print(f"\n   🚀 [Stage 2] 收到任務，開始生成敘事: {file_name}")
        
        # 執行 Stage 2
        try:
            result = process_single_video_stage2(video_path, json_path, final_output_folder)
            if result:
                print(f"   ✅ [Stage 2] {file_name} 敘事生成完畢！")
                success_count += 1
            else:
                print(f"   ⚠️ [Stage 2] {file_name} 生成失敗")
        except Exception as e:
            print(f"   ❌ [Stage 2] 發生錯誤: {e}")

        # 標記此任務已完成
        task_queue.task_done()

    print(f"🏁 [Stage 2 執行緒] 工作結束。共完成 {success_count} 支敘事。")

# ========== 主程式 ==========
def main():
    # 設定路徑
    base_dir = "D:/Vs.code/AI_Anchor"
    video_folder = os.path.join(base_dir, "backend/video_splitter/badminton_segments(1126test)")
    event_json_folder = os.path.join(base_dir, "backend/gemini/event_analysis_output")
    final_output_folder = os.path.join(base_dir, "backend/gemini/final_narratives")

    # 掃描影片
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(".mp4")])
    total_videos = len(video_files)

    if total_videos == 0:
        print("❌ 找不到影片。")
        return

    print(f"\n🚀 [並行流水線模式] 啟動！共 {total_videos} 支影片")
    print("說明：Stage 1 (分析) 與 Stage 2 (寫稿) 將同時進行，大幅縮短等待時間。\n")
    
    intro_text = input("請輸入背景介紹 (Enter 跳過)：") or "羽球比賽"
    
    global_start = time.time()

    # 建立並啟動 Stage 1 執行緒
    t1 = threading.Thread(target=stage1_producer, args=(video_files, video_folder, event_json_folder, intro_text))
    
    # 建立並啟動 Stage 2 執行緒
    t2 = threading.Thread(target=stage2_consumer, args=(final_output_folder,))

    # 開始跑！
    t1.start()
    t2.start()

    # 主程式等待兩個執行緒都跑完
    t1.join()
    t2.join()

    # 最終統計
    total_time = time.time() - global_start
    print("\n" + "="*50)
    print(f"🎉 所有流程完美結束！")
    print(f"⏱️ 總耗時：{format_seconds(total_time)}")
    print(f"⚡ 平均每支：{total_time/total_videos:.1f} 秒 (含並行加速)")
    print("="*50)

if __name__ == "__main__":
    main()

# text = 黑色球衣是台灣的戴資穎，白色球衣是印度的辛度。