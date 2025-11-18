import os
# 假設您的檔案分別命名為 videogen_stage1.py 和 videogen_stage2.py
from videogen_stage1 import process_stage1_events
from videogen_stage2 import process_stage2_narratives

def main():
    # ========== 設定路徑 ==========
    base_dir = "D:/Vs.code/AI_Anchor"
    
    # 1. 原始影片資料夾
    video_folder = os.path.join(base_dir, "backend/video_splitter/badminton_segments")
    
    # 2. 中間產物資料夾 (Stage 1 輸出 -> Stage 2 輸入)
    event_json_folder = os.path.join(base_dir, "backend/gemini/event_analysis_output")
    
    # 3. 最終結果資料夾 (Stage 2 輸出 -> TTS 輸入)
    final_output_folder = os.path.join(base_dir, "backend/gemini/final_narratives")

    # ========== 使用者輸入 ==========
    intro_text = input("請輸入影片背景介紹 (例如：這是一場羽球比賽...)：")

    print("\n🎬 [主程式] 開始執行 AI 主播生成流程...")

    # ========== 執行 Stage 1: 事件分析 ==========
    print("\n========== 執行 Stage 1: 影片上傳與事件分析 ==========")
    try:
        process_stage1_events(video_folder, event_json_folder, intro_text)
    except Exception as e:
        print(f"❌ Stage 1 發生嚴重錯誤，流程終止: {e}")
        return

    # ========== 執行 Stage 2: 敘事生成 ==========
    print("\n========== 執行 Stage 2: 敘事生成與時間軸校準 ==========")
    try:
        # Stage 2 需要讀取原始影片(獲取時長) 和 Stage 1 的 JSON
        process_stage2_narratives(video_folder, event_json_folder, final_output_folder)
    except Exception as e:
        print(f"❌ Stage 2 發生嚴重錯誤: {e}")
        return

    print(f"\n✅ [主程式] 全流程結束！最終結果已儲存至: {final_output_folder}")

if __name__ == "__main__":
    main()