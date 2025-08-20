import os
import sys
from moviepy.editor import VideoFileClip, concatenate_videoclips

def merge_videos(input_folder, output_video):
    print(f"📁 開始合併資料夾：{input_folder}")

    try:
        video_files = sorted([
            f for f in os.listdir(input_folder)
            if f.lower().endswith(('.mp4', '.webm', '.avi', '.mov'))
        ])

        if not video_files:
            print("❌ 沒有找到任何影片，請檢查資料夾與副檔名！")
            return

        clips = []
        for file in video_files:
            file_path = os.path.join(input_folder, file)
            try:
                clip = VideoFileClip(file_path)
                print(f"✅ 成功讀取：{file}（時長：{clip.duration:.2f} 秒）")
                clips.append(clip)
            except Exception as e:
                print(f"⚠️ 讀取失敗：{file_path} -> {e}")
                continue

        if not clips:
            print("❌ 所有影片都無法讀取，無法合併。")
            return

        print(f"\n📦 合併總片段數：{len(clips)}")
        total_duration = sum([clip.duration for clip in clips])
        print(f"🕒 合併後總時長預估：{total_duration:.2f} 秒")

        final_clip = concatenate_videoclips(clips, method="compose")
        final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

        # 關閉所有 clip
        for clip in clips:
            clip.close()
        final_clip.close()

        print(f"\n🎉 ✅ 合併完成！輸出影片：{output_video}")

    except Exception as e:
        print(f"❌ 合併時發生錯誤：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 🛠️ 請根據實際需求修改以下兩個路徑：
    input_folder = "D:/Vs.code/AI_Anchor/merge_audio/badminton_outputs"
    output_video = "D:/Vs.code/AI_Anchor/video_merger/output/badminton_final_outputs.mp4"

    merge_videos(input_folder, output_video)
