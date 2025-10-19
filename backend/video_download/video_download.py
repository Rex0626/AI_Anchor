import yt_dlp
import os
import imageio_ffmpeg  # 导入 imageio_ffmpeg 包

# 默认下载目录（当用户未自定义时使用）
DEFAULT_SAVE_PATH = os.path.join(os.getcwd(), "video_download")

def download_youtube_video(url, file_name, use_original_title, format_type, save_path):
    # 如果用户输入的路径不是绝对路径，则以当前工作目录为基准
    if not os.path.isabs(save_path):
        save_path = os.path.join(os.getcwd(), save_path)
    os.makedirs(save_path, exist_ok=True)

    # 获取 imageio_ffmpeg 提供的 ffmpeg 路径
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    if format_type.lower() == 'mp3':
        options = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s') if use_original_title 
                        else os.path.join(save_path, f"{file_name}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'ffmpeg_location': ffmpeg_path  # 指定 ffmpeg 路径
        }
    else:
        options = {
            # 修改格式选项，强制选择 MP4 视频流和 M4A 音频流，确保合并后音视频都正常
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s') if use_original_title 
                        else os.path.join(save_path, f"{file_name}.%(ext)s"),
            'merge_output_format': 'mp4',
            'ffmpeg_location': ffmpeg_path
        }

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info_dict = ydl.extract_info(url, download=True)  # 下载并获取视频信息
            original_title = info_dict.get('title', '未命名')    # 获取视频原始标题

        # 计算最终文件名（仅用于提示，并不影响实际保存的文件名）
        final_name = f"{original_title}.{format_type.lower()}" if use_original_title else f"{file_name}.{format_type.lower()}"
        print(f"下载完成！文件已保存至：{os.path.join(save_path, final_name)}")
    except Exception as e:
        print(f"下载失败: {e}")

if __name__ == "__main__":
    url = input("请输入 YouTube 影片链接: ").strip()

    # 自定义保存路径
    save_path = input("请输入保存路径（默认为 video_download 目录）: ").strip() or DEFAULT_SAVE_PATH

    # 是否使用原始视频名称
    use_original_title_input = input("是否使用原始视频名称？（Y/N，默认 Y）: ").strip().lower() or "y"
    if use_original_title_input == "y":
        file_name = ""  # 使用原始视频名称
        use_original_title = True
    else:
        file_name = input("请输入储存的文件名（不含扩展名）: ").strip()
        use_original_title = False

    # 选择下载格式
    format_type = input("请选择下载格式（MP3 或 MP4，默认为 MP4）: ").strip().lower() or "mp4"
    if format_type not in ["mp3", "mp4"]:
        print("格式无效，默认为 MP4")
        format_type = "mp4"

    download_youtube_video(url, file_name, use_original_title, format_type, save_path)
