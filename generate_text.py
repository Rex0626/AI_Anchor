# 轉播文本生成 (generate_text.py)

import os
import json
import time
from google import genai
from google.genai.errors import APIError
import main
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type # <-- 新增導入

# 從 main.py 匯入常數
MODEL = main.MODEL_TEXT_GEN
VIDEO_PATH = main.VIDEO_PATH
PLAN_FILE = main.PLAN_FILE
API_KEY = main.API_KEY

# ... (wait_for_file_active 函式保持不變) ...

# 定義重試邏輯：
# - 遇到 APIError 時才重試
# - 最多重試 5 次 (共 6 次嘗試)
# - 使用指數退避等待時間 (1 秒開始，最長 16 秒)
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type(APIError)
)
def call_gemini_with_retry(client, file_obj, prompt):
    """帶有重試機制的 Gemini API 呼叫函式"""
    print("  -> 正在呼叫 Gemini API (如果失敗，將自動重試)...")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=[
                file_obj,
                prompt
            ]
        )
        return response
    except APIError as e:
        # 僅在發生特定錯誤時才觸發 tenacity 重試
        if "503" in str(e) or "overloaded" in str(e):
            print(f"  -> 檢測到 503 服務錯誤，將在 {e.wait} 秒後重試...")
            raise # 拋出異常以觸發 tenacity 重試
        else:
            raise # 拋出其他類型的 API 錯誤

def wait_for_file_active(client, file_obj, timeout=300):

    """等待檔案上傳並處理完成"""

    start = time.time()

    print(f"等待檔案 {VIDEO_PATH} 處理中...")

    while time.time() - start < timeout:

        status = client.files.get(name=file_obj.name)

        if status.state.name == "ACTIVE":

            print("檔案處理完成。")

            return

        if status.state.name == "FAILED":

            raise Exception("檔案處理失敗，請檢查影片格式或 API 權限。")

        time.sleep(3)

    raise TimeoutError("等待檔案處理超時")

def generate_plan():
    """執行影片上傳、文本生成和檔案清理"""
    
    # 1. 初始化 Gemini Client
    try:
        if not API_KEY or API_KEY == "YOUR_GEMINI_API_KEY_HERE":
            raise ValueError("錯誤：請在 main.py 中設定您的實際 Gemini API Key。")
        
        client = genai.Client(api_key=API_KEY)
    except Exception as e:
        print(f"初始化 Gemini Client 失敗: {e}")
        return
        
    file_obj = None

    try:
        print(f"1. 檢查影片檔案: {VIDEO_PATH}")
        if not os.path.exists(VIDEO_PATH):
            raise FileNotFoundError(f"錯誤：找不到影片檔案 {VIDEO_PATH}。請確認檔案存在。")

        # 2. 上傳影片
        print(f"2. 正在上傳影片 {VIDEO_PATH}...")
        file_obj = client.files.upload(file=VIDEO_PATH)
        wait_for_file_active(client, file_obj)

        # 3. 呼叫 Gemini 進行文本生成 (使用重試函式)
        print(f"3. 呼叫模型 {MODEL} 生成轉播文本...")
        response = call_gemini_with_retry(client, file_obj, main.PROMPT) 
        
        # 4. 處理並儲存 JSON 輸出
        
        raw_text = response.text.strip()
        
        # 4.1. 關鍵修正：更積極地隔離 JSON 程式碼塊
        # 尋找第一個 '{' 和最後一個 '}' 的位置，只取這之間的內容。
        start_index = raw_text.find('{')
        end_index = raw_text.rfind('}')
        
        if start_index == -1 or end_index == -1 or end_index < start_index:
             # 如果找不到有效的 JSON 邊界，則返回錯誤
             raise Exception("模型輸出中找不到有效的 JSON 結構 (缺少 { 或 })。請重新運行。")
        
        # 提取被認為是 JSON 的內容，並移除可能存在的 Markdown 標記
        json_text = raw_text[start_index : end_index + 1].strip()
        
        try:
            # 4.2. 嘗試解析 JSON
            plan_data = json.loads(json_text)
        
        except json.JSONDecodeError as e:
            # 4.3. 解析失敗時，提供診斷資訊
            # 由於您不希望印出內容，我們只拋出錯誤資訊
            raise Exception(f"JSON 格式錯誤：{e.msg} (第 {e.lineno} 行) - 模型輸出不符預期") from e

        # 4.4. 成功儲存
        with open(PLAN_FILE, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, ensure_ascii=False, indent=2)
            
        print(f"4. 轉播文本已成功儲存到 {PLAN_FILE}。")

    except APIError as e:
        print(f"\n致命錯誤：Gemini API 最終失敗：{e}")
    except Exception as e:
        # 這裡會捕獲我們在上面拋出的 JSON 錯誤
        print(f"\n發生非 API 錯誤：{e}")
    finally:
        # 5. 清理檔案
        if file_obj:
            print(f"5. 正在清理上傳的檔案 {file_obj.name}...")
            client.files.delete(name=file_obj.name)
            print("檔案清理完成。")

if __name__ == "__main__":
    generate_plan()