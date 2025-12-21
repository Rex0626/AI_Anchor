# 🛠️ AI Anchor Backend System (後端核心系統)

本目錄包含「賽場揚聲」專案的所有後端邏輯，負責處理從影片輸入、AI 分析、敘事生成到最終合成的自動化流水線。

## ⚙️ 環境建置 (Installation)

本系統開發於 **Python 3.10+** 環境。

### 1. 安裝相依套件
請在專案根目錄執行以下指令，安裝 `requirements.txt` 中列出的所有依賴：
```bash
pip install -r ../requirements.txt
```

### 2. Google Cloud 憑證設定
* 本系統依賴 Google Vertex AI 與 Google Cloud Text-to-Speech。
* 請確保您擁有有效的 Google Cloud Service Account Key (.json 格式)。
* 將金鑰檔案放入專案根目錄的 credentials/ 資料夾中。
> 注意：程式碼預設會讀取 ../credentials/ 下的 JSON 檔案，請確保檔案路徑正確。

## 📂 專案結構與程式碼說明 (Project Structure)

本專案後端採用 Python 開發，採用模組化設計。主要程式碼位於 `backend/` 目錄下：

```text
AI_Anchor/
├── backend/
│   ├── detection/              # [⚠️ 研究對照組] 傳統電腦視覺模組
│   │   └── detection.py        # 包含 YOLOv8 + DeepSORT + MediaPipe 的實作代碼。
│   │                           # ★ 注意：本專案核心採用 LLM 虛擬視覺感知，此資料夾僅作為
│   │                           # 傳統技術的效能對照與實驗用途，並未使用於最終自動化流程中。
│   │
│   ├── gemini/                 # [核心模組] LLM 雙層生成架構
│   │   ├── videogen_stage1.py  # Stage 1: 虛擬視覺感知 (影片 -> 事件 JSON)
│   │   ├── videogen_stage2.py  # Stage 2: 敘事推理 (事件 JSON -> 解說文本 JSON)
│   │   ├── videogen.py         # (舊版/單一流程) 影片生成邏輯
│   │   └── main.py             # 系統主程式：採用生產者-消費者模式並行處理 Stage 1 & 2
│   │
│   ├── TextToSpeech/           # [語音模組] 情緒語音合成
│   │   └── generate_tts_google.py # 呼叫 Google Cloud TTS，執行情緒參數映射與 SSML 轉換
│   │
│   ├── video_download/         # [工具模組] 影片下載
│   │   └── video_download.py   # 支援 YouTube 影片下載 (yt-dlp)
│   │
│   ├── video_splitter/         # [前處理模組] 影片切割
│   │   └── video_splitter.py   # 使用 MoviePy 將長影片切分為短片段
│   │
│   ├── merge_audio/            # [後處理模組] 單片段影音合併
│   │   └── merge_audio.py      # 將生成的 TTS 音檔與原始影片片段進行對齊與合併
│   │
│   └── video_merger/           # [後處理模組] 最終合併
│       └── video_merge.py      # 將所有處理好的片段串接為完整的最終影片
│
├── credentials/                # Google Cloud 憑證存放區 (請自行放入 JSON 金鑰)
├── requirements.txt            # 專案依賴套件清單
└── README.md                   # 專案說明文件
```

## 📂 模組詳細說明 (Module Details)
### 1. 核心生成模組 (gemini/)
這是系統的「大腦」，採用 Haystack 框架與 Google Vertex AI 構建，實現了我們提出的「感知-推理解耦」架構。

* main.py (系統主入口)
> * 架構設計：採用多執行緒 (Multi-threading) 的「生產者-消費者 (Producer-Consumer)」模式。

> * 運作邏輯：
>> * 執行緒 1 (Producer)：負責執行 Stage 1，將影片切片送入 LLM 進行視覺分析，產出 JSON。
>> * 執行緒 2 (Consumer)：監聽任務佇列 (Queue)，當 Stage 1 完成分析後立即接手執行 Stage 2，進行敘事撰寫。
> * 優勢：大幅縮短長影片處理的等待時間，實現流水線式作業。

* videogen_stage1.py (Stage 1: 虛擬視覺感知)
> * 任務：扮演「虛擬電腦視覺系統」，進行客觀的動作捕捉。
> * 核心技術：利用 Prompt Engineering 強制 LLM 輸出結構化的 JSON 事件日誌 (Event Logs)，包含時間戳記、球員身分與動作分類。
> * 關鍵機制：外部資訊注入 (External Info Injection) — 將 intro (球員背景資訊) 動態嵌入 Prompt，解決通用模型無法識別特定球員的問題。

* videogen_stage2.py (Stage 2: 敘事推理)
> * 任務：扮演「專業體育主播」，將客觀事件轉化為生動的解說文本。
> * 核心技術：

>> * 滑動視窗記憶 (Sliding Window Memory)：傳遞前一段生成的解說內容作為 Context，確保跨片段的敘事連貫性。
>> * 音節限制 (Syllable Limitation)：根據物理時間計算可容納的字數上限，防止語音超時。
>> * 延遲合併 (Delayed Merging)：將過於零碎的動作（如連續平抽）合併為一個完整的攻防回合 (Rally) 進行描述。

### 2. 語音合成模組 (TextToSpeech/)
* generate_tts_google.py
> * 功能：將生成的文本轉換為 MP3 音檔。
> * 情緒映射機制 (Emotion Mapping)： 系統會讀取 Stage 2 輸出的情緒標籤（如 【激動】、【遺憾】），動態調整 SSML 參數：
>> * 激動：語速 1.7x, 音量 +3.5dB
>> * 緊張：語速 1.4x, 音量 +2.0dB
>> * 平穩：語速 1.2x, 音量 0dB
> * 快取機制：使用 SHA-256 對文本進行雜湊比對，若文本未更動則跳過 API 呼叫，節省成本與時間。

### 3. 影片處理模組 (video_splitter/, merge_audio/, video_merger/)
* video_splitter.py：使用 MoviePy 將長影片切分為固定長度（如 30 秒）的片段，以便 LLM 處理。
* merge_audio.py：將生成的 TTS 音檔與原始影片片段進行時間軸對齊與合成。
* video_merge.py：將所有處理完畢的小片段無縫串接回一支完整的賽事影片。

### ⚠️ 4. 實驗性對照模組 (detection/)
注意：本資料夾內的程式碼僅作為研究對照用途，並未整合至自動化流水線中。
* 內容：包含使用 YOLOv8 (物件偵測)、DeepSORT (多物件追蹤) 與 MediaPipe (骨架分析) 的實作程式碼。
* 目的：在專題研究過程中，我們保留此模組是為了與我們提出的「多模態 LLM 虛擬視覺」方案進行效能與準確度的對照實驗。
* 結論：傳統 CV 方法在理解複雜戰術意圖上存在語意鴻溝，且訓練成本過高，因此本專案最終選擇了 LLM 方案。此資料夾代碼證明了我們對不同技術路線的探索與驗證。