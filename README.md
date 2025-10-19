# 🎙️ 賽場之聲 - 多模態融合即時賽事解說平台

## 專案總覽 (Project Overview)

本專案是一個創新的**多模態 AI 賽事解說平台**，旨在將運動賽事影片自動化轉換為專業的文字解說與富有情感的語音播報。

---

### 📦 專案結構與模組職責

| 資料夾 | 職責 | 核心技術 |
|---|---|---|
| `backend/` | **AI 核心後段模組**：負責影片處理、視覺分析、LLM 旁白生成、TTS 語音合成及影片最終合併。 | Python, YOLOv8, Gemini LLM |
| `frontend/` | **使用者介面**：預留給網頁、應用程式或操作介面。 | [待定：React/Vue/etc.] |
| `Live/` | **及時處理模組**：預留給及時處理的部分。 | [待定：Live API] |
| `tools/` | **輔助工具**：例如影片下載工具等。 | |

---

### 🚀 快速入門 (Quick Start for All Team Members)

1. **複製專案**：
   ```bash
   git clone [https://github.com/Rex0626/AI_Anchor](https://github.com/Rex0626/AI_Anchor)_
   cd AI_Anchor