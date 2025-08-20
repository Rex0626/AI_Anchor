# AI_Anchor
🎙️ AI 主播系統 — 後端模組
本專案能將運動賽事影片自動化轉換為文字解說與語音播報。流程包含：
1.影片切割
2.YOLO 物件偵測與追蹤、MediaPipe 動作辨識
3.LLM 文字生成
4.TTS 語音合成
5.最終輸出完整的 AI 播報結果。

此系統特別針對視障者與冷門賽事設計。

其中每一個流程對應的資料夾如下：
影片切割 --> video_splitter
物件偵測、追蹤與動作辨識 --> detection
LLM文字生成 --> gemini
語音生成 --> TextToSpeech
文字、語音、影片合併 --> merge_audio
影片片段合併 --> video_merge
