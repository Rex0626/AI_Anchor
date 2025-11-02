import { useEffect, useState } from "react";


function ResultPage({ jobId, videoName, segments, currentSegment, setCurrentSegment, setPage, description }) {
  const [results, setResults] = useState({}); // 🧠 存每個片段的生成結果
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");


  // 🚀 第一次進入頁面時，自動開始第一段
  useEffect(() => {
    if (segments.length > 0 && currentSegment === null) {
      setCurrentSegment(0);
    }
  }, [segments]);


  // 🚀 當切換片段時，如果該片段還沒生成，就自動呼叫後端
  useEffect(() => {
    if (segments.length === 0 || currentSegment === null) return;


    const segKey = segments[currentSegment];
    if (!results[segKey]) {
      generateSegment(currentSegment);
    }
  }, [currentSegment, segments]);


  const generateSegment = async (index) => {
    const segKey = segments[index];
    try {
      setIsProcessing(true);
      setErrorMsg("");


      const formData = new FormData();
      formData.append("video_name", videoName);
      formData.append("description", description || "");
      formData.append("segment_index", index + 1);


      const res = await fetch("http://127.0.0.1:5000/api/process_segment_step", {
        method: "POST",
        body: formData,
      });


      const data = await res.json();


      if (data.status === "success") {
        const videoUrl = "http://127.0.0.1:5000" + data.video_url;
        const commentaryText = Array.isArray(data.commentary)
          ? data.commentary.map((c) => c.text).join("\n")
          : "（沒有生成文本）";


        // 🧠 將結果存起來
        setResults((prev) => ({
          ...prev,
          [segKey]: { videoUrl, commentaryText },
        }));
      } else {
        setErrorMsg("❌ 生成失敗：" + (data.message || "未知錯誤"));
      }
    } catch (err) {
      console.error("❌ API 錯誤:", err);
      setErrorMsg("❌ 伺服器錯誤，請稍後再試");
    } finally {
      setIsProcessing(false);
    }
  };


  // 🧠 當前選擇的片段
  const currentSegKey = segments[currentSegment];
  const currentData = results[currentSegKey] || {};


  return (
    <div className="container mx-auto px-4 py-12">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* 左邊：影片與文本 */}
        <div className="md:col-span-2 bg-white rounded-xl shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">📺 最新轉播片段</h2>


          <div className="bg-gray-200 rounded-lg mb-4 aspect-video flex items-center justify-center">
            {currentData.videoUrl ? (
              <video
                src={currentData.videoUrl}
                controls
                autoPlay
                className="w-full h-full object-cover rounded-lg"
              />
            ) : isProcessing ? (
              <span className="text-gray-500 animate-pulse">正在生成影片...</span>
            ) : (
              <span className="text-gray-400">等待選擇或生成影片</span>
            )}
          </div>


          <textarea
            value={currentData.commentaryText || ""}
            onChange={(e) =>
              setResults((prev) => ({
                ...prev,
                [currentSegKey]: {
                  ...currentData,
                  commentaryText: e.target.value,
                },
              }))
            }
            className="w-full px-3 py-2 border border-gray-300 rounded-lg min-h-[150px] text-gray-700"
            placeholder="此片段的轉播文本..."
          />


          {errorMsg && (
            <div className="mt-2 p-2 text-red-600 bg-red-100 rounded text-sm border border-red-300">
              {errorMsg}
            </div>
          )}


          <div className="text-right mt-2 space-x-2">
            <button
              onClick={() => generateSegment(currentSegment)}
              disabled={isProcessing}
              className={`px-4 py-2 text-sm rounded ${
                isProcessing
                  ? "bg-gray-400 text-white"
                  : "bg-green-600 text-white hover:bg-green-700"
              }`}
            >
              {isProcessing ? "處理中..." : "重新生成"}
            </button>
          </div>
        </div>


        {/* 右邊：片段清單 */}
        <div className="bg-white rounded-xl shadow-md p-6 overflow-y-auto max-h-[70vh]">
          <h3 className="text-lg font-semibold text-gray-700 mb-4">🎮 已完成片段</h3>
          <ul className="space-y-3 text-sm text-gray-600">
            {segments.map((seg, idx) => {
              const done = !!results[seg];
              const isActive = currentSegment === idx;
              return (
                <li
                  key={idx}
                  className={`p-2 rounded cursor-pointer ${
                    isActive
                      ? "bg-indigo-100 font-semibold text-indigo-800"
                      : "hover:bg-gray-100"
                  }`}
                  onClick={() => setCurrentSegment(idx)}
                >
                  {`segment_${idx + 1}.mp4`}
                  {done ? " ✅" : isProcessing && isActive ? " (處理中)" : ""}
                </li>
              );
            })}
          </ul>
        </div>
      </div>


      {/* 下方：所有已完成的影片列表 */}
      <div className="mt-10 bg-white rounded-xl shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">📚 已生成片段總覽</h3>
        {Object.keys(results).length === 0 ? (
          <p className="text-gray-500">尚無生成結果</p>
        ) : (
          <div className="grid md:grid-cols-3 gap-4">
            {Object.entries(results).map(([seg, data], idx) => (
              <div key={idx} className="border rounded-lg p-3">
                <video
                  src={data.videoUrl}
                  controls
                  className="w-full rounded mb-2"
                />
                <p className="text-sm text-gray-600 whitespace-pre-wrap">
                  {data.commentaryText.slice(0, 80)}...
                </p>
              </div>
            ))}
          </div>
        )}
      </div>


      {/* 返回按鈕 */}
      <div className="text-right mt-6">
        <button
          onClick={() => setPage("upload")}
          className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-100"
        >
          返回上傳
        </button>
      </div>
    </div>
  );
}


export default ResultPage;
