import { useState } from "react";

function UploadPage({ setPage, setJobId, setSegments, setCurrentSegment }) {
  const [file, setFile] = useState(null);
  const [desc, setDesc] = useState("");
  const [language, setLanguage] = useState("中文");
  const [style, setStyle] = useState("專業分析型");
  const [loading, setLoading] = useState(false);

  // 選檔案
  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) setFile(selected);
  };

  // 上傳表單
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("請先選擇影片");
      return;
    }

    const formData = new FormData();
    formData.append("video", file);
    formData.append("description", desc);
    formData.append("language", language);
    formData.append("style", style);

    setLoading(true);
    try {
      // 🚀 呼叫後端 API：/api/init_job
      const res = await fetch("http://127.0.0.1:5000/api/init_job", {
        method: "POST",
        body: formData
      });
      
      const data = await res.json();

      if (data.status === "success") {
        setJobId(data.job_id);
        setSegments(data.segments);
        setCurrentSegment(0);
        setPage("result");
      } else {
        alert("切片失敗：" + (data.message || "未知錯誤"));
      }
    } catch (err) {
      alert("錯誤：" + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-12">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-md p-8">
        <h1 className="text-3xl font-bold text-center text-gray-800 mb-8">
          運動賽事 AI 轉播系統
        </h1>
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* 檔案選擇 */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
            <input
              type="file"
              accept="video/*"
              className="hidden"
              id="videoUpload"
              onChange={handleFileChange}
            />
            <button
              type="button"
              onClick={() => document.getElementById("videoUpload").click()}
              className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700"
            >
              選擇影片檔
            </button>
            {file && (
              <>
                <p className="mt-2 text-sm text-gray-600">{file.name}</p>
                <div className="mt-4">
                  <video className="rounded-lg" src={URL.createObjectURL(file)} controls />
                </div>
              </>
            )}
          </div>

          {/* 描述 */}
          <div>
            <label className="block text-gray-700 font-medium mb-1">選手描述</label>
            <textarea
              value={desc}
              onChange={(e) => setDesc(e.target.value)}
              required
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:border-indigo-500"
              placeholder="例：紅色衣服是..."
            />
          </div>

          {/* 語言 & 風格 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-700 font-medium mb-1">語言</label>
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              >
                <option>中文</option>
                <option>English</option>
                <option>日本語</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-700 font-medium mb-1">風格</label>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg"
              >
                <option>專業分析型</option>
                <option>激情解說型</option>
                <option>輕鬆娛樂型</option>
              </select>
            </div>
          </div>

          {/* 送出按鈕 */}
          <div className="text-center pt-4">
            <button
              type="submit"
              disabled={loading}
              className="px-8 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              {loading ? "處理中..." : "送出分析請求"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default UploadPage;
