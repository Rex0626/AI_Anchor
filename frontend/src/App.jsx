import { useState } from "react";
import UploadPage from "./pages/UploadPage";
import ResultPage from "./pages/ResultPage";

function App() {
  const [page, setPage] = useState("upload");
  const [jobId, setJobId] = useState("");
  const [segments, setSegments] = useState([]);
  const [currentSegment, setCurrentSegment] = useState(null);

  return (
    <div className="bg-gray-100 min-h-screen">
      {page === "upload" && (
        <UploadPage
          setPage={setPage}
          setJobId={setJobId}
          setSegments={setSegments}
          setCurrentSegment={setCurrentSegment}
        />
      )}
      {page === "result" && (
        <ResultPage
          jobId={jobId}
          segments={segments}
          currentSegment={currentSegment}
          setCurrentSegment={setCurrentSegment}
          setPage={setPage}
        />
      )}
    </div>
  );
}

export default App;
