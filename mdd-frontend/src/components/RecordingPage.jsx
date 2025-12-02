import React, { useState, useRef } from "react";
import "../styles/RecordingPage.css";

export default function RecordingPage() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [uploadedFileName, setUploadedFileName] = useState("");
  const [transcript, setTranscript] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showCorrection, setShowCorrection] = useState(false);
  const [activeSource, setActiveSource] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunks = useRef([]);

 
  const startRecording = async () => {
    setActiveSource("record");
    setUploadedFileName("");
    setAudioURL(null);
    setResult(null);
    audioChunks.current = [];

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const mimeType =
      MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
        ? "audio/webm;codecs=opus"
        : MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")
        ? "audio/ogg;codecs=opus"
        : "audio/webm";

    mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });

    mediaRecorderRef.current.ondataavailable = (e) => {
      audioChunks.current.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(audioChunks.current, { type: mimeType });
      const url = URL.createObjectURL(blob);

      setAudioURL("");
      setTimeout(() => setAudioURL(url), 0);
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setIsRecording(false);
  };

  
  const handleSubmit = async () => {
    if (!audioURL) return alert("Please record or upload audio!");
    if (!transcript.trim()) return alert("Please enter a transcript!");

    setLoading(true);

    const blob = await fetch(audioURL).then((r) => r.blob());
    const file = new File([blob], "audio.webm", { type: "audio/webm" });

    const formData = new FormData();
    formData.append("transcript", transcript);
    formData.append("audio", file);

    try {
      const res = await fetch("http://localhost:8000/score", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      setResult(data.prediction);
    } catch (err) {
      console.error("API Error:", err);
      alert("Error contacting server");
    }

    setLoading(false);
  };

  return (
    <div className="page-wrapper">
      <div className="recording-wrapper">
        <div className="recording-container">
          
          {/* LEFT SECTION */}
          <div className="record-left">

            <h2 className="record-title">Check pronounciation</h2>

            <textarea
              className="transcript-box"
              placeholder="Enter the sentence..."
              value={transcript}
              rows={1}
              onChange={(e) => {
                setTranscript(e.target.value);
                e.target.style.height = "auto";
                e.target.style.height = e.target.scrollHeight + "px";
              }}
            />

            {/* RECORD */}
            <div className="button-row">
              <button
                className={`action-btn ${
                  activeSource === "record" ? "active-purple" : "inactive-blue"
                }`}
                onClick={startRecording}
                disabled={isRecording}
                style={{ display: isRecording ? "none" : "block" }}
              >
                Start Recording
              </button>

              {isRecording && (
                <button className="action-btn stop-btn" onClick={stopRecording}>
                  Stop Recording
                </button>
              )}
            </div>

            {/* UPLOAD */}
            <div className="button-row">
              <label
                className={`action-btn upload-label ${
                  activeSource === "upload" ? "active-purple" : "inactive-blue"
                }`}
              >
                Choose Audio File
                <input
                  type="file"
                  accept="audio/mp3, audio/wav, audio/m4a, audio/mp4, audio/aac"
                  onChange={(e) => {
                    const file = e.target.files[0];
                    if (file) {
                      setActiveSource("upload");
                      setResult(null);
                      setUploadedFileName(file.name);

                      const newURL = URL.createObjectURL(file);
                      setAudioURL("");

                      setTimeout(() => setAudioURL(newURL), 0);
                    }
                  }}
                />
              </label>
            </div>

            <div className="format-info">Formats accepted: MP3, WAV</div>

            {/* AUDIO PLAYER */}
            {audioURL && (
              <div className="audio-player-box">
                {uploadedFileName && (
                  <div className="file-name-label">File: {uploadedFileName}</div>
                )}

                <audio key={audioURL} controls className="audio-player">
                  <source src={audioURL} />
                </audio>
              </div>
            )}

            {/* SUBMIT */}
            {audioURL && transcript.trim() && (
              <button className="submit-btn action-btn" onClick={handleSubmit}>
                {loading ? "Scoring..." : "Click to View Score"}
              </button>
            )}
          </div>

          {/* RIGHT SECTION */}
          <div className="record-right">

            {!result && (
              <img src="/1.jpg" className="record-image" alt="Practice Visual" />
            )}

            {result && (
              <div className="result-right-panel">

                <h3 className="accuracy-title">
                  Accuracy: {result.accuracy}%
                </h3>

                <button
                  className="see-correction-btn"
                  onClick={() => setShowCorrection(!showCorrection)}
                >
                  {showCorrection ? "Hide Correction" : "View Correction"}
                </button>

                {/* SCROLLABLE BLOCK */}
                {showCorrection && (
                  <div className="correction-scroll">
                    {result.correction.words.map((word, idx) => {
                      const expected = result.correction.expected[idx] || [];
                      const recorded = result.correction.recorded[idx] || [];

                      return (
                        <div key={idx} className="word-block">
                          <div className="word-title">{word}</div>

                          {/* PRONOUNCED */}
                          <div className="correction-line">
                            <span className="row-label">Pronounced:&nbsp;</span>

                            {recorded.map((ph, i) => {
                              const wrong = ph !== expected[i];
                              return (
                                <span
                                  key={i}
                                  className="phoneme"
                                  style={{
                                    color: wrong ? "red" : "green",
                                    fontWeight: "bold",
                                    marginRight: "6px",
                                  }}
                                >
                                  {ph}
                                </span>
                              );
                            })}
                          </div>

                          {/* CORRECT */}
                          <div className="correction-line">
                            <span className="row-label">Correct:&nbsp;</span>

                            {expected
                              .filter((p) => p !== "_")
                              .map((ph, i) => (
                                <span
                                  key={i}
                                  className="phoneme"
                                  style={{
                                    color: "green",
                                    fontWeight: "bold",
                                    marginRight: "6px",
                                  }}
                                >
                                  {ph}
                                </span>
                              ))}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
                {showCorrection && (
  <button
    className="try-more-btn"
    onClick={() => {
      setResult(null);
      setShowCorrection(false);
      setTranscript("");
      setAudioURL(null);
      setUploadedFileName("");
      setActiveSource(null);
    }}
  >
    Try More
  </button>
)}

              </div>
            )}

          </div>
        </div>
      </div>
    </div>
  );
}
