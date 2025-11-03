import React, { useState } from "react";
import { predictXray, explainXray } from "../api/api";
import PredictionTable from "./PredictionTable";
import GradcamOverlay from "./GradcamOverlay";
import MetricsPlot from "./MetricsPlot";

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [gradcamImages, setGradcamImages] = useState(null); // {original, heatmap, overlay}
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);

  // -------------------------------
  // Handle file input change
  // -------------------------------
  const onFileChange = (e) => {
    setFile(e.target.files[0]);
    setPredictions([]);
    setGradcamImages(null);
    setMetrics(null);
  };

  // -------------------------------
  // Predict button action
  // -------------------------------
  const submitPredict = async () => {
    if (!file) return alert("Choose an image first!");
    setLoading(true);
    try {
      const data = await predictXray(file);
      console.log("✅ Predict API Response:", data);
      setPredictions(data.predictions || []);
    } catch (err) {
      console.error("❌ Prediction Error:", err);
      alert("Prediction failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------
  // Explain button action (Grad-CAM)
  // -------------------------------
  const submitExplain = async () => {
    if (!file) return alert("Choose an image first!");
    setLoading(true);
    try {
      const data = await explainXray(file, null); // Let backend pick top predicted class
      console.log("✅ Explain API Response:", data);

      setGradcamImages({
        original: data.original_b64,
        heatmap: data.heatmap_b64,
        overlayWithBox: data.bbox_b64 || data.overlay_b64,
      });

      setPredictions(data.predictions || []);
      setMetrics(data.metrics || null);
    } catch (err) {
      console.error("❌ Explain Error:", err);
      alert("Explain failed. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  // -------------------------------
  // Render UI
  // -------------------------------
  return (
    <div className="uploader" style={{ textAlign: "center", marginTop: "20px" }}>

      <input
        type="file"
        onChange={onFileChange}
        accept="image/*"
        style={{ marginTop: "15px" }}
      />

      <div className="buttons" style={{ marginTop: "16px", display: "flex", justifyContent: "center", gap: "12px" }}>
        <button onClick={submitPredict} disabled={loading}>
          🔍 Predict
        </button>
        <button onClick={submitExplain} disabled={loading}>
          🔬 Explain (Grad-CAM)
        </button>
      </div>

      {loading && <p style={{ marginTop: "15px" }}>Processing... ⏳</p>}

      {predictions && predictions.length > 0 && (
        <div style={{ marginTop: "25px" }}>
          <PredictionTable data={predictions} />
        </div>
      )}

      {gradcamImages && (
        <div style={{ marginTop: "25px" }}>
          <GradcamOverlay
            original={gradcamImages.original}
            heatmap={gradcamImages.heatmap}
            overlayWithBox={gradcamImages.overlayWithBox}
          />
        </div>
      )}

      {metrics && (
        <div style={{ marginTop: "25px" }}>
          <MetricsPlot metrics={metrics} />
        </div>
      )}
    </div>
  );
}
