import React, { useState } from "react";
import { predictXray, explainXray } from "../api/api";
import PredictionTable from "./PredictionTable";
import GradcamOverlay from "./GradcamOverlay";
import MetricsPlot from "./MetricsPlot";

export default function ImageUploader() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [gradcamB64, setGradcamB64] = useState(null);
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);

  const onFileChange = (e) => {
    setFile(e.target.files[0]);
    setPredictions([]);
    setGradcamB64(null);
    setMetrics(null);
  };

  // ----------------------------
  // Run Ensemble Prediction
  // ----------------------------
  const submitPredict = async () => {
    if (!file) return alert("Choose an image first");
    setLoading(true);
    try {
      const data = await predictXray(file);
      console.log("Prediction Response:", data);
      setPredictions(data.predictions || []);
    } catch (err) {
      console.error(err);
      alert("Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  // ----------------------------
  // Run Grad-CAM Explainability
  // ----------------------------
  const submitExplain = async () => {
    if (!file) return alert("Choose an image first");
    if (predictions.length === 0) {
      alert("Run Predict first to choose the most confident class.");
      return;
    }

    setLoading(true);
    try {
      // Automatically use top predicted class index (most confident)
      const topClassIndex = 0; 
      const data = await explainXray(file, topClassIndex);

      console.log("Explain Response:", data);
      setGradcamB64(data.gradcam_b64 || null);
      setMetrics(data.metrics || null);
      setPredictions(data.predictions || predictions);
    } catch (err) {
      console.error(err);
      alert("Grad-CAM generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="uploader">
      <input type="file" onChange={onFileChange} accept="image/*" />

      <div className="buttons">
        <button onClick={submitPredict} disabled={loading}>
          Predict
        </button>
        <button onClick={submitExplain} disabled={loading}>
          Explain (Grad-CAM)
        </button>
      </div>

      {predictions.length > 0 && (
        <PredictionTable data={predictions} />
      )}

      {gradcamB64 && file && (
        <GradcamOverlay
          gradcam={gradcamB64}
          original={URL.createObjectURL(file)} // show uploaded X-ray
        />
      )}

      {metrics && <MetricsPlot metrics={metrics} />}

      {loading && <p>Processing...</p>}
    </div>
  );
}
