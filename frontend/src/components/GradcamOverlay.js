import React from "react";

export default function GradcamOverlay({ original, heatmap, overlayWithBox }) {
  return (
    <div className="gradcam-section">
      <h3>Grad-CAM Visualization</h3>
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "flex-start",
          gap: "16px",
          flexWrap: "wrap",
          marginTop: "12px",
        }}
      >
        {original && (
          <div>
            <h4>Original</h4>
            <img src={original} alt="Original X-ray" width="260" />
          </div>
        )}
        {heatmap && (
          <div>
            <h4>Grad-CAM Heatmap</h4>
            <img src={heatmap} alt="GradCAM Heatmap" width="260" />
          </div>
        )}
        {overlayWithBox && (
          <div>
            <h4>Overlay + Bounding Box</h4>
            <img src={overlayWithBox} alt="GradCAM Overlay" width="260" />
          </div>
        )}
      </div>
    </div>
  );
}
