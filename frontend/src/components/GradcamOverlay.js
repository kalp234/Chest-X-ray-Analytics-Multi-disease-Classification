import React, { useState } from "react";

export default function GradcamOverlay({ gradcam, original, bboxes = [] }) {
  const [opacity, setOpacity] = useState(1);

  return (
    <div className="gradcam-container">
      <h3>Grad-CAM Visualization</h3>

      <div className="overlay-wrapper" style={{ position: "relative", display: "inline-block" }}>
        <img
          src={original}
          alt="Original X-ray"
          className="overlay-image base-image"
          style={{ display: "block", maxWidth: "600px" }}
        />

        <img
          src={gradcam}
          alt="GradCAM Overlay"
          className="overlay-image heatmap-image"
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            opacity,
            maxWidth: "600px",
          }}
        />

        {/* Draw bounding boxes */}
        {bboxes.map((box, idx) => (
          <div
            key={idx}
            style={{
              position: "absolute",
              border: "2px solid red",
              left: `${box.x}px`,
              top: `${box.y}px`,
              width: `${box.w}px`,
              height: `${box.h}px`,
              pointerEvents: "none",
            }}
          >
            {/* Label tag */}
            <span
              style={{
                position: "absolute",
                top: "-20px",
                left: "0",
                backgroundColor: "rgba(255, 0, 0, 0.8)",
                color: "white",
                fontSize: "12px",
                padding: "2px 4px",
                borderRadius: "4px",
              }}
            >
              {box.label || "Region"}
            </span>
          </div>
        ))}
      </div>

      <div className="opacity-control">
        <label>Heatmap opacity:</label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
        />
      </div>
    </div>
  );
}
