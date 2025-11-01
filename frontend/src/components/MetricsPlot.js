import React from "react";

export default function MetricsPlot({ metrics }) {
  return (
    <div>
      <h4>Explainability Metrics</h4>
      <pre style={{textAlign: "left"}}>{JSON.stringify(metrics, null, 2)}</pre>
    </div>
  );
}
