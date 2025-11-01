import React from "react";

export default function PredictionTable({ data, onPick }) {
  return (
    <table className="pred-table">
      <thead>
        <tr><th>Disease</th><th>Confidence (%)</th></tr>
      </thead>
      <tbody>
        {data.map((d,i)=>(
          <tr key={i} onClick={() => onPick && onPick(i)}>
            <td>{d.label}</td>
            <td>{d.confidence}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
