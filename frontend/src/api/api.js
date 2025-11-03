import axios from "axios";
const BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export async function predictXray(file, top_k=null) {
  const fd = new FormData();
  fd.append("file", file);
  if (top_k) fd.append("top_k", top_k);
  const res = await axios.post(`${BASE}/predict/`, fd, { headers: { "Content-Type": "multipart/form-data" }});
  return res.data;
}

export async function explainXray(file, gradcam_class = null) {
  const fd = new FormData();
  fd.append("file", file);
  if (gradcam_class !== null) {
    fd.append("gradcam_class", gradcam_class);
  }
  const res = await axios.post(`${BASE}/explain/`, fd, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return res.data;
}
