import axios from "axios";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

const api = axios.create({ baseURL: API_BASE });

export const uploadFile = (file) => {
  const form = new FormData();
  form.append("file", file);
  return api.post("/api/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};

export const loadSample = () => api.post("/api/load-sample");

export const getDashboard = (sessionId) =>
  api.get(`/api/dashboard/${sessionId}`);

export const getMLModels = (sessionId) =>
  api.get(`/api/ml-models/${sessionId}`);

export const getAtRisk = (sessionId, threshold) =>
  api.get(`/api/at-risk/${sessionId}`, { params: { threshold } });

export const getStudents = (sessionId) =>
  api.get(`/api/students/${sessionId}`);

export const getCourses = (sessionId) =>
  api.get(`/api/courses/${sessionId}`);

export const getRecommendations = (sessionId, studentId) =>
  api.get(`/api/student/${sessionId}/${studentId}/recommendations`);

export const getReport = (sessionId, reportType, targetId) =>
  api.get(`/api/report/${sessionId}`, {
    params: { report_type: reportType, target_id: targetId },
  });

export const getExportUrl = (sessionId) =>
  `${API_BASE}/api/export/${sessionId}`;
