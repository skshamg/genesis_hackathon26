import { demoModel, demoScanEvents, demoScanResults, demoTraining } from "./demo-data";
import type { ModelInfo, RealTestResult, ScanEvent, ScanResults, TrainConfig, TrainStatus } from "./sybil-types";

export const API_BASE_URL = (import.meta.env["VITE_API_BASE_URL"] || "http://localhost:8000").replace(/\/$/, "");
const wsBase = API_BASE_URL.replace(/^http/, "ws");

/** The server answered but rejected the request (bad id, duplicate id, unknown model...). Not the same as "API unreachable". */
export class ApiError extends Error { constructor(public status: number, message: string) { super(message); } }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, { ...init, headers: { "Content-Type": "application/json", ...init?.headers } });
  } catch { throw new Error("API unavailable"); }
  if (!response.ok) {
    let detail = `API ${response.status}`;
    try { const body = await response.json(); const d = body?.detail; detail = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: { msg?: string }) => x.msg).join("; ") : detail; } catch { /* keep default */ }
    throw new ApiError(response.status, detail);
  }
  return await response.json() as T;
}
export const api = {
  startTraining: (config: TrainConfig) => request<{job_id:string}>("/api/train/start", { method: "POST", body: JSON.stringify(config) }),
  trainingStatus: (id: string) => request<TrainStatus>(`/api/train/status/${id}`),
  modelInfo: (modelId?: string) => request<ModelInfo>(`/api/model/info?model_id=${encodeURIComponent(modelId?.trim() || "latest")}`),
  realTest: (modelId?: string) => request<RealTestResult>("/api/facebooktest", { method: "POST", body: JSON.stringify({ keyword: "go", model_id: modelId?.trim() || "latest" }) }),
  startScan: (body: {address:string;max_nodes:number;max_depth:number;model_type?:string;model_id?:string;api_key?:string}) => request<{job_id:string}>("/api/scan/start", { method:"POST", body:JSON.stringify(body) }),
  scanStatus: (id: string) => request<ScanEvent>(`/api/scan/status/${id}`),
  scanResults: (id: string) => request<ScanResults>(`/api/scan/results/${id}`),
  socket: (kind: "train"|"scan", id: string) => new WebSocket(`${wsBase}/ws/${kind}/${id}`),
};
export const demo = { model: demoModel, training: demoTraining, scanEvents: demoScanEvents, scanResults: demoScanResults };