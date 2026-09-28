export type GraphNode = { id: string; label?: 0 | 1; node_type?: "wallet" | "funder" | "contract"; score?: number };
export type GraphData = { nodes: GraphNode[]; edges: [string, string][] };
export type TrainingGraph = { graph_id: string; label: string; nodes: GraphNode[]; edges: [string, string][] };
export const EMPTY_GRAPH: GraphData = { nodes: [], edges: [] };

export type TrainConfig = {
  num_graphs: number; min_nodes: number; max_nodes: number;
  sybil_fraction_min: number; sybil_fraction_max: number;
  epochs: number; learning_rate: number;
  model_id?: string;
};
export type TrainStatus = {
  status: "running" | "completed" | "error"; current_graph_index: number;
  total_graphs: number; epoch: number; total_epochs: number; loss: number | null;
  honest_count: number | null; sybil_count: number | null; current_graph: GraphData | null;
  model_id?: string; message?: string; training_graphs?: TrainingGraph[];
};
export type ModelInfo = {
  model_id?: string;
  trained_at: string;
  architecture: { layers: { name: string; input_dim: number; output_dim: number }[] };
  weights: { layer_name: string; matrix: number[][] }[];
  metrics: { auc: number; accuracy: number; precision: number; recall: number };
  confusion_matrix: { true_positive: number; false_positive: number; true_negative: number; false_negative: number };
  roc_curve: { fpr: number; tpr: number }[];
  training_graphs?: TrainingGraph[];
};
export type ScanEvent =
  | { type: "node"; id: string; node_type: "wallet" | "funder" | "contract" }
  | { type: "edge"; source: string; target: string }
  | { type: "progress"; wallets_found: number; edges_found: number; api_calls_used: number }
  | { type: "done" };
export type ScanResults = {
  graph: GraphData;
  clusters: { funder: string; wallets: string[]; time_window_seconds: number; risk_score: number }[];
};
export type RealTestResult = {
  status: "ok" | "waiting" | "missing_dataset" | "error";
  message: string; model_id?: string;
  metrics: Record<string, { auc: number; accuracy: number }>;
  confusion: { correctly_flagged_sybil: number[]; incorrectly_flagged_sybil: number[]; correctly_flagged_safe: number[]; incorrectly_flagged_safe: number[] };
  graph: { node_count: number; edge_count: number };
  smell: { suspicious_nodes: { node: number; avg_score: number; true_label: number }[] };
};