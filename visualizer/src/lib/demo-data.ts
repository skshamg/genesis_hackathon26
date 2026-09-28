import type { GraphData, ModelInfo, ScanEvent, ScanResults, TrainStatus } from "./sybil-types";

const ids = ["0x71c84a93b0fd7aa4412c", "0xd03e19b68a5f9c24f1b8", "0x9a2c7ed11845fa112dc0", "0x65ef8d2180c9ae619bb3", "0xb4f7618a9de0c8392f11", "0x14ca9107b61aef56d208", "0xc810ef4d17aa9b28240e", "0x0f72a1519f4c88d372a5", "0xa74d8fb619e177058df2"] as const;

export const demoGraph: GraphData = {
  nodes: ids.map((id, i) => ({ id, label: i > 5 ? 1 : 0 })),
  edges: [[ids[0],ids[1]],[ids[0],ids[2]],[ids[1],ids[3]],[ids[2],ids[4]],[ids[3],ids[5]],[ids[4],ids[6]],[ids[6],ids[7]],[ids[6],ids[8]],[ids[7],ids[8]]],
};

export const demoTraining = (epoch = 12): TrainStatus => ({
  status: epoch >= 20 ? "completed" : "running", current_graph_index: Math.min(32, epoch + 20), total_graphs: 50,
  epoch, total_epochs: 20, loss: Number((0.61 * Math.exp(-epoch / 5) + 0.035).toFixed(4)), honest_count: 76,
  sybil_count: 18, current_graph: demoGraph,
});

export const demoModel: ModelInfo = {
  trained_at: "2026-09-27T08:41:00.000Z",
  architecture: { layers: [
    { name: "GraphConv_01", input_dim: 16, output_dim: 32 }, { name: "GraphConv_02", input_dim: 32, output_dim: 16 },
    { name: "Dense", input_dim: 16, output_dim: 2 },
  ]},
  weights: [
    { layer_name: "GraphConv_01", matrix: [[-.8,.2,.5,-.1],[.6,-.4,.1,.9],[-.2,.7,-.6,.3],[.4,.1,-.9,.6]] },
    { layer_name: "GraphConv_02", matrix: [[.7,-.2,.4],[-.5,.8,-.1],[.2,-.7,.9],[-.4,.3,.6]] },
    { layer_name: "Dense", matrix: [[-.7,.8],[.6,-.4],[-.2,.5],[.9,-.6]] },
  ],
  metrics: { auc: .942, accuracy: .918, precision: .887, recall: .901 },
  confusion_matrix: { true_positive: 184, false_positive: 21, true_negative: 736, false_negative: 18 },
  roc_curve: [{fpr:0,tpr:0},{fpr:.03,tpr:.38},{fpr:.07,tpr:.67},{fpr:.13,tpr:.84},{fpr:.25,tpr:.94},{fpr:.48,tpr:.98},{fpr:1,tpr:1}],
};

const scores = [0.08,.92,.21,.86,.72,.14,.78,.81,.67] as const;
const scanNodes = ids.map((id, i) => ({ id, node_type: (i === 0 ? "contract" : i === 1 || i === 6 ? "funder" : "wallet") as "contract"|"funder"|"wallet", score: scores[i] ?? 0 }));
export const demoScanResults: ScanResults = {
  graph: { nodes: scanNodes, edges: demoGraph.edges },
  clusters: [
    { funder: ids[1], wallets: [ids[3],ids[5]], time_window_seconds: 252, risk_score: .92 },
    { funder: ids[6], wallets: [ids[7],ids[8]], time_window_seconds: 611, risk_score: .78 },
  ],
};
export const demoScanEvents: ScanEvent[] = [
  ...scanNodes.map(({id,node_type}) => ({ type: "node" as const, id, node_type })),
  ...demoGraph.edges.map(([source,target]) => ({ type: "edge" as const, source, target })),
  { type: "progress", wallets_found: 6, edges_found: 9, api_calls_used: 14 }, { type: "done" },
];