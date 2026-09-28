import ForceGraph2D from "react-force-graph-2d";
import { EMPTY_GRAPH, type GraphData } from "@/lib/sybil-types";

function riskToColor(score: number | undefined) {
  const safe = Math.max(0, Math.min(1, score ?? 0));
  const hue = 210 - safe * 200;
  return `hsl(${hue}, 82%, 58%)`;
}

export default function NetworkGraph({ data, mode = "risk", clusters = [] }: { data?: GraphData | null; mode?: "truth"|"risk"; clusters?: string[] }) {
  const safeData = data ?? EMPTY_GRAPH;
  const graphData = { nodes: safeData.nodes.map(n => ({...n})), links: safeData.edges.map(([source,target]) => ({source,target})) };
  return <ForceGraph2D
    graphData={graphData}
    width={640} height={320}
    backgroundColor="rgba(0,0,0,0)"
    nodeRelSize={5}
    nodeCanvasObject={(node, ctx, scale) => {
      const n = node as typeof safeData.nodes[number] & {x?:number;y?:number};
      if (n.x === undefined || n.y === undefined) return;
      const risk = n.score ?? 0;
      const fill = mode === "truth" ? (n.label ? "#ff5c6c" : "#43d39e") : riskToColor(risk);
      const r = clusters.includes(n.id) ? 8 : 5;
      if (clusters.includes(n.id)) { ctx.beginPath(); ctx.arc(n.x,n.y,r+5,0,Math.PI*2); ctx.strokeStyle="rgba(32,227,210,.45)"; ctx.lineWidth=2/scale; ctx.stroke(); }
      ctx.beginPath();
      if(n.node_type === "contract") ctx.rect(n.x-r,n.y-r,r*2,r*2);
      else if(n.node_type === "funder") { ctx.moveTo(n.x,n.y-r); ctx.lineTo(n.x+r,n.y); ctx.lineTo(n.x,n.y+r); ctx.lineTo(n.x-r,n.y); ctx.closePath(); }
      else ctx.arc(n.x,n.y,r,0,Math.PI*2);
      ctx.fillStyle=fill; ctx.fill();
    }}
    linkColor={(link) => {
      const sourceNode = link.source as { score?: number } | string;
      const targetNode = link.target as { score?: number } | string;
      const sourceScore = typeof sourceNode === "object" ? sourceNode.score ?? 0.5 : 0.5;
      const targetScore = typeof targetNode === "object" ? targetNode.score ?? 0.5 : 0.5;
      const score = (sourceScore + targetScore) / 2;
      const hue = 210 - score * 200;
      return `hsla(${hue}, 82%, 58%, 0.38)`;
    }}
    linkWidth={link => clusters.includes(String((link.source as {id?:string}).id ?? link.source)) ? 2.4 : .8}
    cooldownTicks={80}
  />;
}