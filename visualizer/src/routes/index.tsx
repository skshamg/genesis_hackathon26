import { ClientOnly, createFileRoute } from "@tanstack/react-router";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, BrainCircuit, ChevronDown, Clipboard, Crosshair, Database, FlaskConical, Network, Play, Radar, ShieldCheck, Terminal, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ApiError, api, demo } from "@/lib/sybil-api";
import { EMPTY_GRAPH, type GraphData, type ModelInfo, type RealTestResult, type ScanEvent, type ScanResults, type TrainConfig, type TrainStatus } from "@/lib/sybil-types";

const NetworkGraph = lazy(() => import("@/components/network-graph"));
type Tab = "train" | "inspect" | "scan";

export const Route = createFileRoute("/")({
  head: () => ({ meta: [
    { title: "SYBIL_SHIELD — Graph Anomaly Detection" },
    { name: "description", content: "Train, inspect, and operate a graph neural network for structural anomaly detection." },
    { property: "og:title", content: "SYBIL_SHIELD — Graph Anomaly Detection" },
    { property: "og:description", content: "A security operations console for graph-based account anomaly analysis." },
    { property: "og:type", content: "website" }, { name: "twitter:card", content: "summary_large_image" },
  ]}),
  component: SybilShield,
});

const tooltips = { contentStyle: { background: "#101619", border: "1px solid #253238", borderRadius: 3, color: "#dce7e9", fontSize: 12 } };
const field = "h-10 w-full rounded-sm border border-input bg-input/40 px-3 font-mono text-sm text-foreground outline-none transition focus:border-primary focus:ring-1 focus:ring-primary/30";
const nav = [{id:"train",label:"Train",icon:BrainCircuit},{id:"inspect",label:"Inspect Model",icon:Network},{id:"scan",label:"Live Scan",icon:Radar}] as const;

function SybilShield() {
  const [tab,setTab] = useState<Tab>("train");
  const [demoMode,setDemoMode] = useState(false);
  const [trainedAt,setTrainedAt] = useState<string>();
  return <main className="min-h-screen bg-background text-foreground">
    <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-4 lg:px-7">
        <div className="flex items-center gap-3"><div className="grid size-8 place-items-center border border-primary/60 bg-primary/10"><ShieldCheck className="size-4 text-primary"/></div><div><div className="font-mono text-sm font-bold tracking-widest">SYBIL_SHIELD</div><div className="text-[10px] uppercase tracking-[.22em] text-muted-foreground">Graph defense console</div></div></div>
        <nav className="absolute left-1/2 hidden -translate-x-1/2 items-center gap-1 md:flex">{nav.map(({id,label,icon:Icon})=><button key={id} onClick={()=>setTab(id)} className={`flex h-9 items-center gap-2 border-b-2 px-4 text-xs font-semibold transition ${tab===id?"border-primary text-primary":"border-transparent text-muted-foreground hover:text-foreground"}`}><Icon className="size-3.5"/>{label}</button>)}</nav>
        <div className="flex items-center gap-3"><span className="hidden font-mono text-[10px] text-muted-foreground sm:block">API / STANDBY</span><span className="size-2 rounded-full bg-success shadow-[0_0_10px_var(--success)]"/></div>
      </div>
      <div className="flex border-t border-border md:hidden">{nav.map(({id,label})=><button key={id} onClick={()=>setTab(id)} className={`h-10 flex-1 text-xs ${tab===id?"bg-primary/10 text-primary":"text-muted-foreground"}`}>{label}</button>)}</div>
      <div className="flex min-h-9 items-center justify-between border-t border-border bg-panel px-4 font-mono text-[10px] uppercase tracking-wider lg:px-7"><span className="flex items-center gap-2"><span className={`size-1.5 rounded-full ${trainedAt?"bg-primary":"bg-muted-foreground"}`}/>{trainedAt ? `Model trained at ${new Date(trainedAt).toLocaleString()}` : "No model trained yet"}</span><span className="text-muted-foreground">GCN / v1.4.2</span></div>
    </header>
    {demoMode && <div className="fixed bottom-5 right-5 z-50 flex items-center gap-2 border border-primary/40 bg-primary/10 px-3 py-2 font-mono text-[10px] uppercase tracking-widest text-primary backdrop-blur"><span className="size-1.5 animate-pulse rounded-full bg-primary"/>Demo mode</div>}
    <div className="mx-auto max-w-[1500px] p-4 lg:p-7">
      {tab==="train" && <TrainTab onDemo={()=>setDemoMode(true)} onTrained={setTrainedAt}/>} 
      {tab==="inspect" && <InspectTab onDemo={()=>setDemoMode(true)} onLoaded={setTrainedAt}/>} 
      {tab==="scan" && <ScanTab onDemo={()=>setDemoMode(true)}/>} 
    </div>
  </main>;
}

function SectionTitle({eyebrow,title,detail}:{eyebrow:string;title:string;detail:string}) { return <div className="mb-6 flex flex-col justify-between gap-2 border-b border-border pb-5 md:flex-row md:items-end"><div><div className="mb-2 font-mono text-[10px] uppercase tracking-[.2em] text-primary">// {eyebrow}</div><h1 className="text-2xl font-semibold">{title}</h1></div><p className="max-w-xl text-sm text-muted-foreground">{detail}</p></div> }
function Panel({children,className=""}:{children:React.ReactNode;className?:string}) { return <section className={`border border-border bg-card ${className}`}>{children}</section> }
function PanelHead({title,meta}:{title:string;meta?:string}) { return <div className="flex h-11 items-center justify-between border-b border-border px-4"><h2 className="text-xs font-semibold uppercase tracking-wider">{title}</h2>{meta&&<span className="font-mono text-[10px] text-muted-foreground">{meta}</span>}</div> }
function NumberField({label,value,onChange,step=1,min,max}:{label:string;value:number;onChange:(n:number)=>void;step?:number;min?:number;max?:number}) { return <label className="space-y-2"><span className="text-xs text-muted-foreground">{label}</span><input className={field} type="number" value={value} step={step} min={min} max={max} onChange={e=>onChange(Number(e.target.value))}/></label> }

function TrainTab({onDemo,onTrained}:{onDemo:()=>void;onTrained:(s:string)=>void}) {
  const [cfg,setCfg]=useState<TrainConfig>({num_graphs:50,min_nodes:100,max_nodes:500,sybil_fraction_min:.05,sybil_fraction_max:.3,epochs:20,learning_rate:.02});
  const [modelId,setModelId]=useState("");const [savedId,setSavedId]=useState<string>();const [error,setError]=useState<string>();
  const [advanced,setAdvanced]=useState(false); const [status,setStatus]=useState<TrainStatus | null>(null); const [running,setRunning]=useState(false); const [loss,setLoss]=useState<{epoch:number;loss:number}[]>([]);
  const [trainingGraphs,setTrainingGraphs]=useState<ModelInfo["training_graphs"]>([]); const [selectedTrainGraph,setSelectedTrainGraph]=useState<GraphData | null>(null);
  const set=(k:keyof TrainConfig,v:number)=>setCfg(c=>({...c,[k]:v}));
  const fetchTrainingGraphs=async(id?:string)=>{ if(!id) return; try { const m = await api.modelInfo(id); const graphs = m.training_graphs ?? []; setTrainingGraphs(graphs); if (graphs.length) setSelectedTrainGraph({nodes: graphs[0].nodes, edges: graphs[0].edges}); } catch { /* ignore for now */ } };
  const simulate=()=>{onDemo();setRunning(true);setLoss([]);let epoch=0;const timer=window.setInterval(()=>{epoch++;const s=demo.training(epoch);setStatus(s);setLoss(p=>[...p,{epoch:s.epoch,loss:s.loss as number}]);if(epoch>=cfg.epochs){clearInterval(timer);setRunning(false);onTrained(new Date().toISOString());}},180)};
  const start=async()=>{
    if(cfg.min_nodes>cfg.max_nodes||cfg.sybil_fraction_min>cfg.sybil_fraction_max)return;
    setError(undefined);setSavedId(undefined);setRunning(true);setLoss([]);setTrainingGraphs([]);setSelectedTrainGraph(null);
    const id=modelId.trim();
    try{const {job_id}=await api.startTraining({...cfg,...(id?{model_id:id}:{})});follow(job_id)}
    catch(e){if(e instanceof ApiError){setRunning(false);setError(e.message)}else simulate()}
  };
  // WebSocket first; if it errors or closes before the run finishes, fall back to polling the status endpoint.
  const follow=(jobId:string)=>{
    let finished=false,polling=false;
    const fallback=()=>{if(finished||polling)return;polling=true;poll(jobId)};
    try{
      const socket=api.socket("train",jobId);
      socket.onmessage=e=>{const s=JSON.parse(e.data) as TrainStatus;if(s.status!=="running")finished=true;consume(s)};
      socket.onerror=()=>{socket.close();fallback()};
      socket.onclose=()=>fallback();
    }catch{fallback()}
  };
  const consume=(s:TrainStatus)=>{
    setStatus(s);
    if(s.status==="running"&&s.loss!=null)setLoss(p=>[...p,{epoch:s.epoch,loss:s.loss as number}]);
    if(s.status==="completed"){setRunning(false);setSavedId(s.model_id);onTrained(new Date().toISOString());void fetchTrainingGraphs(s.model_id)}
    if(s.status==="error"){setRunning(false);setError(s.message??"Training failed")}
  };
  const poll=(id:string)=>{const timer=window.setInterval(async()=>{try{const s=await api.trainingStatus(id);consume(s);if(s.status!=="running")clearInterval(timer)}catch{clearInterval(timer);simulate()}},1000)};
  return <><SectionTitle eyebrow="Training pipeline" title="Train detection model" detail="Generate labeled synthetic transaction graphs and fit the graph convolutional network."/>
    <div className="grid gap-4 xl:grid-cols-[380px_1fr]">
      <Panel><PanelHead title="Training parameters" meta="CONFIG"/><div className="space-y-5 p-5"><label className="block space-y-2"><span className="text-xs text-muted-foreground">Model ID <span className="text-muted-foreground/60">(optional — the trained model is saved under this id)</span></span><input className={field} value={modelId} placeholder="e.g. baseline-v1 (blank = auto id)" maxLength={64} disabled={running} onChange={e=>setModelId(e.target.value)}/></label><NumberField label="Synthetic graphs" value={cfg.num_graphs} onChange={v=>set("num_graphs",v)} min={1}/><div className="grid grid-cols-2 gap-3"><NumberField label="Min nodes" value={cfg.min_nodes} onChange={v=>set("min_nodes",v)}/><NumberField label="Max nodes" value={cfg.max_nodes} onChange={v=>set("max_nodes",v)}/></div><div><div className="mb-2 flex justify-between text-xs text-muted-foreground"><span>Sybil fraction range</span><span className="font-mono text-primary">{cfg.sybil_fraction_min.toFixed(2)}—{cfg.sybil_fraction_max.toFixed(2)}</span></div><input className="accent-primary w-full" type="range" min="0" max="1" step=".01" value={cfg.sybil_fraction_max} onChange={e=>set("sybil_fraction_max",Number(e.target.value))}/></div><button onClick={()=>setAdvanced(v=>!v)} className="flex w-full items-center justify-between border-y border-border py-3 text-xs text-muted-foreground"><span>Advanced settings</span><ChevronDown className={`size-4 transition ${advanced?"rotate-180":""}`}/></button>{advanced&&<div className="grid grid-cols-2 gap-3"><NumberField label="Epochs" value={cfg.epochs} onChange={v=>set("epochs",v)}/><NumberField label="Learning rate" value={cfg.learning_rate} step={.01} onChange={v=>set("learning_rate",v)}/></div>}<Button className="w-full" disabled={running} onClick={start}><Play className="size-4"/>{running?"Training in progress":"Start training"}</Button>{error&&<div className="border border-negative/40 bg-negative/10 px-3 py-2 font-mono text-[11px] text-negative">{error}</div>}{savedId&&<div className="border border-primary/40 bg-primary/10 px-3 py-2 font-mono text-[11px] text-primary">Saved as model ID: <b>{savedId}</b></div>}</div></Panel>
      <div className="grid gap-4 lg:grid-cols-2"><Panel><PanelHead title="Loss telemetry" meta={running?"STREAMING":"READY"}/><div className="h-[290px] p-4">{loss.length?<ResponsiveContainer><AreaChart data={loss}><defs><linearGradient id="loss" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--primary)" stopOpacity={.35}/><stop offset="100%" stopColor="var(--primary)" stopOpacity={0}/></linearGradient></defs><CartesianGrid stroke="var(--border)" vertical={false}/><XAxis dataKey="epoch" stroke="var(--muted-foreground)" fontSize={10}/><YAxis stroke="var(--muted-foreground)" fontSize={10}/><Tooltip {...tooltips}/><Area type="monotone" dataKey="loss" stroke="var(--primary)" fill="url(#loss)" strokeWidth={2}/></AreaChart></ResponsiveContainer>:<Empty icon={Activity} label="Loss stream awaiting training run"/>}</div></Panel>
      <Panel><PanelHead title="Current synthetic graph" meta="TRUE LABELS SHOWN"/><div className="graph-frame h-[290px] overflow-hidden">{status && status.current_graph ? <Graph data={status.current_graph} mode="truth"/> : <Empty icon={Network} label="Graph generation offline"/>}</div></Panel>
      <Panel className="lg:col-span-2"><PanelHead title="Run telemetry" meta={status?.status.toUpperCase() ?? "IDLE"}/><div className="grid gap-px bg-border sm:grid-cols-4">{[["Graph",status?`${status.current_graph_index} / ${status.total_graphs}`:"—"],["Epoch",status?`${status.epoch} / ${status.total_epochs}`:"—"],["Loss",status?.loss != null ? status.loss.toFixed(4) : "—"],["Class split",status?.honest_count != null && status?.sybil_count != null ? `${status.honest_count} H / ${status.sybil_count} S` : "—"]].map(([k,v])=><div className="bg-card p-4" key={String(k)}><div className="mb-2 text-[10px] uppercase tracking-wider text-muted-foreground">{String(k)}</div><div className="font-mono text-lg text-primary">{String(v)}</div></div>)}</div></Panel></div>
    </div>
    {trainingGraphs.length > 0 && <Panel className="mt-4"><PanelHead title="Training graph catalog" meta={`${trainingGraphs.length} graphs`}/><div className="grid gap-4 p-4 xl:grid-cols-[280px_1fr]"><div className="space-y-2">{trainingGraphs.map((graph)=><button key={graph.graph_id} onClick={()=>setSelectedTrainGraph({nodes: graph.nodes, edges: graph.edges})} className={`w-full border p-3 text-left ${selectedTrainGraph && selectedTrainGraph.nodes.length === graph.nodes.length && selectedTrainGraph.edges.length === graph.edges.length ? "border-primary bg-primary/10" : "border-border bg-panel"}`}><div className="font-mono text-[10px] uppercase tracking-widest text-primary">{graph.label}</div><div className="mt-2 text-xs text-muted-foreground">{graph.nodes.length} nodes · {graph.edges.length} edges</div></button>)}</div><div className="h-[300px] overflow-hidden rounded border border-border bg-panel">{selectedTrainGraph ? <Graph data={selectedTrainGraph} mode="truth"/> : <Empty icon={Network} label="Select a training graph"/>}</div></div></Panel>}
  </>;
}

function InspectTab({onDemo,onLoaded}:{onDemo:()=>void;onLoaded:(s:string)=>void}) {
  const [model,setModel]=useState<ModelInfo>();const [modelId,setModelId]=useState("");const [loading,setLoading]=useState(true);const [error,setError]=useState<string>();
  const [test,setTest]=useState<RealTestResult>();const [testing,setTesting]=useState(false);const [testError,setTestError]=useState<string>();
  const load=async(id?:string):Promise<boolean>=>{
    setLoading(true);setError(undefined);setTest(undefined);setTestError(undefined);
    try{const m=await api.modelInfo(id);setModel(m);onLoaded(m.trained_at);return true}
    catch(e){if(e instanceof ApiError){setError(e.message);return false}setModel(demo.model);onLoaded(demo.model.trained_at);onDemo();return true}
    finally{setLoading(false)}
  };
  useEffect(()=>{void load()},[]);
  // Runs the real (Facebook benchmark) test with the model in the ID box; loads/shows that model first if it isn't the one on screen.
  const runTest=async()=>{
    const id=modelId.trim()||model?.model_id;
    setTest(undefined);setTestError(undefined);
    if(id&&id!==model?.model_id&&!(await load(id)))return;
    setTesting(true);
    try{const r=await api.realTest(id);if(r.status==="ok")setTest(r);else setTestError(r.message)}
    catch(e){setTestError(e instanceof Error?e.message:"Real test failed")}
    finally{setTesting(false)}
  };
  const controls=<Panel className="mb-4"><div className="flex flex-col gap-3 p-5 md:flex-row md:items-end"><label className="flex-1 space-y-2"><span className="text-xs text-muted-foreground">Model ID</span><input className={field} value={modelId} placeholder={model?.model_id??"latest"} onChange={e=>setModelId(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")void load(modelId)}}/></label><Button variant="outline" disabled={loading||testing} onClick={()=>void load(modelId)}><Database className="size-4"/>{loading?"Loading":"Load model"}</Button><Button disabled={loading||testing} onClick={()=>void runTest()}><FlaskConical className="size-4"/>{testing?"Running real test":"Run real test"}</Button></div>{(error||testError)&&<div className="mx-5 mb-5 border border-negative/40 bg-negative/10 px-3 py-2 font-mono text-[11px] text-negative">{error??testError}</div>}</Panel>;
  const realTest=(testing||test)&&<Panel className="mt-4"><PanelHead title={`Real test — model ${model?.model_id??""}`} meta={testing?"RUNNING":"FACEBOOK BENCHMARK"}/>{testing?<Empty icon={FlaskConical} label="Scoring real graph with this model"/>:test&&<div className="space-y-px bg-border"><div className="grid gap-px sm:grid-cols-4">{Object.entries(test.metrics).flatMap(([name,m])=>[[`${name} AUC`,m.auc],[`${name} accuracy`,m.accuracy]] as [string,number][]).map(([k,v])=><div className="bg-card p-5" key={k}><div className="text-[10px] uppercase tracking-widest text-muted-foreground">{k}</div><div className="mt-2 font-mono text-3xl text-primary">{(v*100).toFixed(1)}<span className="text-sm">%</span></div></div>)}</div><div className="grid gap-px sm:grid-cols-4">{([["Sybil caught",test.confusion.correctly_flagged_sybil],["Sybil missed",test.confusion.incorrectly_flagged_safe],["Safe cleared",test.confusion.correctly_flagged_safe],["Safe flagged",test.confusion.incorrectly_flagged_sybil]] as [string,number[]][]).map(([k,v])=><div className="bg-card p-4" key={k}><div className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div><div className="mt-1 font-mono text-xl">{v.length}</div></div>)}</div><div className="bg-card p-4 font-mono text-[10px] text-muted-foreground">{test.graph.node_count} nodes · {test.graph.edge_count} edges · top risk: {test.smell.suspicious_nodes.slice(0,5).map(n=>`${n.node} (${n.avg_score.toFixed(2)})`).join(", ")}</div></div>}</Panel>;
  if(!model)return <>{controls}{loading?<Empty icon={Database} label="Loading model artifact"/>:<Empty icon={Database} label="No model loaded"/>}</>;
  const metrics=Object.entries(model.metrics);
  return <><SectionTitle eyebrow="Model observability" title={`Inspect trained artifact${model.model_id?` — ${model.model_id}`:""}`} detail="Review the learned architecture, parameter fields, and held-out synthetic evaluation."/>{controls}
    <Panel className="mb-4"><PanelHead title="Network architecture" meta="FORWARD PASS →"/><div className="flex overflow-x-auto p-5">{model.architecture.layers.map((l,i)=><div key={l.name} className="flex min-w-0 items-center"><div className="min-w-40 border border-primary/35 bg-primary/5 p-4"><BrainCircuit className="mb-5 size-4 text-primary"/><div className="font-mono text-xs">{l.name}</div><div className="mt-1 font-mono text-[10px] text-muted-foreground">{l.input_dim} → {l.output_dim}</div></div>{i<model.architecture.layers.length-1&&<div className="w-12 border-t border-dashed border-primary/50"/>}</div>)}</div></Panel>
    <Panel className="mb-4"><PanelHead title="Neuron circuit" meta="WEIGHTED LINKS"/><div className="p-5"><NeuralNetworkDiagram model={model}/></div></Panel>
    <div className="mb-4 grid gap-px border border-border bg-border sm:grid-cols-4">{metrics.map(([k,v])=><div className="bg-card p-5" key={k}><div className="text-[10px] uppercase tracking-widest text-muted-foreground">{k}</div><div className="mt-2 font-mono text-3xl text-primary">{v==null?"—":(v*100).toFixed(1)}<span className="text-sm">%</span></div></div>)}<div className="col-span-full border-t border-border bg-panel px-5 py-2 text-[10px] text-muted-foreground">Measured on held-out synthetic test graphs</div></div>
    <div className="grid gap-4 xl:grid-cols-[1.3fr_.7fr]"><Panel><PanelHead title="Weight matrices" meta="MAGNITUDE / SIGN"/><div className="grid gap-5 p-5 md:grid-cols-3">{model.weights.map(w=><div key={w.layer_name}><div className="mb-3 font-mono text-[10px] text-muted-foreground">{w.layer_name}</div><div className="grid gap-1" style={{gridTemplateColumns:`repeat(${w.matrix[0]?.length ?? 1},1fr)`}}>{w.matrix.flat().map((v,i)=><span key={i} title={v.toFixed(3)} className={`aspect-square border border-border ${v>0?"bg-primary":"bg-negative"}`} style={{opacity:.15+Math.abs(v)*.85}}/>)}</div></div>)}</div></Panel>
      <Panel><PanelHead title="Confusion matrix" meta={`N = ${Object.values(model.confusion_matrix??{}).reduce((a,b)=>a+b,0)}`}/><div className="grid grid-cols-[auto_1fr_1fr] gap-px bg-border p-px text-center"><span className="bg-card p-3"/><span className="bg-panel p-3 text-[10px] text-muted-foreground">PRED +</span><span className="bg-panel p-3 text-[10px] text-muted-foreground">PRED −</span>{[["ACT +",model.confusion_matrix.true_positive,model.confusion_matrix.false_negative],["ACT −",model.confusion_matrix.false_positive,model.confusion_matrix.true_negative]].map(r=><div className="contents" key={r[0]}><span className="bg-panel p-4 text-[10px] text-muted-foreground">{r[0]}</span><span className="bg-card p-4 font-mono text-xl text-primary">{r[1]}</span><span className="bg-card p-4 font-mono text-xl">{r[2]}</span></div>)}</div></Panel>
      <Panel className="xl:col-span-2"><PanelHead title="ROC curve" meta={`AUC ${model.metrics.auc}`}/><div className="h-64 p-4"><ResponsiveContainer><LineChart data={model.roc_curve}><CartesianGrid stroke="var(--border)"/><XAxis dataKey="fpr" stroke="var(--muted-foreground)" fontSize={10}/><YAxis stroke="var(--muted-foreground)" fontSize={10}/><Tooltip {...tooltips}/><Line dataKey="fpr" stroke="var(--muted-foreground)" strokeDasharray="4 4" dot={false}/><Line type="monotone" dataKey="tpr" stroke="var(--primary)" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></div></Panel></div>{realTest}</>;
}

function ScanTab({onDemo}:{onDemo:()=>void}) {
  const examples=["0x43370146366cfea21302962ae3a558d72bbcc487","0xdfd5293d8e347dfe59e90efd55b2956a1343963d","0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"];
  const [address,setAddress]=useState(examples[0] ?? "");const [maxNodes,setMaxNodes]=useState(500);const [depth,setDepth]=useState(2);const [modelId,setModelId]=useState("latest");const [apiKey,setApiKey]=useState("");const [phase,setPhase]=useState<"idle"|"mapping"|"inference"|"done">("idle");const [graph,setGraph]=useState<GraphData>({nodes:[],edges:[]});const [progress,setProgress]=useState({wallets_found:0,edges_found:0,api_calls_used:0});const [results,setResults]=useState<ScanResults>();
  const handle=(event:ScanEvent)=>{if(event.type==="node")setGraph(g=>({...g,nodes:[...g.nodes,{id:event.id,node_type:event.node_type}]}));if(event.type==="edge")setGraph(g=>({...g,edges:[...g.edges,[event.source,event.target]]}));if(event.type==="progress")setProgress(event);};
  const simulate=()=>{onDemo();setPhase("mapping");setGraph({nodes:[],edges:[]});let i=0;const timer=window.setInterval(()=>{const event=demo.scanEvents[i++];if(!event){clearInterval(timer);setPhase("inference");window.setTimeout(()=>{setResults(demo.scanResults);setGraph(demo.scanResults.graph);setPhase("done")},1500);return}handle(event)},120)};
  const start=async()=>{if(!address)return;setResults(undefined);setPhase("mapping");setGraph({nodes:[],edges:[]});try{const {job_id}=await api.startScan({address,max_nodes:maxNodes,max_depth:depth,model_id:modelId.trim() || "latest",api_key:apiKey.trim() || undefined});let socket:WebSocket;try{socket=api.socket("scan",job_id);socket.onmessage=async e=>{const event=JSON.parse(e.data) as ScanEvent;handle(event);if(event.type==="done"){setPhase("inference");const start=Date.now();try{const r=await api.scanResults(job_id);await new Promise(x=>setTimeout(x,Math.max(0,1500-(Date.now()-start))));setResults(r);setGraph(r.graph);setPhase("done")}catch{simulate()}}};socket.onerror=()=>simulate()}catch{simulate()}}catch{simulate()}};
  const clusters=useMemo(()=>results?.clusters.slice().sort((a,b)=>b.risk_score-a.risk_score)??[],[results]);
  return <><SectionTitle eyebrow="Live intelligence" title="Trace coordinated activity" detail="Map funding relationships, score structural patterns, and prioritize anomalous clusters for review."/>
    <Panel className="mb-4"><div className="grid gap-4 p-5 lg:grid-cols-[1fr_150px_150px_180px_180px]"><label className="space-y-2 lg:col-span-2"><span className="text-xs text-muted-foreground">Wallet address to scan</span><input className={field} value={address} onChange={e=>setAddress(e.target.value)} /></label><label className="space-y-2"><span className="text-xs text-muted-foreground">Model ID</span><input className={field} value={modelId} onChange={e=>setModelId(e.target.value)} /></label><NumberField label="Max nodes" value={maxNodes} onChange={setMaxNodes}/><NumberField label="Funding depth" value={depth} onChange={setDepth}/><label className="space-y-2 lg:col-span-2"><span className="text-xs text-muted-foreground">Etherscan API key <span className="text-muted-foreground/60">(optional override)</span></span><input className={field} type="password" value={apiKey} onChange={e=>setApiKey(e.target.value)} placeholder="Optional Etherscan key" /></label><Button className="mt-auto lg:col-span-2" onClick={start} disabled={phase==="mapping"||phase==="inference"}><Crosshair className="size-4"/>Start scan</Button><div className="flex flex-wrap gap-2 lg:col-span-5">{examples.map((x,i)=><button key={x} onClick={()=>setAddress(x)} className="border border-border bg-panel px-3 py-1.5 font-mono text-[10px] text-muted-foreground hover:border-primary hover:text-primary">Example: Token {String.fromCharCode(65+i)}</button>)}</div></div></Panel>
    <div className="grid gap-4 xl:grid-cols-[1fr_360px]"><Panel><PanelHead title="Funding topology" meta={phase.toUpperCase()}/><div className="relative graph-frame h-[480px] overflow-hidden">{graph.nodes.length?<Graph data={graph} clusters={clusters.map(c=>c.funder)}/>:<Empty icon={Radar} label="Awaiting scan target"/>}<div className="absolute right-3 top-3 z-10 w-32 rounded border border-border bg-background/90 p-2"><div className="font-mono text-[9px] uppercase tracking-widest text-muted-foreground">Risk spectrum</div><div className="mt-2 h-2 rounded" style={{background:"linear-gradient(90deg,#3b82f6 0%,#67e8f9 35%,#fbbf24 65%,#ef4444 100%)"}}/><div className="mt-2 flex justify-between font-mono text-[9px] text-muted-foreground"><span>Safe</span><span>Sybil</span></div></div>{phase==="inference"&&<div className="absolute inset-0 grid place-items-center bg-background/80 backdrop-blur-sm"><div className="text-center"><Zap className="mx-auto mb-3 size-6 animate-pulse text-primary"/><div className="font-mono text-xs uppercase tracking-widest">Running GCN inference...</div></div></div>}<div className="absolute bottom-3 left-3 border border-border bg-background/90 px-3 py-2 font-mono text-[10px] text-muted-foreground">Wallets: {progress.wallets_found||graph.nodes.filter(n=>n.node_type==="wallet").length} · Edges: {progress.edges_found||graph.edges.length} · API calls: {progress.api_calls_used}</div></div></Panel>
      <Panel><PanelHead title="Priority clusters" meta={`${clusters.length} DETECTED`}/><div className="max-h-[480px] overflow-y-auto">{clusters.length?clusters.map((c,i)=><div className="border-b border-border p-4" key={c.funder}><div className="mb-3 flex items-center justify-between"><span className="font-mono text-[10px] text-muted-foreground">CLUSTER {String(i+1).padStart(2,"0")}</span><span className="border border-primary/35 bg-primary/10 px-2 py-1 font-mono text-xs text-primary">{Math.round(c.risk_score*100)} RISK</span></div><Address value={c.funder}/><div className="mt-3 grid grid-cols-2 gap-2 text-xs"><span className="text-muted-foreground">Wallets <b className="float-right font-mono text-foreground">{c.wallets.length}</b></span><span className="text-muted-foreground">Window <b className="float-right font-mono text-foreground">{Math.floor(c.time_window_seconds/60)}m {c.time_window_seconds%60}s</b></span></div><div className="mt-3 text-[11px] text-primary">Flagged as structurally anomalous</div></div>):<Empty icon={Terminal} label="No scored clusters"/>}</div></Panel></div>
    <footer className="mt-4 border-t border-border py-4 text-center text-[11px] text-muted-foreground">Flags are model output based on structural patterns, not confirmed fraud.</footer></>;
}

function NeuralNetworkDiagram({ model }:{ model: ModelInfo }) {
  const layers = model.architecture.layers.map((layer, idx) => ({
    ...layer,
    weight: model.weights[idx] ?? model.weights[model.weights.length - 1],
  }));
  const maxLayerSize = Math.max(...layers.map(layer => Math.min(layer.output_dim, 10)), 1);
  const xStep = 160;
  const left = 36;
  const top = 28;
  const height = Math.max(180, maxLayerSize * 18 + 30);
  const rendered = layers.map((layer, layerIndex) => {
    const count = Math.min(layer.output_dim, 10);
    const x = left + layerIndex * xStep;
    return Array.from({ length: count }, (_, idx) => {
      const y = top + (count === 1 ? height / 2 : idx * ((height - 60) / Math.max(1, count - 1))); 
      return { x, y, layerName: layer.name, idx };
    });
  });
  const connections = layers.flatMap((layer, layerIndex) => {
    if (layerIndex === layers.length - 1) return [];
    const nextLayer = layers[layerIndex + 1];
    const inputNodes = rendered[layerIndex];
    const outputNodes = rendered[layerIndex + 1];
    const sourceWeight = model.weights[layerIndex] ?? { matrix: [] };
    const matrix = sourceWeight.matrix;
    return inputNodes.flatMap((source, i) => outputNodes.map((target, j) => {
      const value = matrix?.[i % Math.max(1, matrix.length)]?.[j % Math.max(1, (matrix[0]?.length ?? 1))] ?? 0;
      const stroke = value >= 0 ? "#38bdf8" : "#f87171";
      const opacity = Math.min(1, 0.2 + Math.abs(value) * 0.9);
      return <line key={`${layerIndex}-${i}-${j}`} x1={source.x + 18} y1={source.y} x2={target.x - 18} y2={target.y} stroke={stroke} strokeOpacity={opacity} strokeWidth={Math.max(0.7, Math.abs(value) * 1.6)}><title>{value.toFixed(3)}</title></line>;
    }));
  });
  return <svg viewBox={`0 0 ${left + (layers.length - 1) * xStep + 120} ${height}`} className="w-full h-[220px]">
    <defs>
      <linearGradient id="neuronLegend" x1="0" x2="1">
        <stop offset="0%" stopColor="#38bdf8"/>
        <stop offset="50%" stopColor="#7dd3fc"/>
        <stop offset="100%" stopColor="#f87171"/>
      </linearGradient>
    </defs>
    {connections}
    {rendered.flatMap((nodes, layerIndex) => nodes.map((node, idx) => <g key={`${layerIndex}-${idx}`}>
      <circle cx={node.x} cy={node.y} r={12} fill={layerIndex === layers.length - 1 ? "#f59e0b" : "#1e293b"} stroke="rgba(96,165,250,.9)" strokeWidth={1.5} />
      <text x={node.x} y={node.y + 3} textAnchor="middle" fontSize="8" fill="#e2e8f0">{idx + 1}</text>
    </g>))}
    <g transform={`translate(${left}, ${height - 18})`}>
      <rect x={0} y={0} width={120} height={10} rx={5} fill="url(#neuronLegend)" />
      <text x={0} y={-4} fontSize="9" fill="#94a3b8">weight</text>
    </g>
  </svg>;
}

function Graph({ data, ...props }:{data?: GraphData | null; mode?:"truth"|"risk"; clusters?:string[]}) {
  const safeData = data ?? EMPTY_GRAPH;
  return <ClientOnly fallback={<Empty icon={Network} label="Initializing graph renderer"/>}><Suspense fallback={<Empty icon={Network} label="Initializing graph renderer"/>}><NetworkGraph data={safeData} {...props}/></Suspense></ClientOnly>
}
function Address({value}:{value:string}) { const short=`${value.slice(0,6)}...${value.slice(-4)}`; return <button onClick={()=>navigator.clipboard.writeText(value)} title="Copy full address" className="flex items-center gap-2 font-mono text-xs text-foreground hover:text-primary">{short}<Clipboard className="size-3"/></button> }
function Empty({icon:Icon,label}:{icon:typeof Activity;label:string}) { return <div className="grid h-full min-h-40 place-items-center p-8 text-center"><div><Icon className="mx-auto mb-3 size-5 text-muted-foreground"/><div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">{label}</div></div></div> }