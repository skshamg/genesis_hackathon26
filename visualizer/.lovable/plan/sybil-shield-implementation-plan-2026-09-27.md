# SYBIL_SHIELD implementation plan

## What I’ll build
- A single-screen, three-tab security operations dashboard: Train, Inspect Model, and Live Scan.
- A persistent model-status strip, compact cyan-accented navigation, and a Demo Mode indicator.
- A robust typed API layer using `VITE_API_BASE_URL`, with WebSocket streaming, polling fallback, and realistic canned responses when the backend is unavailable.
- Training controls, live progress, loss chart, force-directed graph, model architecture and weight inspection, metrics, confusion matrix, ROC curve, and incremental scan visualization.
- Responsive layouts optimized for laptop recordings, with compact mobile behavior, copyable truncated addresses, and required safety language.

## Visual direction
- Near-black operations console with charcoal surfaces, fine neutral borders, cyan reserved for focus, active state, and model-risk emphasis.
- Clean sans-serif interface text with monospace identifiers and numeric telemetry.
- Dense, restrained dashboard composition with subtle grid texture, status lights, and purposeful motion only.

## Technical details
- Keep `/` as the single content route with internal tabs.
- Add `react-force-graph-2d`; use existing Recharts and Lucide packages.
- Separate API contracts, demo data, API transport, shared controls, and dashboard views.
- Convert HTTP URLs safely to WebSocket URLs and keep the localhost default only in the API module.
- Validate form ranges, contain all request failures, and avoid blank states.
- Verify the generated page at laptop and mobile widths and check the latest build diagnostics.
