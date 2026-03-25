# Node-RED Fraud Detection Dashboard

Real-time operational dashboard for the PPA fraud detection pipeline.
Visualises the complete system in one place: live event scoring, admin review with SHAP explanations, parquet storage, retraining loop, and concept drift monitoring.

## Prerequisites

- **Node.js ≥ 18** — [nodejs.org](https://nodejs.org)
- **Python API running** on port 8000 — see `make api` in the project root

## Quick Start

```bash
# 1. Start the fraud detection API (one terminal)
make api

# 2. Start Node-RED (another terminal)
make nodered

# 3. Open the dashboard
open http://localhost:1880/dashboard
```

Or use the convenience target that prints both instructions:

```bash
make demo
```

## Manual Setup

```bash
cd nodered
npm install
npm start
```

- **Dashboard**: http://localhost:1880/dashboard
- **Flow editor**: http://localhost:1880/

## Dashboard Tabs

### Tab 1 — Pipeline Simulator
Fires a synthetic fraud event every 5 seconds through the full pipeline.

- **Fraud Probability gauge** — live score from the XGBoost model (0–1, thresholds at 0.3 / 0.7)
- **Score Timeline chart** — rolling 50-event history
- **Decision counters** — running totals for Approved / Review / Declined
- **FRAUD ALERT notification** — pops up whenever a HIGH risk event is detected

### Tab 2 — Admin Review
Human-in-the-loop review for MEDIUM risk events.

- **Pending Review table** — polls `/admin/queue` every 3 seconds
- **SHAP waterfall** — enter an `insertion_id` and click "Get SHAP" to fetch a real SHAP explanation from the API
- **Risk factor bars** — top contributing signals rendered as a colour-coded bar chart
- **Approve / Decline form** — submit a decision with optional notes; the event is written to parquet and removed from the queue

### Tab 3 — Storage & Retraining
Parquet storage overview and retraining controls.

- **Decision counts** — approved, declined, and rows pending retraining
- **Decision distribution bar chart** — updated every 10 seconds
- **Trigger Retraining button** — calls `/admin/retrain/trigger`; requires ≥ 50 labelled rows in `artifacts/decisions/retrain/pending.parquet`
- **Model Registry status** — model URI, MLflow run ID, and alias (polls `/model-info` every 15 s)
- **Service health** — events in store and graph node count (polls `/health`)

### Tab 4 — Drift Monitor
Concept drift tracking based on the live prediction distribution.

- **PSI gauge** — Population Stability Index (green < 0.1, yellow < 0.25, red ≥ 0.25)
- **Drift status label** — STABLE / MODERATE SHIFT / SIGNIFICANT DRIFT
- **Drifted features list** — highlighted in red when drift is detected
- **Score distribution histogram** — bucketed fraud probabilities (0–0.2, 0.2–0.4, …)
- **Reset Reference button** — resets the reference distribution to the current window; use after a scheduled retraining cycle

## Architecture

```
Node-RED (port 1880)                  FastAPI (port 8000)
─────────────────────                 ──────────────────────────────────
Tab 1: inject every 5s  ──────────►  POST /admin/simulate
Tab 2: queue poll        ──────────►  GET  /admin/queue
Tab 2: SHAP request      ──────────►  POST /admin/shap/{id}
Tab 2: decision submit   ──────────►  POST /admin/decision/{id}
Tab 3: storage poll      ──────────►  GET  /admin/storage/stats
Tab 3: retrain button    ──────────►  POST /admin/retrain/trigger
Tab 3: model info poll   ──────────►  GET  /model-info
Tab 4: drift poll        ──────────►  GET  /admin/drift/status
Tab 4: drift reset       ──────────►  POST /admin/drift/reset
```

All ML computation (feature engineering, XGBoost inference, SHAP, drift detection) happens in the Python API. Node-RED handles only visualisation and HTTP orchestration.

## Configuration

The dashboard expects the API at `http://localhost:8000`. To point it at a different host, open the flow editor and update the URL in any HTTP request node (use Ctrl+A → search "localhost:8000").

## Flows

`flows.json` contains all four tabs and can be imported into any Node-RED instance via **Menu → Import → Clipboard**. The `node-red-dashboard` package is required (installed automatically by `npm install`).
