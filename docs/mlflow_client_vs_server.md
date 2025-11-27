# MLflow Client Mode vs Server Mode

## Current Setup: Client Mode (Direct Database Connection)

### What Indicates Client Mode?

Your current setup uses **client mode** because:

1. **Tracking URI Format**: `sqlite:///mlflow.db`
   - Direct database connection string (not HTTP)
   - Code connects directly to SQLite database file
   - No intermediate server process

2. **No Server Process Required**:
   - Training scripts run and write directly to `mlflow.db`
   - No need to start `mlflow server` before training
   - Database file is accessed like any local file

3. **Code Location**: `src/utils/mlflow_init.py`
   ```python
   # Default tracking URI (line 69)
   tracking_uri = f"sqlite:///{db_path}"  # ← Client mode indicator
   ```

4. **How It Works**:
   ```
   Training Script → MLflow Client Library → SQLite Database (mlflow.db)
   ```
   - Direct file I/O operations
   - No network communication
   - Single-process access

### Current Architecture

```
┌─────────────────┐         ┌─────────────────┐
│ Training Script │         │   MLflow UI     │
│  (Python code)  │         │  (Web Browser)  │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ mlflow.log_*() calls      │ HTTP requests
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ MLflow Client   │         │ MLflow Server   │
│  (Library)      │         │  (mlflow ui)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ Direct SQLite connection  │ Direct SQLite connection
         │                           │
         └───────────┬───────────────┘
                     │
                     ▼
         ┌─────────────────┐
         │  mlflow.db      │
         │  (SQLite file)  │
         └─────────────────┘
```

**Key Point**: Both training scripts and the MLflow UI can access the same `mlflow.db` file simultaneously:
- **Training scripts**: Write directly to database (client mode)
- **MLflow UI**: Reads from same database (via `mlflow ui` command)
- **No conflict**: SQLite handles concurrent read/write access
- **No code changes needed**: Your training scripts continue using `sqlite:///mlflow.db`

---

## Server Mode (Centralized Service)

### What Would Indicate Server Mode?

1. **Tracking URI Format**: `http://localhost:5000` or `http://mlflow-server:5000`
   - HTTP endpoint (not direct database)
   - Code connects to MLflow server via REST API
   - Separate server process manages the database

2. **Server Process Required**:
   - Must start `mlflow server` before training
   - Server runs as separate process/service
   - Multiple clients can connect to same server

3. **How It Works**:
   ```
   Training Script → MLflow Client Library → HTTP REST API → MLflow Server → Database
   ```
   - Network communication (HTTP requests)
   - Server handles concurrent access
   - Centralized metadata and artifact storage

### Server Mode Architecture

```
┌─────────────────┐         ┌─────────────────┐
│ Training Script │         │ Training Script │
│  (Client 1)     │         │  (Client 2)     │
└────────┬────────┘         └────────┬────────┘
         │                          │
         │ mlflow.log_*() calls     │ mlflow.log_*() calls
         │                          │
         └──────────┬───────────────┘
                    │
                    │ HTTP REST API
                    │
                    ▼
         ┌──────────────────────┐
         │   MLflow Server       │
         │  (http://host:5000)    │
         │  - REST API           │
         │  - Web UI             │
         │  - Request handling   │
         └──────────┬────────────┘
                    │
                    │ Database operations
                    │
                    ▼
         ┌──────────────────────┐
         │   Database            │
         │  (PostgreSQL/SQLite)  │
         └──────────────────────┘
```

---

## Do You Need to Change Model Code?

### **NO! The model code stays exactly the same.**

The beauty of MLflow's design is that **the client API is identical** regardless of whether you're using client mode or server mode. All your existing code will work without any changes.

### What Changes?

**Only the tracking URI configuration:**

#### Current (Client Mode):
```python
# In src/utils/mlflow_init.py
tracking_uri = f"sqlite:///{db_path}"  # Direct DB connection
mlflow.set_tracking_uri(tracking_uri)
```

#### After Switching (Server Mode):
```python
# Option 1: Environment variable (recommended)
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Option 2: In code
mlflow.set_tracking_uri("http://localhost:5000")
```

### All Your Existing Code Works Unchanged

Your training code uses standard MLflow APIs that work identically in both modes:

```python
# These all work the same in client or server mode:
mlflow.set_experiment("ppa-fraud-detection")
mlflow.start_run()
mlflow.log_params({...})
mlflow.log_metrics({...})
mlflow.log_artifact(...)
mlflow.pytorch.log_model(...)
mlflow.register_model(...)
mlflow.end_run()
```

The MLflow client library automatically:
- Detects the tracking URI format
- Uses direct DB connection for `sqlite:///` URIs
- Uses HTTP REST API for `http://` URIs
- Handles all the differences transparently

---

## Migration Path: Client → Server Mode

### Step 1: Start MLflow Server

```bash
# Start server with same database
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

Or use your existing command:
```bash
make mlflow-ui
# or
python src/cli.py mlflow-ui
```

### Step 2: Update Tracking URI

**Option A: Environment Variable (Recommended)**
```bash
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

**Option B: Update `init_mlflow()` default**
```python
# In src/utils/mlflow_init.py, change default:
if tracking_uri is None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
```

**Option C: Pass explicitly**
```python
init_mlflow("http://localhost:5000")
```

### Step 3: Run Training (No Code Changes!)

```bash
# All existing commands work unchanged
make train-baseline
make train-sage
make train-hgt
```

That's it! Your model code doesn't need any changes.

---

## When to Use Each Mode?

### Client Mode (Current Setup) ✅
**Best for:**
- Local development
- Single-user scenarios
- Quick experiments
- No need for centralized access

**Advantages:**
- Simple setup (no server to manage)
- Faster for local development
- No network overhead
- Works offline

**Limitations:**
- Single-process write access (SQLite limitation)
- Limited scalability for concurrent writes
- No built-in web UI (but you can start UI separately - see below)

**Using MLflow UI with Client Mode:**
You can still use the MLflow UI! The UI is just another client that reads from the same database:

```bash
# Terminal 1: Training (writes to mlflow.db)
make train-baseline

# Terminal 2: UI (reads from same mlflow.db)
make mlflow-ui
# Opens http://localhost:5000
```

Both access the same `mlflow.db` file - training writes, UI reads. This works perfectly for local development!

### Server Mode
**Best for:**
- Production environments
- Team collaboration
- Multiple concurrent training jobs
- Centralized monitoring and management
- Remote access

**Advantages:**
- Centralized access for team
- Concurrent access from multiple clients
- Built-in web UI always available
- Better for production deployments
- Can use production databases (PostgreSQL, MySQL)

**Requirements:**
- Server process must be running
- Network connectivity
- Slightly more setup complexity

---

## Summary

1. **Current Setup = Client Mode** because:
   - Tracking URI: `sqlite:///mlflow.db` (direct DB connection)
   - No server process needed
   - Direct file access

2. **To Switch to Server Mode**:
   - Start `mlflow server` process
   - Change tracking URI to `http://localhost:5000`
   - **No model code changes needed!**

3. **All MLflow APIs work identically** in both modes - the client library handles the differences automatically.

