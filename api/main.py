from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import subprocess
from pathlib import Path

app = FastAPI(title="Manufacturing Analytics API", version="0.2")

# allow local UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parents[1]

def run_job(script_name: str):
    """Run a job script from jobs/ folder."""
    path = BASE_DIR / "jobs" / f"{script_name}.py"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Job script not found: {path}")
    res = subprocess.run(["python", str(path)], capture_output=True, text=True, cwd=BASE_DIR)
    return {
        "script": script_name,
        "returncode": res.returncode,
        "stdout": res.stdout[-1000:],
        "stderr": res.stderr[-1000:]
    }

@app.get("/health")
def health():
    return {"status": "ok", "base_dir": str(BASE_DIR)}

@app.post("/run/bronze")
def run_bronze(): return run_job("ingest_parts"), run_job("ingest_production"), run_job("ingest_maintenance")

@app.post("/run/silver")
def run_silver(): return run_job("silver_transform")

@app.post("/run/gold")
def run_gold(): return run_job("gold_features")

@app.post("/run/train")
def run_train(): return run_job("ml_train")

@app.post("/run/score")
def run_score(): return run_job("ml_score")
