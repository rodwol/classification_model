import time
import subprocess
from pathlib import Path

DATA_DIR = Path("processed")
FLAG_FILE = Path("retrain.flag")
CHECK_INTERVAL = 60  # seconds

def needs_retrain():
    # simple: retrain if flag file exists
    if FLAG_FILE.exists():
        return True
    # alternative: retrain if processed/X.npy modified since last model
    x_path = DATA_DIR / "X.npy"
    if not x_path.exists():
        return False
    # compare X.npy mtime with latest model mtime
    try:
        latest_model = max(Path("models").glob("run_*"), key=lambda p: p.stat().st_mtime)
        model_mtime = latest_model.stat().st_mtime
    except ValueError:
        return True
    return x_path.stat().st_mtime > model_mtime

def trigger_retrain():
    print("Triggering retrain...")
    # start training as a subprocess (non-blocking)
    subprocess.Popen(["python", "train.py", "--models_dir", "models"])
    # remove flag if present
    if FLAG_FILE.exists():
        FLAG_FILE.unlink()

if __name__ == "__main__":
    print("Starting retrain watcher (press Ctrl+C to stop)")
    while True:
        try:
            if needs_retrain():
                trigger_retrain()
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            break
