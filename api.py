from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import numpy as np
import tensorflow as tf
from preprocessing import AudioPreprocessor
from pathlib import Path
import tempfile
import subprocess
import os

app = FastAPI()
MODEL_DIR = Path("models")
# by default load most recent model
def latest_model_path():
    runs = list(MODEL_DIR.glob("run_*"))
    if not runs:
        return None
    latest = max(runs, key=lambda p: p.stat().st_mtime)
    return latest / "ckpt.h5"

MODEL = None
def load_model():
    global MODEL
    model_path = latest_model_path()
    if model_path is None or not model_path.exists():
        raise RuntimeError("No model found. Train a model first.")
    MODEL = tf.keras.models.load_model(str(model_path))

@app.on_event("startup")
def startup():
    try:
        load_model()
    except Exception as e:
        print("Model not loaded on startup:", e)

preprocessor = AudioPreprocessor(
    target_sr=22050, n_mels=128, n_fft=2048, hop_length=512, duration=3.0, target_shape=(224,224)
)

PREDICTION_LABELS = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}
RETRAIN_TOKEN = os.environ.get("RETRAIN_TOKEN", "changeme")  # set env var in prod

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        load_model()
    # save upload temporarily
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    # preprocess using your AudioPreprocessor
    spec = preprocessor.generate_mel_spectrogram(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    if spec is None:
        raise HTTPException(status_code=400, detail="Could not process audio file.")
    # ensure shape (1, H, W, C)
    import numpy as np
    x = np.array(spec, dtype="float32")
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 2:
        x = np.stack([x]*3, axis=-1)
    x = x[np.newaxis, ...]
    preds = MODEL.predict(x)
    p = preds[0]
    top_idx = int(p.argmax())
    return {"label": PREDICTION_LABELS.get(top_idx, str(top_idx)), "probs": p.tolist()}

@app.post("/retrain")
def retrain(token: str):
    if token != RETRAIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
    # spawn train.py as a subprocess
    subprocess.Popen(["python", "train.py", "--models_dir", "models"])
    return {"status": "retraining_started"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
