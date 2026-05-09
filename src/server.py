import os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from neuralnet2 import Network


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.json")
STATIC_INDEX = os.path.join(os.path.dirname(__file__), "static", "index.html")

app = FastAPI()
net: Network | None = None


@app.on_event("startup")
def load_model():
    global net
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model.json not found at {MODEL_PATH} — run main.py first")
    net = Network.load(MODEL_PATH)
    net.set_training(False)


class PredictRequest(BaseModel):
    pixels: list[list[float]]  # 28x28, values in [0, 1]


class LayerActivation(BaseModel):
    name: str
    kind: str  # "feature_maps" | "vector"
    shape: list[int]
    data: list[float]  # flattened activations, normalized to [0, 1]


class PredictResponse(BaseModel):
    digit: int
    probabilities: list[float]
    activations: list[LayerActivation]


@app.get("/")
def index():
    return FileResponse(STATIC_INDEX)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if net is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    arr = np.array(req.pixels, dtype=np.float64)
    if arr.shape != (28, 28):
        raise HTTPException(status_code=400, detail=f"expected 28x28, got {arr.shape}")
    arr = np.clip(arr, 0.0, 1.0).reshape(1, 1, 28, 28)
    probs = net.feedforward(arr, mini_batch_size=1).flatten()

    activations = []
    for i, layer in enumerate(net.layers):
        out = np.asarray(layer.output)
        # Conv layer output is (N, n_filters, H, W); FC/Output is (n_out, N).
        if out.ndim == 4:
            # (1, n_filters, H, W) -> n_filters maps of HxW
            n, f, h, w = out.shape
            maps = out[0]                                  # (f, h, w)
            # Per-map normalization to [0, 1] so weak filters still show structure
            mn = maps.reshape(f, -1).min(axis=1).reshape(f, 1, 1)
            mx = maps.reshape(f, -1).max(axis=1).reshape(f, 1, 1)
            denom = np.where(mx - mn > 1e-9, mx - mn, 1.0)
            norm = (maps - mn) / denom
            activations.append(LayerActivation(
                name=f"conv{i}",
                kind="feature_maps",
                shape=[f, h, w],
                data=norm.flatten().tolist(),
            ))
        else:
            vec = out.flatten()
            mx = float(vec.max()) if vec.size else 1.0
            norm = vec / mx if mx > 1e-9 else vec
            kind = "output" if i == len(net.layers) - 1 else "vector"
            activations.append(LayerActivation(
                name=f"layer{i}",
                kind=kind,
                shape=list(out.shape),
                data=norm.tolist(),
            ))

    return PredictResponse(
        digit=int(np.argmax(probs)),
        probabilities=probs.tolist(),
        activations=activations,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
