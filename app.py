# app.py
import os
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# ----Config ----
MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl") # adjust filename if needed

# ----App----
app = Flask(_name_)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Fail fast with a helpful message 
    raise RuntimeError(f"Could not load model form {MODEL_PATH}: {e}")

@app.get("/health")
def health():
    return {"status": "ok"},200

@app.post("/predict")
def predict():
   """
   Accepts either:
   {"input": [[...Feature vector...], [...]]}  # 2D
   {"input": [...Feature vector...]}
   """
try:
    payload = request.get_json(force=True)
    x = payload.get("input")
    if x is None:
         return jsonify(error="Missing 'input'"),400

# Normalize to 2D array
if isinstances(x, list) and (len(x) > 0) and not isinstances(x[0], list):

X = np.array(x, dtype=float)
preds = model.predict(X)
# If your model returns numpy types, convert to Python
preds = preds.tolist()
return jsonify(prediction=preds),200

except Exception as e:
    return jsonify(error=str(e)),500

if__name__ == "__main__":
   # Local dev only; Render will run with Gunicorn (see startCommand below)
   app.run(host="0.0.0.0", port=int(os.environ.get("PORT",8000)))




