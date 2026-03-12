from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE      = os.path.join(BASE_DIR, "models", "knn_model.pkl")
TRANSFORMER_FILE = os.path.join(BASE_DIR, "models", "transformer.pkl")
DATASET_CACHE   = os.path.join(BASE_DIR, "models", "data_cache.pkl")

# ── Lazy-load models once on first request ───────────────────────────────────
_knn = None
_preprocessor = None
_df = None

def load_models():
    global _knn, _preprocessor, _df
    if _knn is None:
        _knn = joblib.load(MODEL_FILE)
        _preprocessor = joblib.load(TRANSFORMER_FILE)
        _df = pd.read_pickle(DATASET_CACHE)


# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return jsonify({"status": "ok", "message": "Wahe Jewellery Recommendation API"})


# ── Recommend endpoint ────────────────────────────────────────────────────────
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        load_models()

        data = request.get_json(force=True, silent=True) or {}

        shape    = data.get("shape", "unknown")
        carat    = data.get("carat", None)
        gold     = data.get("gold", "unknown")
        category = data.get("category", None)
        n        = int(data.get("n", 5))

        df = _df.copy()

        # ── Filter by category ────────────────────────────────────────────────
        if category and category.lower() not in ("not sure", "unknown"):
            cat_lower = category.lower()
            # Normalise plurals
            cat_map = {
                "rings": "ring", "earrings": "earring",
                "necklaces": "necklace", "bangles": "bangle",
                "bracelets": "bangle",
            }
            cat_lower = cat_map.get(cat_lower, cat_lower)
            filtered = df[df["category"] == cat_lower]
            if len(filtered) > 0:
                df = filtered

        # ── Prepare input ─────────────────────────────────────────────────────
        if carat is None:
            carat_val = float(_df["carat"].median())
        else:
            carat_val = float(carat)

        input_data = pd.DataFrame({
            "shape": [str(shape).lower()],
            "carat": [carat_val],
            "gold":  [str(gold).lower()],
        })

        features = ["shape", "carat", "gold"]
        X_filtered   = df[features]
        X_transformed = _preprocessor.transform(X_filtered)

        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(n, len(df))
        knn_dynamic = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
        knn_dynamic.fit(X_transformed)

        input_transformed = _preprocessor.transform(input_data)
        _, indices = knn_dynamic.kneighbors(input_transformed, n_neighbors=n_neighbors)
        recommendations = df.iloc[indices[0]]

        # ── Serialise ─────────────────────────────────────────────────────────
        recs = recommendations.replace({np.nan: None}).to_dict(orient="records")
        return jsonify(recs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Local dev entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
