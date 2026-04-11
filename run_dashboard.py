"""
run_dashboard.py — Local server that runs experiments and serves the dashboard.

Usage:  python run_dashboard.py          # starts server on port 8050
        python run_dashboard.py 9000     # custom port
"""
import json
import os
import sys
import threading
import traceback
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from io import StringIO

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, balanced_accuracy_score)

from config import TRAIN_YEARS, TEST_YEARS, AUT_EXCLUDE_YEARS, LABEL_COL
from data import load_dataset
from feature_selection import build_feature_sets
from evaluate import make_model

# ── Experiment logic ──────────────────────────────────────────────────

def run_and_collect(df, feature_sets, model_type="rf", log=print):
    """Run ablation and return structured results dict."""
    results = {}

    for model_key, (feature_cols, model_label) in feature_sets.items():
        results[model_key] = {"label": model_label, "years": {}}

        for test_year in TEST_YEARS:
            train_mask = df["year"].isin(TRAIN_YEARS)
            test_mask  = df["year"] == test_year
            if not test_mask.any():
                continue

            X_train = df.loc[train_mask, feature_cols]
            y_train = df.loc[train_mask, LABEL_COL]
            X_test  = df.loc[test_mask, feature_cols]
            y_test  = df.loc[test_mask, LABEL_COL]

            k = test_year - max(TRAIN_YEARS)

            model = make_model(model_type)
            if model_type == "xgb":
                neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
                model.set_params(scale_pos_weight=neg / pos)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            results[model_key]["years"][str(test_year)] = {
                "k": k,
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
                "train_malware_pct": round(float(y_train.mean()) * 100, 1),
                "test_malware_pct": round(float(y_test.mean()) * 100, 1),
                "accuracy":  round(float(accuracy_score(y_test, preds)), 4),
                "precision": round(float(precision_score(y_test, preds, average="binary")), 4),
                "recall":    round(float(recall_score(y_test, preds, average="binary")), 4),
                "f1_macro":  round(float(f1_score(y_test, preds, average="macro")), 4),
                "f1_weighted": round(float(f1_score(y_test, preds, average="weighted")), 4),
                "bal_acc":   round(float(balanced_accuracy_score(y_test, preds)), 4),
                "excluded_from_aut": test_year in AUT_EXCLUDE_YEARS,
            }

            log(f"  {model_label} k={k} ({test_year}): "
                f"acc={results[model_key]['years'][str(test_year)]['accuracy']:.4f}  "
                f"prec={results[model_key]['years'][str(test_year)]['precision']:.4f}  "
                f"rec={results[model_key]['years'][str(test_year)]['recall']:.4f}")

    return results


def run_experiment(model_type="rf"):
    """Full pipeline: load data, build features, run, return JSON-ready dict."""
    log_buf = StringIO()
    def log(msg):
        print(msg, flush=True)
        log_buf.write(msg + "\n")

    log("Loading dataset...")
    df, perm_cols, temp_cols = load_dataset(verbose=False)
    log(f"Dataset: {len(df)} samples, {len(perm_cols)} static flags, {len(temp_cols)} temporal features")

    log("Building feature sets...")
    feature_sets, _ = build_feature_sets(df, perm_cols, temp_cols, prune=True)

    log("Running experiment...")
    results = run_and_collect(df, feature_sets, model_type=model_type, log=log)

    meta = {
        "train_years": TRAIN_YEARS,
        "test_years": TEST_YEARS,
        "aut_exclude": AUT_EXCLUDE_YEARS,
        "model_type": model_type,
        "model": {
            "rf": "RandomForest (n=100, balanced, max_features=0.5)",
            "xgb": "XGBoost (n=300, depth=6, lr=0.1)",
            "lr": "LogisticRegression (saga, balanced)",
        }.get(model_type, model_type),
    }

    output = {"meta": meta, "results": results, "log": log_buf.getvalue()}

    # Also save to disk
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard_data.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"Results saved to {out_path}")

    return output


# ── Background job state ──────────────────────────────────────────────

_job_lock = threading.Lock()
_job = {
    "status": "idle",      # idle | running | done | error
    "result": None,
    "error": None,
}


def _run_job(model_type):
    global _job
    try:
        result = run_experiment(model_type=model_type)
        with _job_lock:
            _job["status"] = "done"
            _job["result"] = result
    except Exception as e:
        traceback.print_exc()
        with _job_lock:
            _job["status"] = "error"
            _job["error"] = str(e)


# ── HTTP Server ───────────────────────────────────────────────────────

class DashboardHandler(SimpleHTTPRequestHandler):
    """Serves static files + POST /run to trigger experiments."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def _json_response(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global _job
        if self.path == "/run":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len) if content_len else b"{}"
            try:
                params = json.loads(body) if body.strip() else {}
            except json.JSONDecodeError:
                params = {}

            model_type = params.get("model_type", "rf")

            with _job_lock:
                if _job["status"] == "running":
                    self._json_response(409, {"error": "Experiment already running"})
                    return
                _job = {"status": "running", "result": None, "error": None}

            thread = threading.Thread(target=_run_job, args=(model_type,), daemon=True)
            thread.start()
            self._json_response(202, {"status": "started"})

        elif self.path == "/status":
            with _job_lock:
                resp = {"status": _job["status"]}
                if _job["status"] == "done":
                    resp["data"] = _job["result"]
                    _job["status"] = "idle"
                elif _job["status"] == "error":
                    resp["error"] = _job["error"]
                    _job["status"] = "idle"
            self._json_response(200, resp)

        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        if "POST" in str(args):
            super().log_message(format, *args)


SERVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    server = HTTPServer(("127.0.0.1", port), DashboardHandler)
    url = f"http://127.0.0.1:{port}/dashboard.html"

    print(f"Dashboard server running at {url}")
    print("Press Ctrl+C to stop.\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
