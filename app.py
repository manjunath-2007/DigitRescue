"""DigitRescue Flask Web Application."""

from __future__ import annotations

from collections import OrderedDict
import csv
import json
import os
from pathlib import Path
from threading import Lock
import uuid

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, Response, jsonify, render_template, request, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

import pipeline

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
STRIP_MANIFEST_PATH = BASE_DIR / "static" / "images" / "strip_manifest.json"

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/pjpeg"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_DOWNLOAD_CACHE = 64

NOISE_LABELS = {
    "gaussian": "Gaussian noise",
    "salt_pepper": "Salt-and-pepper",
    "blur": "Blur",
    "fading": "Fading",
    "erasure": "Erasure",
}


class DownloadStore:
    """Thread-safe bounded in-memory store for restored PNG downloads."""

    def __init__(self, max_items: int = MAX_DOWNLOAD_CACHE) -> None:
        self._items: OrderedDict[str, bytes] = OrderedDict()
        self._max_items = max_items
        self._lock = Lock()

    def put(self, png_bytes: bytes) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._items[token] = png_bytes
            while len(self._items) > self._max_items:
                self._items.popitem(last=False)
        return token

    def get(self, token: str) -> bytes | None:
        with self._lock:
            return self._items.get(token)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _parse_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_performance_data(results_dir: Path) -> dict:
    """Load recorded metrics exclusively from CSV files in results/."""
    summary_rows = _read_csv_rows(results_dir / "summary_metrics.csv")
    clean_accuracy: float | str = "Not recorded"
    test_images: int | str = "Not recorded"
    for row in summary_rows:
        if row.get("metric", "").strip().lower() == "clean accuracy":
            val = _parse_float(row.get("accuracy_percent"))
            if val is not None:
                clean_accuracy = round(val, 2)
            cnt = _parse_float(row.get("test_images"))
            if cnt is not None:
                test_images = int(cnt)

    acc_rows = []
    for row in _read_csv_rows(results_dir / "accuracy_comparison.csv"):
        key = row.get("Corruption", "").strip()
        noisy = _parse_float(row.get("Noisy Accuracy (%)"))
        preproc = _parse_float(row.get("Preprocessed Accuracy (%)"))
        restored = _parse_float(row.get("AI Restored Accuracy (%)"))
        acc_rows.append({
            "key": key,
            "label": NOISE_LABELS.get(key, key or "Not recorded"),
            "noisy": round(noisy, 2) if noisy is not None else "Not recorded",
            "preprocessed": round(preproc, 2) if preproc is not None else "Not recorded",
            "restored": round(restored, 2) if restored is not None else "Not recorded",
        })

    psnr_rows = []
    for row in _read_csv_rows(results_dir / "psnr_comparison.csv"):
        key = (row.get("") or row.get("Corruption") or "").strip()
        noisy_p = _parse_float(row.get("Noisy PSNR"))
        rest_p = _parse_float(row.get("Restored PSNR"))
        psnr_rows.append({
            "key": key,
            "label": NOISE_LABELS.get(key, key or "Not recorded"),
            "noisy_psnr": round(noisy_p, 2) if noisy_p is not None else "Not recorded",
            "restored_psnr": round(rest_p, 2) if rest_p is not None else "Not recorded",
        })

    ae_history = []
    for row in _read_csv_rows(results_dir / "autoencoder_training_history.csv"):
        ep = _parse_float(row.get("epoch"))
        loss = _parse_float(row.get("loss"))
        val_loss = _parse_float(row.get("val_loss"))
        mae = _parse_float(row.get("mae"))
        val_mae = _parse_float(row.get("val_mae"))
        ae_history.append({
            "epoch": int(ep) + 1 if ep is not None else "Not recorded",
            "loss": round(loss, 4) if loss is not None else "Not recorded",
            "val_loss": round(val_loss, 4) if val_loss is not None else "Not recorded",
            "mae": round(mae, 4) if mae is not None else "Not recorded",
            "val_mae": round(val_mae, 4) if val_mae is not None else "Not recorded",
        })

    cnn_history = []
    for row in _read_csv_rows(results_dir / "cnn_training_history.csv"):
        ep = _parse_float(row.get("epoch"))
        loss = _parse_float(row.get("loss"))
        val_loss = _parse_float(row.get("val_loss"))
        cnn_history.append({
            "epoch": int(ep) + 1 if ep is not None else "Not recorded",
            "loss": round(loss, 4) if loss is not None else "Not recorded",
            "val_loss": round(val_loss, 4) if val_loss is not None else "Not recorded",
        })

    return {
        "clean_accuracy": clean_accuracy,
        "test_images": test_images,
        "accuracy_rows": acc_rows,
        "psnr_rows": psnr_rows,
        "ae_history": ae_history,
        "cnn_history": cnn_history,
    }


def load_strip_manifest() -> list[dict]:
    if not STRIP_MANIFEST_PATH.exists():
        return []
    return json.loads(STRIP_MANIFEST_PATH.read_text(encoding="utf-8"))


def create_app(base_dir: Path = BASE_DIR) -> Flask:
    """Create and configure the DigitRescue Flask application."""
    model_bundle = pipeline.load_models(base_dir)
    downloads = DownloadStore()

    flask_app = Flask(
        __name__,
        template_folder=str(base_dir / "templates"),
        static_folder=str(base_dir / "static"),
    )
    flask_app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    flask_app.config["JSON_SORT_KEYS"] = False

    @flask_app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(_err: RequestEntityTooLarge):
        return jsonify({"error": "File exceeds the 5 MB upload limit."}), 413

    @flask_app.route("/")
    def index():
        return render_template(
            "index.html",
            active_page="home",
            strip_items=load_strip_manifest(),
        )

    @flask_app.route("/restoration")
    def restoration():
        return render_template(
            "restoration.html",
            active_page="restoration",
            damage_options=[
                {"value": key, "label": info["label"], "filter": info["classical_filter"]}
                for key, info in pipeline.DAMAGE_CATALOG.items()
            ],
        )

    @flask_app.route("/performance")
    def performance():
        perf = load_performance_data(base_dir / "results")
        return render_template(
            "performance.html",
            active_page="performance",
            perf=perf,
        )

    @flask_app.route("/how-it-works")
    def how_it_works():
        return render_template(
            "how_it_works.html",
            active_page="how_it_works",
        )

    @flask_app.route("/about")
    def about():
        return render_template(
            "about.html",
            active_page="about",
        )

    @flask_app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "models_loaded": True,
            "cnn_model": model_bundle.cnn_path.name,
            "autoencoder_model": model_bundle.autoencoder_path.name,
        })

    @flask_app.route("/api/restore", methods=["POST"])
    def api_restore():
        if "image" not in request.files:
            return jsonify({"error": "Please select an image file to upload."}), 400

        uploaded = request.files["image"]
        filename = secure_filename(uploaded.filename or "")
        if not filename:
            return jsonify({"error": "Please choose a PNG, JPG, or JPEG image."}), 400

        ext = Path(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Unsupported file type. Upload a PNG, JPG, or JPEG image."}), 400

        mime = (uploaded.mimetype or "").lower()
        if mime and mime not in ALLOWED_MIME_TYPES:
            return jsonify({"error": f"Unsupported MIME type '{mime}'. Expected PNG or JPEG."}), 400

        raw_bytes = uploaded.read()
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            return jsonify({"error": "File exceeds the 5 MB upload limit."}), 413

        damage_type = request.form.get("damage_type", "gaussian").strip().lower()
        if damage_type not in pipeline.DAMAGE_CATALOG:
            return jsonify({"error": f"Unsupported damage type '{damage_type}'."}), 400

        add_synthetic = request.form.get("add_synthetic_damage", "false").lower() in ("true", "1", "on", "yes")

        try:
            result = pipeline.run_full_pipeline(
                bundle=model_bundle,
                raw_bytes=raw_bytes,
                damage_type=damage_type,
                add_synthetic_damage=add_synthetic,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        png_bytes = result.pop("restored_download_bytes")
        token = downloads.put(png_bytes)
        result["download_url"] = url_for("api_download", token=token)
        return jsonify(result)

    @flask_app.route("/api/download/<token>")
    def api_download(token: str):
        if not token.isalnum() or len(token) != 32:
            return jsonify({"error": "Invalid download token."}), 400

        png_bytes = downloads.get(token)
        if png_bytes is None:
            return jsonify({"error": "Restored image has expired. Please run restoration again."}), 404

        return Response(
            png_bytes,
            mimetype="image/png",
            headers={
                "Content-Disposition": f'attachment; filename="digitrescue_restored_{token[:8]}.png"',
                "Cache-Control": "no-store",
            },
        )

    return flask_app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
