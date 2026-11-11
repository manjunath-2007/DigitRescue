"""DigitRescue ML inference, preprocessing, and synthetic corruption pipeline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import tensorflow as tf


DAMAGE_CATALOG: dict[str, dict[str, str]] = {
    "gaussian": {
        "label": "Gaussian noise",
        "classical_filter": "3x3 Gaussian smoothing (sigma auto)",
        "synthetic_spec": "Zero-mean Gaussian noise (sigma = 0.40), clipped to [0, 1]",
    },
    "salt_pepper": {
        "label": "Salt-and-pepper",
        "classical_filter": "3x3 Median filter",
        "synthetic_spec": "15% impulse noise (7.5% black, 7.5% white pixels)",
    },
    "blur": {
        "label": "Blur",
        "classical_filter": "Unsharp masking (sigma = 1.0, weight 1.5 / -0.5)",
        "synthetic_spec": "Gaussian blur (sigma = 1.3)",
    },
    "fading": {
        "label": "Fading",
        "classical_filter": "CLAHE contrast enhancement (clipLimit = 2.0, grid 4x4)",
        "synthetic_spec": "Spatial brightness attenuation (factor 0.32-0.48, smooth field sigma = 4.0)",
    },
    "erasure": {
        "label": "Erasure",
        "classical_filter": "CLAHE contrast enhancement (clipLimit = 2.0, grid 4x4)",
        "synthetic_spec": "2 rectangular zeroed patches (4-6 px) centered on foreground strokes",
    },
    "unknown": {
        "label": "Unknown",
        "classical_filter": "Generic pipeline (Gaussian smoothing sigma = 0.7 + CLAHE clipLimit = 1.5)",
        "synthetic_spec": "Fallback Gaussian noise (sigma = 0.40) when synthetic damage is enabled",
    },
}

_CLAHE_STRONG = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
_CLAHE_MILD = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))


@dataclass(frozen=True)
class ModelBundle:
    cnn: tf.keras.Model
    autoencoder: tf.keras.Model
    cnn_path: Path
    autoencoder_path: Path


def resolve_model_paths(base_dir: Path) -> tuple[Path, Path]:
    """Locate CNN and Autoencoder model files under models/ or project root."""
    candidates = [
        (base_dir / "models" / "digitrescue_cnn.keras", base_dir / "models" / "digitrescue_autoencoder.keras"),
        (base_dir / "digitrescue_cnn.keras", base_dir / "digitrescue_autoencoder.keras"),
    ]
    for cnn_p, ae_p in candidates:
        if cnn_p.exists() and ae_p.exists():
            return cnn_p, ae_p

    missing = [str(p) for p in candidates[0] if not p.exists()]
    raise FileNotFoundError(
        "Required model file(s) missing: " + ", ".join(missing)
    )


def load_models(base_dir: Path) -> ModelBundle:
    """Load both Keras models once at application startup."""
    cnn_path, ae_path = resolve_model_paths(base_dir)
    try:
        cnn = tf.keras.models.load_model(cnn_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load CNN model at {cnn_path}: {exc}") from exc
    try:
        autoencoder = tf.keras.models.load_model(ae_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load Autoencoder model at {ae_path}: {exc}") from exc
    return ModelBundle(cnn=cnn, autoencoder=autoencoder, cnn_path=cnn_path, autoencoder_path=ae_path)


def decode_uploaded_image(raw_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes with OpenCV and convert to single-channel uint8 grayscale."""
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("File could not be decoded as a valid PNG or JPEG image.")

    if decoded.ndim == 2:
        return decoded.astype(np.uint8)

    if decoded.ndim == 3 and decoded.shape[2] == 4:
        # Composite transparent PNGs onto white paper before converting to grayscale
        bgr = decoded[:, :, :3].astype(np.float32)
        alpha = decoded[:, :, 3:4].astype(np.float32) / 255.0
        composited = bgr * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(np.clip(composited, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if decoded.ndim == 3 and decoded.shape[2] >= 3:
        return cv2.cvtColor(decoded[:, :, :3], cv2.COLOR_BGR2GRAY)

    raise ValueError("Unsupported image channel format.")


def prepare_digit(gray: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Normalize a grayscale image to MNIST format (28x28 uint8, white digit on black).

    28x28 MNIST-style inputs pass through without bounding-box cropping or resizing;
    larger images with dark strokes on light backgrounds are inverted, cropped to
    their foreground bounding box, and centered in a 20x20 box inside 28x28.
    """
    img = gray.astype(np.uint8, copy=True)
    border = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    inverted = bool(np.mean(border) > 127.0)
    if inverted:
        img = 255 - img

    if img.shape == (28, 28):
        return img, inverted

    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    points = cv2.findNonZero(mask)
    if points is None:
        return cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA), inverted

    x, y, w, h = cv2.boundingRect(points)
    digit = img[y : y + h, x : x + w]
    scale = 20.0 / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = digit_resized
    return canvas, inverted


def apply_synthetic_damage(
    image_u8: np.ndarray, damage_type: str, seed: int | None = None
) -> tuple[np.ndarray, str]:
    """Apply notebook-exact synthetic corruption to a 28x28 uint8 digit image."""
    applied = "gaussian" if damage_type == "unknown" else damage_type
    if applied not in DAMAGE_CATALOG:
        raise ValueError(f"Unsupported damage type: {damage_type}")

    rng = np.random.default_rng(seed)
    x = image_u8.astype(np.float32) / 255.0

    if applied == "gaussian":
        x = np.clip(x + rng.normal(0.0, 0.4, x.shape), 0.0, 1.0)
    elif applied == "salt_pepper":
        r = rng.random(x.shape)
        x[r < 0.075] = 0.0
        x[(r >= 0.075) & (r < 0.15)] = 1.0
    elif applied == "blur":
        x = gaussian_filter(x, sigma=1.3)
    elif applied == "fading":
        factor = float(rng.uniform(0.32, 0.48))
        smooth = gaussian_filter(rng.random(x.shape), sigma=4.0)
        low = float(smooth.min())
        high = float(smooth.max())
        smooth = (smooth - low) / (high - low + 1e-8)
        x = x * factor * (0.5 + 0.5 * smooth)
    elif applied == "erasure":
        out = image_u8.copy()
        ys, xs = np.nonzero(image_u8 > 80)
        if len(ys) > 0:
            for _ in range(2):
                k = int(rng.integers(len(ys)))
                h = int(rng.integers(4, 7))
                w = int(rng.integers(4, 7))
                top = max(0, int(ys[k]) - h // 2)
                left = max(0, int(xs[k]) - w // 2)
                out[top : top + h, left : left + w] = 0
        return out, applied

    return np.clip(x * 255.0, 0.0, 255.0).astype(np.uint8), applied


def preprocess_classical(image_u8: np.ndarray, damage_type: str) -> np.ndarray:
    """Apply classical OpenCV preprocessing branch matching AutoDigit.ipynb Cell 39."""
    img = cv2.resize(image_u8, (28, 28), interpolation=cv2.INTER_AREA)
    if damage_type == "gaussian":
        return cv2.GaussianBlur(img, (3, 3), 0)
    if damage_type == "salt_pepper":
        return cv2.medianBlur(img, 3)
    if damage_type == "blur":
        blurred = cv2.GaussianBlur(img, (0, 0), 1.0)
        return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    if damage_type in ("fading", "erasure"):
        return _CLAHE_STRONG.apply(img)
    if damage_type == "unknown":
        smoothed = cv2.GaussianBlur(img, (3, 3), 0.7)
        return _CLAHE_MILD.apply(smoothed)
    raise ValueError(f"Unsupported damage type: {damage_type}")


def restore_with_autoencoder(ae_model: tf.keras.Model, image_u8: np.ndarray) -> np.ndarray:
    """Restore a 28x28 uint8 damaged image using the Denoising Autoencoder."""
    x = (image_u8.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
    restored = ae_model.predict(x, verbose=0)[0, :, :, 0]
    return np.clip(restored * 255.0, 0.0, 255.0).astype(np.uint8)


def predict_digit(cnn_model: tf.keras.Model, image_u8: np.ndarray) -> dict[str, Any]:
    """Run the CNN classifier on a 28x28 uint8 image and return top-1 and top-3 probabilities."""
    x = (image_u8.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)
    probs = cnn_model.predict(x, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:3]
    top_digit = int(top_indices[0])
    return {
        "digit": top_digit,
        "confidence": round(float(probs[top_digit]) * 100.0, 2),
        "top3": [
            {"digit": int(idx), "probability": round(float(probs[idx]) * 100.0, 2)}
            for idx in top_indices
        ],
    }


def encode_png_bytes(image_u8: np.ndarray, upscale: int = 10) -> bytes:
    """Encode a 28x28 uint8 image as crisp nearest-neighbor upscaled PNG bytes."""
    if upscale > 1:
        h, w = image_u8.shape[:2]
        out = cv2.resize(image_u8, (w * upscale, h * upscale), interpolation=cv2.INTER_NEAREST)
    else:
        out = image_u8
    ok, buf = cv2.imencode(".png", out)
    if not ok:
        raise RuntimeError("Failed to encode PNG output.")
    return buf.tobytes()


def encode_png_data_url(image_u8: np.ndarray, upscale: int = 8) -> str:
    """Encode a 28x28 uint8 image as a base64 PNG data URL."""
    b64 = base64.b64encode(encode_png_bytes(image_u8, upscale=upscale)).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_full_pipeline(
    bundle: ModelBundle,
    raw_bytes: bytes,
    damage_type: str,
    add_synthetic_damage: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Execute normalization, optional synthetic damage, classical branch, AE restoration, and CNN predictions."""
    if damage_type not in DAMAGE_CATALOG:
        raise ValueError(f"Invalid damage type '{damage_type}'.")

    gray = decode_uploaded_image(raw_bytes)
    normalized, inverted = prepare_digit(gray)

    applied_synthetic = None
    if add_synthetic_damage:
        original_for_models, applied_synthetic = apply_synthetic_damage(
            normalized, damage_type, seed=seed
        )
    else:
        original_for_models = normalized

    preprocessed = preprocess_classical(original_for_models, damage_type)
    restored = restore_with_autoencoder(bundle.autoencoder, original_for_models)

    pred_original = predict_digit(bundle.cnn, original_for_models)
    pred_preprocessed = predict_digit(bundle.cnn, preprocessed)
    pred_restored = predict_digit(bundle.cnn, restored)

    return {
        "damage_type": damage_type,
        "damage_label": DAMAGE_CATALOG[damage_type]["label"],
        "classical_filter": DAMAGE_CATALOG[damage_type]["classical_filter"],
        "inverted": inverted,
        "synthetic_damage_applied": applied_synthetic,
        "original": {
            "image": encode_png_data_url(original_for_models, upscale=8),
            **pred_original,
        },
        "preprocessed": {
            "image": encode_png_data_url(preprocessed, upscale=8),
            **pred_preprocessed,
        },
        "restored": {
            "image": encode_png_data_url(restored, upscale=8),
            **pred_restored,
        },
        "restored_download_bytes": encode_png_bytes(restored, upscale=10),
    }
