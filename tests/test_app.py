from __future__ import annotations

import io
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import app as app_module
import pipeline


@pytest.fixture(scope="module")
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _make_digit_image_bytes(fmt: str = ".png", size: int = 28, white_bg: bool = False) -> bytes:
    """Create a synthetic digit '1' or '7' image encoded as PNG or JPEG."""
    img = np.zeros((size, size), dtype=np.uint8)
    thickness = max(2, size // 12)
    cv2.line(img, (size // 4, size // 4), (size * 3 // 4, size // 4), 255, thickness)
    cv2.line(img, (size * 3 // 4, size // 4), (size // 2, size * 4 // 5), 255, thickness)
    if white_bg:
        img = 255 - img
    ok, buf = cv2.imencode(fmt, img)
    assert ok
    return buf.tobytes()


def test_health_and_pages(client):
    health = client.get("/health")
    assert health.status_code == 200
    data = health.get_json()
    assert data["status"] == "ok"
    assert data["models_loaded"] is True

    for route in ("/", "/restoration", "/performance", "/how-it-works", "/about"):
        resp = client.get(route)
        assert resp.status_code == 200, f"Route {route} failed with {resp.status_code}"

    perf_html = client.get("/performance").get_data(as_text=True)
    assert "99.19%" in perf_html
    assert "10000" in perf_html
    assert "Not recorded" in perf_html


def test_missing_models_readable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Required model file"):
        pipeline.load_models(tmp_path)


def test_28x28_mnist_style_passthrough_unchanged():
    arr = np.zeros((28, 28), dtype=np.uint8)
    arr[4:22, 12:15] = 240
    out, inverted = pipeline.prepare_digit(arr)
    assert inverted is False
    assert np.array_equal(out, arr)


@pytest.mark.parametrize(
    "filename,fmt,mimetype,white_bg",
    [
        ("digit28.png", ".png", "image/png", False),
        ("digit_white.jpg", ".jpg", "image/jpeg", True),
        ("digit_white.jpeg", ".jpeg", "image/jpeg", True),
    ],
)
def test_valid_png_jpg_jpeg_upload_and_download(client, filename, fmt, mimetype, white_bg):
    size = 28 if not white_bg else 96
    raw = _make_digit_image_bytes(fmt=fmt, size=size, white_bg=white_bg)
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(raw), filename, mimetype),
            "damage_type": "gaussian",
            "add_synthetic_damage": "false",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["inverted"] is white_bg
    for stage in ("original", "preprocessed", "restored"):
        assert payload[stage]["image"].startswith("data:image/png;base64,")
        assert 0 <= payload[stage]["digit"] <= 9
        assert 0.0 <= payload[stage]["confidence"] <= 100.0
        assert len(payload[stage]["top3"]) == 3

    dl = client.get(payload["download_url"])
    assert dl.status_code == 200
    assert dl.mimetype == "image/png"
    decoded = cv2.imdecode(np.frombuffer(dl.data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    assert decoded is not None
    assert decoded.shape == (280, 280)


def test_wrong_file_type_rejected(client):
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(b"not an image"), "notes.txt", "text/plain"),
            "damage_type": "gaussian",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "Unsupported file type" in resp.get_json()["error"]


def test_corrupt_image_rejected(client):
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(b"corrupted-png-bytes-12345"), "broken.png", "image/png"),
            "damage_type": "gaussian",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "could not be decoded" in resp.get_json()["error"]


def test_oversize_file_rejected(client):
    oversize = b"\x89PNG\r\n\x1a\n" + (b"0" * (5 * 1024 * 1024 + 1024))
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(oversize), "huge.png", "image/png"),
            "damage_type": "gaussian",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 413
    assert "5 MB" in resp.get_json()["error"]


@pytest.mark.parametrize(
    "damage_type",
    ["gaussian", "salt_pepper", "blur", "fading", "erasure", "unknown"],
)
def test_all_corruption_types_with_synthetic_damage(client, damage_type):
    raw = _make_digit_image_bytes(fmt=".png", size=28, white_bg=False)
    resp = client.post(
        "/api/restore",
        data={
            "image": (io.BytesIO(raw), f"sample_{damage_type}.png", "image/png"),
            "damage_type": damage_type,
            "add_synthetic_damage": "true",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    expected_applied = "gaussian" if damage_type == "unknown" else damage_type
    assert payload["synthetic_damage_applied"] == expected_applied
    assert len(payload["restored"]["top3"]) == 3
    assert payload["download_url"].startswith("/api/download/")
