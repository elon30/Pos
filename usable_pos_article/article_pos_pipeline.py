#!/usr/bin/env python3
"""
Article-style rPPG/POS heart-rate extraction and lightweight video modification.

This script modernizes the old 2016-era POS prototype into a safer pipeline that
matches the 2025 paper's described rPPG estimation choices as closely as possible
without requiring the full pyVHR repository:

- MediaPipe Face Mesh ROI.
- Facial skin ROI excluding eyes and mouth, or selected forehead/cheeks/nose.
- 100 MediaPipe landmarks from the paper's released code for RGB extraction.
- 8-second windows with 1-second stride.
- POS BVP extraction with a 1.6-second internal window.
- 6th-order Butterworth bandpass filtering from 0.65 to 4.0 Hz.
- Welch PSD heart-rate estimation in the same 0.65 to 4.0 Hz range.

It also includes the lightweight video modification methods from the article
(blur, noise, cumulative time averaging, and 20-frame sliding time averaging).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable

import cv2
try:
    import mediapipe as mp
except Exception:
    mp = None
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
PAPER_LANDMARKS = [
    2, 3, 4, 5, 6, 8, 9, 10, 18, 21, 32, 35, 36, 43, 46, 47, 48, 50, 54,
    58, 67, 68, 69, 71, 92, 93, 101, 103, 104, 108, 109, 116, 117, 118,
    123, 132, 134, 135, 138, 139, 142, 148, 149, 150, 151, 152, 182, 187,
    188, 193, 197, 201, 205, 206, 207, 210, 211, 212, 216, 234, 248, 251,
    262, 265, 266, 273, 277, 278, 280, 284, 288, 297, 299, 322, 323, 330,
    332, 333, 337, 338, 345, 346, 361, 363, 364, 367, 368, 371, 377, 379,
    411, 412, 417, 421, 425, 426, 427, 430, 432, 436,
]

FOREHEAD_LMK = [
    54, 68, 103, 104, 67, 69, 109, 108, 10, 151, 338, 337, 297, 299, 332,
    333, 298, 63, 105, 66, 107, 9, 336, 296, 334, 293,
]
LEFT_CHEEK_LMK = [
    116, 117, 118, 119, 120, 234, 100, 101, 142, 123, 137, 36, 93, 205,
    147, 177, 187, 132, 207, 213, 215, 192, 58,
]
RIGHT_CHEEK_LMK = [
    349, 348, 347, 346, 345, 329, 447, 454, 330, 371, 266, 352, 323, 366,
    425, 376, 401, 427, 361, 433, 435, 416, 288,
]
MOUTH_LMK = [61, 185, 40, 39, 37, 0, 267, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61]
LEFT_EYE_LMK = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 133]
RIGHT_EYE_LMK = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398, 362]


@dataclass
class WindowEstimate:
    start_s: float
    center_s: float
    end_s: float
    bpm: float
    peak_hz: float
    valid: bool


@dataclass
class ExtractionSummary:
    video: str
    fps: float
    frames_read: int
    frames_with_face: int
    roi_mode: str
    rgb_mode: str
    patch_size: int
    window_seconds: float
    stride_seconds: float
    min_hz: float
    max_hz: float
    mean_bpm: float | None
    median_bpm: float | None
    valid_windows: int
    total_windows: int


class FaceMeshProcessor:
    def __init__(self, model_path: Path | None = None) -> None:
        self._backend = "haar"
        self._face_mesh = None
        self._landmarker = None
        self._haar = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        self._timestamp_ms = 0

        if mp is not None and hasattr(mp, "solutions"):
            self._backend = "solutions"
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return

        if mp is None:
            return

        if model_path is None:
            model_path = default_model_path()
        ensure_face_landmarker_model(model_path)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        if self._face_mesh is not None:
            self._face_mesh.close()
        if self._landmarker is not None:
            self._landmarker.close()

    def landmarks(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        height, width = frame_rgb.shape[:2]
        if self._backend == "solutions":
            result = self._face_mesh.process(frame_rgb)
            if not result.multi_face_landmarks:
                return None
            points = result.multi_face_landmarks[0].landmark
        elif self._backend == "tasks":
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect_for_video(image, self._timestamp_ms)
            self._timestamp_ms += 1
            if not result.face_landmarks:
                return None
            points = result.face_landmarks[0]
        else:
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                return None
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            # Return 468 pseudo-landmarks as a grid inside the detected face rectangle.
            cols, rows = 26, 18
            pts = []
            for r in range(rows):
                for c in range(cols):
                    px = int(x + (c + 0.5) * w / cols)
                    py = int(y + (r + 0.5) * h / rows)
                    pts.append([min(width - 1, max(0, px)), min(height - 1, max(0, py))])
            return np.array(pts[:468], dtype=np.int32)

        return np.array([[int(round(point.x * width)), int(round(point.y * height))] for point in points], dtype=np.int32)


def default_model_path() -> Path:
    return Path.home() / ".cache" / "article_pos_pipeline" / "face_landmarker.task"


def ensure_face_landmarker_model(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe Face Landmarker model to {model_path}...", file=sys.stderr)
    urllib.request.urlretrieve(FACE_LANDMARKER_URL, model_path)


def _fill_hull(mask: np.ndarray, points: np.ndarray, value: int) -> None:
    if points.size == 0:
        return
    hull = cv2.convexHull(points.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, value)


def roi_mask_from_landmarks(shape: tuple[int, int], landmarks: np.ndarray, mode: str) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if mode == "full-frame":
        mask[:, :] = 255
        return mask

    if mode == "face":
        _fill_hull(mask, landmarks, 255)
        _fill_hull(mask, landmarks[MOUTH_LMK], 0)
        _fill_hull(mask, landmarks[LEFT_EYE_LMK], 0)
        _fill_hull(mask, landmarks[RIGHT_EYE_LMK], 0)
        return mask

    if mode == "selected":
        _fill_hull(mask, landmarks[FOREHEAD_LMK], 255)
        _fill_hull(mask, landmarks[LEFT_CHEEK_LMK], 255)
        _fill_hull(mask, landmarks[RIGHT_CHEEK_LMK], 255)
        return mask

    raise ValueError(f"Unknown ROI mode: {mode}")


def rgb_from_landmark_patches(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    landmark_indices: Iterable[int] = PAPER_LANDMARKS,
    patch_size: int = 28,
    low_threshold: int = 2,
    high_threshold: int = 254,
) -> np.ndarray | None:
    height, width = frame_rgb.shape[:2]
    half = patch_size // 2
    samples: list[np.ndarray] = []

    for landmark_id in landmark_indices:
        x, y = landmarks[landmark_id]
        x0 = max(0, x - half)
        x1 = min(width, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(height, y + half + 1)
        patch = frame_rgb[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        valid = ~(
            np.all(patch <= low_threshold, axis=2)
            | np.all(patch >= high_threshold, axis=2)
        )
        pixels = patch[valid]
        if pixels.size:
            samples.append(pixels.reshape(-1, 3))

    if not samples:
        return None
    return np.concatenate(samples, axis=0).mean(axis=0).astype(np.float64)


def patch_rgb_estimators(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    landmark_indices: Iterable[int] = PAPER_LANDMARKS,
    patch_size: int = 28,
    low_threshold: int = 2,
    high_threshold: int = 254,
) -> np.ndarray | None:
    height, width = frame_rgb.shape[:2]
    half = patch_size // 2
    values: list[np.ndarray] = []

    for landmark_id in landmark_indices:
        x, y = landmarks[landmark_id]
        x0 = max(0, x - half)
        x1 = min(width, x + half + 1)
        y0 = max(0, y - half)
        y1 = min(height, y + half + 1)
        patch = frame_rgb[y0:y1, x0:x1]
        if patch.size == 0:
            values.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))
            continue

        valid = ~(
            np.all(patch <= low_threshold, axis=2)
            | np.all(patch >= high_threshold, axis=2)
        )
        pixels = patch[valid]
        if pixels.size:
            values.append(pixels.reshape(-1, 3).mean(axis=0).astype(np.float64))
        else:
            values.append(np.array([np.nan, np.nan, np.nan], dtype=np.float64))

    estimators = np.vstack(values)
    if np.all(~np.isfinite(estimators)):
        return None
    return estimators


def rgb_from_mask(
    frame_rgb: np.ndarray,
    mask: np.ndarray,
    low_threshold: int = 2,
    high_threshold: int = 254,
) -> np.ndarray | None:
    pixels = frame_rgb[mask > 0]
    if pixels.size == 0:
        return None
    valid = ~(
        np.all(pixels <= low_threshold, axis=1)
        | np.all(pixels >= high_threshold, axis=1)
    )
    pixels = pixels[valid]
    if pixels.size == 0:
        return None
    return pixels.mean(axis=0).astype(np.float64)


def read_rgb_trace(
    video_path: Path,
    roi_mode: str,
    rgb_mode: str,
    patch_size: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video FPS could not be read. Provide a normal video file with FPS metadata.")

    processor = FaceMeshProcessor()
    rgb_values: list[np.ndarray] = []
    frames_read = 0
    frames_with_face = 0

    try:
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames_read += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = processor.landmarks(frame_rgb)
            if landmarks is None:
                continue
            frames_with_face += 1

            if rgb_mode == "patches":
                estimators = patch_rgb_estimators(frame_rgb, landmarks, patch_size=patch_size)
                if estimators is None:
                    rgb = None
                else:
                    rgb = np.nanmean(estimators, axis=0)
            elif rgb_mode == "mask":
                mask = roi_mask_from_landmarks(frame_rgb.shape[:2], landmarks, roi_mode)
                rgb = rgb_from_mask(frame_rgb, mask)
            else:
                raise ValueError(f"Unknown RGB mode: {rgb_mode}")

            if rgb is not None and np.all(np.isfinite(rgb)):
                rgb_values.append(rgb)
    finally:
        processor.close()
        cap.release()

    if not rgb_values:
        raise RuntimeError("No valid RGB samples were extracted. Check video visibility and face detection.")
    return np.vstack(rgb_values), fps, frames_read, frames_with_face


def read_patch_estimator_trace(
    video_path: Path,
    patch_size: int,
    max_frames: int | None = None,
) -> tuple[np.ndarray, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError("Video FPS could not be read. Provide a normal video file with FPS metadata.")

    processor = FaceMeshProcessor()
    traces: list[np.ndarray] = []
    frames_read = 0
    frames_with_face = 0

    try:
        while True:
            if max_frames is not None and frames_read >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frames_read += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = processor.landmarks(frame_rgb)
            if landmarks is None:
                continue
            frames_with_face += 1

            estimators = patch_rgb_estimators(frame_rgb, landmarks, patch_size=patch_size)
            if estimators is not None:
                traces.append(estimators)
    finally:
        processor.close()
        cap.release()

    if not traces:
        raise RuntimeError("No valid patch RGB samples were extracted. Check video visibility and face detection.")
    return np.stack(traces, axis=0), fps, frames_read, frames_with_face


def bandpass_filter(signal: np.ndarray, fps: float, min_hz: float, max_hz: float, order: int = 6) -> np.ndarray:
    nyquist = fps / 2.0
    high = min(max_hz, nyquist * 0.99)
    low = min_hz
    if low <= 0 or low >= high:
        raise ValueError(f"Invalid bandpass range {min_hz}-{max_hz} Hz for fps={fps}")
    sos = butter(order, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal, axis=0)


def interpolate_nans(signal: np.ndarray) -> np.ndarray:
    out = np.array(signal, dtype=np.float64, copy=True)
    if out.ndim == 1:
        out = out[:, None]
        squeeze = True
    else:
        squeeze = False

    x = np.arange(out.shape[0])
    for col in range(out.shape[1]):
        values = out[:, col]
        valid = np.isfinite(values)
        if valid.all():
            continue
        if valid.sum() == 0:
            values[:] = 0.0
        elif valid.sum() == 1:
            values[:] = values[valid][0]
        else:
            values[~valid] = np.interp(x[~valid], x[valid], values[valid])
        out[:, col] = values
    return out[:, 0] if squeeze else out


def pos_bvp(rgb: np.ndarray, fps: float, pos_window_seconds: float = 1.6) -> np.ndarray:
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [frames, 3]")

    window = int(round(pos_window_seconds * fps))
    if window < 2:
        raise ValueError("POS window is too short for this FPS")
    if rgb.shape[0] <= window:
        raise ValueError("Not enough frames for POS extraction")

    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)
    h = np.zeros(rgb.shape[0], dtype=np.float64)
    eps = np.finfo(np.float64).eps

    for end in range(window, rgb.shape[0]):
        start = end - window + 1
        c = rgb[start:end + 1].T
        mean_color = np.mean(c, axis=1)
        if np.any(np.abs(mean_color) <= eps):
            continue
        cn = c / mean_color[:, None]
        s = projection @ cn
        std0 = np.std(s[0])
        std1 = np.std(s[1])
        if std0 <= eps or std1 <= eps:
            continue
        p = s[0] + (std0 / std1) * s[1]
        p_std = np.std(p)
        if p_std <= eps:
            continue
        h[start:end + 1] += (p - np.mean(p)) / p_std

    return h


def article_style_patch_bvp(
    patch_trace: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
) -> np.ndarray:
    if patch_trace.ndim != 3 or patch_trace.shape[2] != 3:
        raise ValueError("patch_trace must have shape [frames, estimators, 3]")

    bvps: list[np.ndarray] = []
    for estimator_idx in range(patch_trace.shape[1]):
        rgb = interpolate_nans(patch_trace[:, estimator_idx, :])
        if np.std(rgb) <= np.finfo(np.float64).eps:
            continue
        try:
            rgb_filtered = bandpass_filter(rgb, fps, min_hz, max_hz)
            bvp = pos_bvp(rgb_filtered, fps)
            bvp = bandpass_filter(bvp, fps, min_hz, max_hz)
        except ValueError:
            continue
        if np.any(np.isfinite(bvp)) and np.std(bvp) > np.finfo(np.float64).eps:
            bvps.append(bvp)

    if not bvps:
        raise RuntimeError("POS failed for all landmark patch estimators.")
    return np.median(np.vstack(bvps), axis=0)


def estimate_window_bpm(
    bvp: np.ndarray,
    fps: float,
    start: int,
    end: int,
    min_hz: float,
    max_hz: float,
) -> tuple[float, float, bool]:
    window_signal = bvp[start:end]
    if window_signal.size < 4 or not np.any(np.isfinite(window_signal)):
        return float("nan"), float("nan"), False

    window_signal = window_signal - np.nanmean(window_signal)
    if np.nanstd(window_signal) <= np.finfo(np.float64).eps:
        return float("nan"), float("nan"), False

    nperseg = min(len(window_signal), max(8, int(round(len(window_signal)))))
    freqs, psd = welch(window_signal, fs=fps, window="flattop", nperseg=nperseg)
    candidates = np.where((freqs >= min_hz) & (freqs <= max_hz))[0]
    if candidates.size == 0:
        return float("nan"), float("nan"), False
    peak_idx = candidates[np.argmax(psd[candidates])]
    peak_hz = float(freqs[peak_idx])
    return peak_hz * 60.0, peak_hz, True


def extract_article_style_pos(
    video_path: Path,
    roi_mode: str = "face",
    rgb_mode: str = "patches",
    patch_size: int = 28,
    window_seconds: float = 8.0,
    stride_seconds: float = 1.0,
    min_hz: float = 0.65,
    max_hz: float = 4.0,
    max_frames: int | None = None,
) -> tuple[ExtractionSummary, list[WindowEstimate], np.ndarray, np.ndarray]:
    if rgb_mode == "patches":
        patch_trace, fps, frames_read, frames_with_face = read_patch_estimator_trace(
            video_path=video_path,
            patch_size=patch_size,
            max_frames=max_frames,
        )
        rgb = np.nanmean(patch_trace, axis=1)
        bvp = article_style_patch_bvp(patch_trace, fps, min_hz, max_hz)
    else:
        rgb, fps, frames_read, frames_with_face = read_rgb_trace(
            video_path=video_path,
            roi_mode=roi_mode,
            rgb_mode=rgb_mode,
            patch_size=patch_size,
            max_frames=max_frames,
        )
        rgb = interpolate_nans(rgb)
        rgb_filtered = bandpass_filter(rgb, fps, min_hz, max_hz)
        bvp = pos_bvp(rgb_filtered, fps)
        bvp = bandpass_filter(bvp, fps, min_hz, max_hz)

    window_frames = int(round(window_seconds * fps))
    stride_frames = int(round(stride_seconds * fps))
    if rgb.shape[0] < window_frames:
        raise RuntimeError(
            f"Need at least {window_seconds:g}s of valid face frames; got {rgb.shape[0] / fps:.2f}s."
        )

    estimates: list[WindowEstimate] = []
    for start in range(0, rgb.shape[0] - window_frames + 1, stride_frames):
        end = start + window_frames
        bpm, peak_hz, valid = estimate_window_bpm(bvp, fps, start, end, min_hz, max_hz)
        estimates.append(
            WindowEstimate(
                start_s=start / fps,
                center_s=(start + window_frames / 2.0) / fps,
                end_s=end / fps,
                bpm=bpm,
                peak_hz=peak_hz,
                valid=valid,
            )
        )

    valid_bpms = np.array([estimate.bpm for estimate in estimates if estimate.valid], dtype=np.float64)
    summary = ExtractionSummary(
        video=str(video_path),
        fps=fps,
        frames_read=frames_read,
        frames_with_face=frames_with_face,
        roi_mode=roi_mode,
        rgb_mode=rgb_mode,
        patch_size=patch_size,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        min_hz=min_hz,
        max_hz=max_hz,
        mean_bpm=float(np.mean(valid_bpms)) if valid_bpms.size else None,
        median_bpm=float(np.median(valid_bpms)) if valid_bpms.size else None,
        valid_windows=int(valid_bpms.size),
        total_windows=len(estimates),
    )
    return summary, estimates, rgb, bvp


def _noise(frame: np.ndarray, mode: str, amount: float = 0.05, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame_float = frame.astype(np.float32)

    if mode == "gaussian":
        noisy = frame_float + rng.normal(0, 10.0, size=frame.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if mode == "saltpepper":
        output = frame.copy()
        selector = rng.random(frame.shape[:2])
        output[selector < amount / 2.0] = 0
        output[(selector >= amount / 2.0) & (selector < amount)] = 255
        return output
    if mode == "pepper":
        output = frame.copy()
        output[rng.random(frame.shape[:2]) < amount] = 0
        return output
    if mode == "poisson":
        vals = 2 ** np.ceil(np.log2(len(np.unique(frame))))
        noisy = rng.poisson(frame_float * vals) / vals
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if mode == "speckle":
        noisy = frame_float + frame_float * rng.normal(0, 0.1, size=frame.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    raise ValueError(f"Unknown noise mode: {mode}")


def modify_frame(frame_bgr: np.ndarray, method: str, history: Deque[np.ndarray], kernel_size: int) -> np.ndarray:
    if method == "median":
        return cv2.medianBlur(frame_bgr, kernel_size)
    if method == "gaussian":
        return cv2.GaussianBlur(frame_bgr, (kernel_size, kernel_size), 0)
    if method == "bilateral":
        return cv2.bilateralFilter(frame_bgr, kernel_size, 75, 75)
    if method in {"gaussian-noise", "saltpepper-noise", "poisson-noise", "pepper-noise", "speckle-noise"}:
        mode = method.replace("-noise", "")
        return _noise(frame_bgr, mode)
    if method == "time-average-cumulative":
        history.append(frame_bgr.copy())
        return np.mean(np.stack(history, axis=0), axis=0).astype(np.uint8)
    if method == "time-average-sliding":
        history.append(frame_bgr.copy())
        return np.mean(np.stack(history, axis=0), axis=0).astype(np.uint8)
    raise ValueError(f"Unknown modification method: {method}")


def modify_video(
    input_path: Path,
    output_path: Path,
    method: str,
    roi_mode: str = "face",
    kernel_size: int = 5,
    time_window_frames: int = 20,
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("Could not read video metadata")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    processor = FaceMeshProcessor()
    history: Deque[np.ndarray]
    if method == "time-average-sliding":
        history = deque(maxlen=time_window_frames)
    else:
        history = deque()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            landmarks = processor.landmarks(frame_rgb)
            if landmarks is None:
                writer.write(frame_bgr)
                continue

            mask = roi_mask_from_landmarks((height, width), landmarks, roi_mode)
            modified = modify_frame(frame_bgr, method, history, kernel_size)
            output = frame_bgr.copy()
            output[mask > 0] = modified[mask > 0]
            writer.write(output)
    finally:
        processor.close()
        cap.release()
        writer.release()


def write_outputs(
    output_prefix: Path,
    summary: ExtractionSummary,
    estimates: list[WindowEstimate],
    rgb: np.ndarray,
    bvp: np.ndarray,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output_prefix.with_suffix(".summary.json").write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    np.save(output_prefix.with_suffix(".rgb.npy"), rgb)
    np.save(output_prefix.with_suffix(".bvp.npy"), bvp)

    with output_prefix.with_suffix(".windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["start_s", "center_s", "end_s", "bpm", "peak_hz", "valid"])
        writer.writeheader()
        for estimate in estimates:
            writer.writerow(asdict(estimate))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Article-style POS rPPG pipeline and video modifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Estimate BPM from a facial video with POS.")
    extract_parser.add_argument("--video", required=True, type=Path)
    extract_parser.add_argument("--output-prefix", required=True, type=Path)
    extract_parser.add_argument("--roi", choices=["face", "selected", "full-frame"], default="face")
    extract_parser.add_argument("--rgb-mode", choices=["patches", "mask"], default="patches")
    extract_parser.add_argument("--patch-size", type=int, default=28)
    extract_parser.add_argument("--window-seconds", type=float, default=8.0)
    extract_parser.add_argument("--stride-seconds", type=float, default=1.0)
    extract_parser.add_argument("--min-hz", type=float, default=0.65)
    extract_parser.add_argument("--max-hz", type=float, default=4.0)
    extract_parser.add_argument("--max-frames", type=int)

    modify_parser = subparsers.add_parser("modify", help="Apply article-style lightweight ROI modification.")
    modify_parser.add_argument("--input", required=True, type=Path)
    modify_parser.add_argument("--output", required=True, type=Path)
    modify_parser.add_argument(
        "--method",
        required=True,
        choices=[
            "median",
            "gaussian",
            "bilateral",
            "gaussian-noise",
            "saltpepper-noise",
            "poisson-noise",
            "pepper-noise",
            "speckle-noise",
            "time-average-cumulative",
            "time-average-sliding",
        ],
    )
    modify_parser.add_argument("--roi", choices=["face", "selected", "full-frame"], default="face")
    modify_parser.add_argument("--kernel-size", type=int, default=5)
    modify_parser.add_argument("--time-window-frames", type=int, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "extract":
            summary, estimates, rgb, bvp = extract_article_style_pos(
                video_path=args.video,
                roi_mode=args.roi,
                rgb_mode=args.rgb_mode,
                patch_size=args.patch_size,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                min_hz=args.min_hz,
                max_hz=args.max_hz,
                max_frames=args.max_frames,
            )
            write_outputs(args.output_prefix, summary, estimates, rgb, bvp)
            print(json.dumps(asdict(summary), indent=2))
        elif args.command == "modify":
            if args.kernel_size % 2 == 0:
                raise ValueError("--kernel-size must be odd for OpenCV blur filters")
            modify_video(
                input_path=args.input,
                output_path=args.output,
                method=args.method,
                roi_mode=args.roi,
                kernel_size=args.kernel_size,
                time_window_frames=args.time_window_frames,
            )
            print(f"Modified video saved to {args.output}")
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
