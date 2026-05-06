from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from article_pos_pipeline import extract_article_style_pos

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze_video():
    file = request.files.get("video")
    if file is None or file.filename is None or file.filename.strip() == "":
        return jsonify({"error": "Please upload a video file."}), 400

    filename = secure_filename(file.filename)
    suffix = Path(filename).suffix or ".mp4"

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / f"input{suffix}"
            file.save(input_path)

            summary, estimates, _, _ = extract_article_style_pos(video_path=input_path)
            payload = {
                "summary": asdict(summary),
                "windows": [asdict(estimate) for estimate in estimates],
            }
            return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True)
