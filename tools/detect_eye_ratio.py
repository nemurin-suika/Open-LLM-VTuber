"""
Live2D model eye position detector.

Parses the moc3 binary to extract vertex positions and estimates
where the character's eyes are, then outputs eyeTopRatio and
eyeLeftRatio values for model_dict.json.

Usage:
    uv run python tools/detect_eye_ratio.py <model_name>
    uv run python tools/detect_eye_ratio.py mao_pro

The model name must match an entry in model_dict.json.
"""

import json
import os
import struct
import sys

import chardet
import numpy as np


def load_model_dict(model_dict_path: str = "model_dict.json") -> list:
    with open(model_dict_path, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    return json.loads(raw.decode(enc, errors="replace"))


def find_model(model_dict: list, model_name: str) -> dict:
    for m in model_dict:
        if m["name"] == model_name:
            return m
    raise KeyError(f"Model '{model_name}' not found in model_dict.json")


def parse_moc3(moc3_path: str) -> dict:
    """
    Parse a Cubism 4 moc3 binary and return:
      - canvas_info: {ppu, originX, originY, canvasW, canvasH}
      - vertices:    numpy array (N, 2) of vertex positions in pixel space
    """
    with open(moc3_path, "rb") as f:
        data = f.read()

    magic = data[:4]
    if magic != b"MOC3":
        raise ValueError(f"Not a moc3 file: {moc3_path}")

    version = data[4]
    if version < 3:
        raise ValueError(f"Unsupported moc3 version {version} (need ≥3)")

    def u32(off):
        return struct.unpack_from("<I", data, off)[0]

    def f32(off):
        return struct.unpack_from("<f", data, off)[0]

    # --- CountInfo (section 0) ---
    sec0 = u32(0x40)
    n_artmeshes = u32(sec0 + 4 * 4)       # [4] = ArtMesh count
    n_verts_total = u32(sec0 + 15 * 4)    # [15] = total vertex count

    # --- CanvasInfo (section 1) ---
    # Layout: pixelsPerUnit, originX, originY, canvasWidth, canvasHeight
    sec1 = u32(0x44)
    ppu = f32(sec1 + 0)
    origin_x = f32(sec1 + 4)
    origin_y = f32(sec1 + 8)
    canvas_w = f32(sec1 + 12)
    canvas_h = f32(sec1 + 16)

    # Locate the vertex-position keyform section.
    # It is the first large section (>1 MB) whose floats fall in a
    # reasonable pixel range for this canvas.
    max_coord = max(canvas_w, canvas_h) * 2  # allow some overshoot
    vert_sec_offset = None
    offsets = []
    for i in range(200):
        off = u32(0x40 + i * 4)
        if off == 0 and i > 5:
            break
        offsets.append(off)

    for i, off in enumerate(offsets):
        size = (offsets[i + 1] - off) if i + 1 < len(offsets) else 0
        if size < n_verts_total * 8:
            continue
        # Sample a few float values
        sample = [f32(off + j * 4) for j in range(10)]
        if all(-max_coord < v < max_coord for v in sample):
            # Confirm by checking roughly expected vertex count
            vert_sec_offset = off
            break

    if vert_sec_offset is None:
        raise RuntimeError("Could not locate vertex position section in moc3")

    # Read first n_verts_total vertex positions (default-pose keyframe)
    raw_verts = data[vert_sec_offset: vert_sec_offset + n_verts_total * 8]
    verts = np.frombuffer(raw_verts, dtype="<f4").reshape(n_verts_total, 2)

    return {
        "canvas_info": {
            "ppu": ppu,
            "originX": origin_x,
            "originY": origin_y,
            "canvasW": canvas_w,
            "canvasH": canvas_h,
        },
        "vertices": verts,
        "n_artmeshes": n_artmeshes,
    }


def detect_eye_ratios(moc3_data: dict) -> dict:
    """
    Estimate the eye position from vertex density analysis.

    Returns:
        {
          "eyeTopRatio":  float (canvas y fraction of detected eye centre),
          "eyeLeftRatio": float (canvas x fraction of detected eye centre),
          "eye_canvas_x": float,
          "eye_canvas_y": float,
          "confidence":   str,
        }
    """
    ci = moc3_data["canvas_info"]
    verts = moc3_data["vertices"]

    # Convert pixel coordinates to canvas fraction (0=top-left, 1=bottom-right)
    frac_x = (verts[:, 0] + ci["originX"]) / ci["canvasW"]
    frac_y = (verts[:, 1] + ci["originY"]) / ci["canvasH"]

    # Clamp to valid canvas region to ignore deformation artefacts
    mask = (frac_x > 0.05) & (frac_x < 0.95) & (frac_y > 0.05) & (frac_y < 0.95)
    fx = frac_x[mask]
    fy = frac_y[mask]

    # Determine model content bounding box
    y_min, y_max = fy.min(), fy.max()
    x_min, x_max = fx.min(), fx.max()

    # The face occupies the upper ~40 % of the visible bounding box.
    # The lower portion (body, clothing) usually has far more vertices.
    face_y_cutoff = y_min + (y_max - y_min) * 0.45
    face_mask = fy < face_y_cutoff
    face_fx = fx[face_mask]
    face_fy = fy[face_mask]

    if len(face_fy) < 10:
        # Fallback: use upper quarter of all vertices
        face_y_cutoff = y_min + (y_max - y_min) * 0.30
        face_mask = fy < face_y_cutoff
        face_fx = fx[face_mask]
        face_fy = fy[face_mask]
        confidence = "low"
    else:
        confidence = "medium"

    # The eye row: find the y-band with highest HORIZONTAL density variance
    # (two eyes create two peaks → higher variance than e.g. uniform forehead)
    best_var = -1
    best_y = face_fy.mean()

    n_bins = 40
    y_edges = np.linspace(y_min, face_y_cutoff, n_bins + 1)
    for b in range(n_bins):
        band = (fy >= y_edges[b]) & (fy < y_edges[b + 1])
        if band.sum() < 20:
            continue
        bx = fx[band]
        # Split into left / right half and compare densities
        var = np.var(bx)
        if var > best_var:
            best_var = var
            best_y = (y_edges[b] + y_edges[b + 1]) / 2

    # Eye X centre: within the face region near the detected eye row, find centroid
    eye_band = (fy >= best_y - 0.05) & (fy < best_y + 0.05)
    if eye_band.sum() > 5:
        eye_cx = fx[eye_band].mean()
        eye_cy = fy[eye_band].mean()
        confidence = "high"
    else:
        eye_cx = (x_min + x_max) / 2
        eye_cy = best_y

    # eyeTopRatio  = canvas y fraction of the eye
    # eyeLeftRatio = canvas x fraction of the eye
    # (formula in live2d_model.py uses these directly as canvas fractions)
    eye_top_ratio = float(eye_cy)
    eye_left_ratio = float(eye_cx)

    return {
        "eyeTopRatio": round(eye_top_ratio, 3),
        "eyeLeftRatio": round(eye_left_ratio, 3),
        "eye_canvas_x": round(eye_cx, 3),
        "eye_canvas_y": round(eye_cy, 3),
        "model_top": round(float(y_min), 3),
        "model_bottom": round(float(y_max), 3),
        "confidence": confidence,
    }


def print_ascii_heatmap(moc3_data: dict) -> None:
    """Print a simple ASCII heatmap of vertex density."""
    ci = moc3_data["canvas_info"]
    verts = moc3_data["vertices"]
    frac_x = (verts[:, 0] + ci["originX"]) / ci["canvasW"]
    frac_y = (verts[:, 1] + ci["originY"]) / ci["canvasH"]
    mask = (frac_x > 0.05) & (frac_x < 0.95) & (frac_y > 0.05) & (frac_y < 0.95)
    fx, fy = frac_x[mask], frac_y[mask]

    cols, rows = 20, 12
    print("\nVertex density heatmap (canvas space, X=left→right, Y=top→bottom):")
    print("    " + "".join(f"{int(c/cols*10)}" for c in range(cols)))
    for r in range(rows):
        y0, y1 = r / rows, (r + 1) / rows
        row_str = f"{y0:.2f} "
        for c in range(cols):
            x0, x1 = c / cols, (c + 1) / cols
            cnt = ((fx >= x0) & (fx < x1) & (fy >= y0) & (fy < y1)).sum()
            if cnt > 2000:
                row_str += "█"
            elif cnt > 500:
                row_str += "▓"
            elif cnt > 100:
                row_str += "▒"
            elif cnt > 20:
                row_str += "░"
            else:
                row_str += " "
        print(row_str)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model_name = sys.argv[1]
    model_dict_path = sys.argv[2] if len(sys.argv) > 2 else "model_dict.json"

    print(f"Loading model_dict from {model_dict_path} ...")
    model_dict = load_model_dict(model_dict_path)
    model = find_model(model_dict, model_name)

    # Resolve moc3 path
    model_url: str = model.get("url", "")
    if model_url.startswith("/"):
        model_url = model_url.lstrip("/")
    model3_path = model_url  # e.g. live2d-models/xxx/xxx.model3.json

    with open(model3_path, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)["encoding"] or "utf-8"
    model3 = json.loads(raw.decode(enc, errors="replace"))

    base_dir = os.path.dirname(model3_path)
    moc_rel = model3.get("FileReferences", {}).get("Moc", "")
    moc3_path = os.path.join(base_dir, moc_rel)

    print(f"Parsing {moc3_path} ...")
    moc3_data = parse_moc3(moc3_path)
    ci = moc3_data["canvas_info"]

    print(f"\nCanvas: {ci['canvasW']:.0f} x {ci['canvasH']:.0f} px, "
          f"origin=({ci['originX']:.0f}, {ci['originY']:.0f}), ppu={ci['ppu']:.0f}")
    print(f"ArtMeshes: {moc3_data['n_artmeshes']}, "
          f"Vertices: {len(moc3_data['vertices'])}")

    print_ascii_heatmap(moc3_data)

    result = detect_eye_ratios(moc3_data)

    print(f"\n{'='*50}")
    print(f"Detected eye position (canvas fraction):")
    print(f"  x = {result['eye_canvas_x']}  (0=left, 1=right)")
    print(f"  y = {result['eye_canvas_y']}  (0=top,  1=bottom)")
    print(f"  Model visible range: y = {result['model_top']} – {result['model_bottom']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"\nAdd to model_dict.json for '{model_name}':")
    print(f'  "eyeTopRatio":  {result["eyeTopRatio"]},')
    print(f'  "eyeLeftRatio": {result["eyeLeftRatio"]},')

    # If current values exist, show comparison
    cur_top = model.get("eyeTopRatio", "not set")
    cur_left = model.get("eyeLeftRatio", "not set")
    print(f"\nCurrent values in model_dict.json:")
    print(f"  eyeTopRatio:  {cur_top}")
    print(f"  eyeLeftRatio: {cur_left}")

    print(f"\nNote: these are estimates from vertex density analysis.")
    print("Fine-tune by testing [focus:0.5,0.5] (screen center) and adjusting")
    print("until the character looks toward the actual screen center.")


if __name__ == "__main__":
    main()
