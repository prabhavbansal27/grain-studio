"""Grain Studio (hosted) — import a video, tune film-grain settings live, export.

Hosted twin of the local Grain Studio (~/Downloads/grain-lab/app.py). Same
calibrated compositing formula throughout: the merge happens in numpy, not
ffmpeg's `blend` filter, because blend silently range-converts a branch when
the two streams carry different color_range tags — that corrupts every
strength reading. See that project's build.py for the original derivation.

Two things differ from the local tool, both forced by hosting:
  - Import is a real upload (st.file_uploader), not a local-path/native
    file picker — a hosted app has no access to your filesystem.
  - The grain plate is a 4s, compressed proxy (assets/grain_plate.mp4)
    instead of the original 15s ProRes master, because GitHub caps files at
    100MB and film grain — being calibrated noise — barely compresses.
    Measured character (mean/sigma/kernel) matches the original closely;
    see the design conversation this shipped from for the fidelity numbers.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
PLATE = os.path.join(ROOT, "assets", "grain_plate.mp4")
PLATE_W, PLATE_H = 4096, 2160
PLATE_SS = 0.0
TARGET_SIGMA = 8.0
SIZES = {"native": 1, "1.5x": 1.5, "2x": 2, "3x": 3}

# Community Cloud's free tier caps a container at ~1GB RAM. Every allocation
# below that holds a run of decoded frames is sized off a byte budget rather
# than a flat frame count, so it scales itself down automatically at higher
# resolutions instead of silently exceeding that limit. Frame count is not a
# constant memory cost — a "3 second window" is ~20x heavier at 4K than at
# 480p — which is what the flat-frame-count version of this got wrong.
PREVIEW_BUDGET = 60 * 1024 * 1024       # source preview window (session-local)
PLATE_CAL_BUDGET = 16 * 1024 * 1024     # plate calibration window (process-wide cache)
EXPORT_CHUNK_BUDGET = 16 * 1024 * 1024  # per-chunk transient buffers during export


def frames_for_budget(budget, w, h, min_frames, max_frames=None):
    """How many w×h yuv420p frames fit in `budget` bytes, clamped to
    [min_frames, max_frames]. One frame already spatially averages over
    w*h pixels, so even a small min_frames gives a stable grain statistic —
    the floor exists for motion-check duration, not statistical accuracy.
    """
    n = max(min_frames, budget // frame_size(w, h))
    return int(min(n, max_frames)) if max_frames else int(n)


# ---- ffmpeg / numpy pipeline (ported from the local tool's app.py) ------

def ffprobe_info(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr.strip() or "ffprobe failed")
    info = json.loads(p.stdout)
    if not info.get("streams"):
        raise RuntimeError("no video stream found")
    st_ = info["streams"][0]
    num, den = st_["r_frame_rate"].split("/")
    fps = float(num) / float(den or 1)
    dur = float(st_.get("duration") or info.get("format", {}).get("duration") or 0)
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip() != ""
    return dict(width=int(st_["width"]), height=int(st_["height"]), fps=fps,
                duration=dur, has_audio=has_audio)


def frame_size(w, h):
    return w * h + 2 * (w // 2) * (h // 2)


def read_exact(stream, n):
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def planes(buf, w, h):
    ysz, csz = w * h, (w // 2) * (h // 2)
    n = buf.shape[0]
    return (buf[:, :ysz].reshape(n, h, w),
            buf[:, ysz:ysz + csz].reshape(n, h // 2, w // 2),
            buf[:, ysz + csz:].reshape(n, h // 2, w // 2))


def pack(y, u, v):
    n = y.shape[0]
    return np.concatenate([y.reshape(n, -1), u.reshape(n, -1), v.reshape(n, -1)], axis=1)


def decode_window(path, ss, n, w, h, vf=None):
    args = ["ffmpeg", "-v", "error", "-threads", "2", "-ss", str(ss), "-i", path]
    if vf:
        args += ["-vf", vf]
    args += ["-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]
    p = subprocess.run(args, capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-2000:])
    fsz = frame_size(w, h)
    buf = np.frombuffer(p.stdout, dtype=np.uint8)
    got = buf.shape[0] // fsz
    if got < 1:
        raise RuntimeError(f"decoded 0 frames from {os.path.basename(path)}")
    return buf[:got * fsz].reshape(got, fsz)


def mean_std(arr):
    """Mean/std of a uint8 array without ndarray.std()'s internal
    (x - x.mean())**2 step — that materializes a full-size float64
    temporary (8 bytes/pixel vs. 1 for uint8), which alone cost ~265MB
    on a 4-frame 4K calibration window. Summing in bounded sub-chunks
    keeps the temporary's size constant regardless of the array's.
    """
    flat = arr.reshape(-1)
    n = flat.size
    mean = float(flat.sum(dtype=np.float64) / n)
    ss = 0.0
    step = 4_000_000
    for i in range(0, n, step):
        chunk = flat[i:i + step].astype(np.float64)
        chunk -= mean
        ss += float(np.dot(chunk, chunk))
    return mean, (ss / n) ** 0.5


def weight(y):
    t = y.astype(np.float32)
    t -= 16.0
    t *= (1.0 / 219.0)
    np.clip(t, 0.0, 1.0, out=t)
    u = 1.0 - t
    t *= u
    t *= 4.0
    np.sqrt(t, out=t)
    np.maximum(t, 0.18, out=t)
    return t


COMPOSITE_CHUNK_BYTES = 12 * 1024 * 1024  # bounds peak working-set regardless of batch size


def composite(sy, py, mean, sigma, k, weighted):
    """Grain compositor. Chunks internally over the frame axis so a
    caller handing this a big multi-frame batch (a "preview clip" render,
    an export chunk) can't blow past this function's own memory bound —
    the float32 working arrays inside used to scale with whatever batch
    size the caller happened to pass, uncapped.
    """
    if sy.ndim == 2:
        return _composite_batch(sy, py, mean, sigma, k, weighted)
    n = sy.shape[0]
    frame_bytes = sy[0].nbytes
    chunk = max(1, COMPOSITE_CHUNK_BYTES // frame_bytes)
    if chunk >= n:
        return _composite_batch(sy, py, mean, sigma, k, weighted)
    out = np.empty(sy.shape, dtype=np.uint8)
    for i in range(0, n, chunk):
        out[i:i + chunk] = _composite_batch(sy[i:i + chunk], py[i:i + chunk], mean, sigma, k, weighted)
    return out


def _composite_batch(sy, py, mean, sigma, k, weighted):
    # weight(sy) is computed — and its own internal scratch buffer freed —
    # BEFORE dev is allocated. Doing it in the other order (as an earlier
    # version of this did) keeps three same-shape float32 buffers alive
    # at once instead of two; that alone roughly doubled peak memory here.
    w = weight(sy) if weighted else None
    dev = py.astype(np.float32)
    dev -= np.float32(mean)
    dev *= np.float32(TARGET_SIGMA * k / sigma)
    if weighted:
        dev *= w
        del w
    dev += sy
    np.clip(dev, 0, 255, out=dev)
    np.rint(dev, out=dev)
    return dev.astype(np.uint8)


def plate_geometry(preset, w, h):
    m = SIZES.get(preset, 1)
    if m == 1:
        if w <= PLATE_W and h <= PLATE_H:
            cx = ((PLATE_W - w) // 2) & ~1
            cy = ((PLATE_H - h) // 2) & ~1
            return f"crop={w}:{h}:{cx}:{cy}"
        return f"scale={w}:{h}:flags=lanczos"
    cw = max(2, int(w / m) & ~1)
    ch = max(2, int(h / m) & ~1)
    cw, ch = min(cw, PLATE_W) & ~1, min(ch, PLATE_H) & ~1
    cx = ((PLATE_W - cw) // 2) & ~1
    cy = ((PLATE_H - ch) // 2) & ~1
    return f"crop={cw}:{ch}:{cx}:{cy},scale={w}:{h}:flags=bicubic"


@st.cache_resource(show_spinner=False, max_entries=2)
def get_plate(preset, w, h):
    """Decode+cache a calibration window of the plate for (preset,w,h).
    Same stats feed preview and export, so what you tune matches what ships.

    max_entries=2 matters as much as the byte budget: this cache is
    process-wide (shared across every session) and never expired on its
    own, so trying a couple of grain-size presets without a cap would keep
    every one of them resident — that's what actually exhausted the 1GB
    container, not any single allocation.
    """
    vf = plate_geometry(preset, w, h)
    n = frames_for_budget(PLATE_CAL_BUDGET, w, h, min_frames=4, max_frames=64)
    buf = decode_window(PLATE, PLATE_SS, n, w, h, vf=f"{vf},format=yuv420p")
    py, _, _ = planes(buf, w, h)
    mean, sigma = mean_std(py)
    return py, mean, sigma


def encode_png_bytes(frame, w, h):
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
         "-s", f"{w}x{h}", "-i", "-", "-frames:v", "1", "-f", "image2",
         "-vcodec", "png", "-"],
        input=np.ascontiguousarray(frame).tobytes(), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode(errors="replace")[-2000:])
    return p.stdout


def export_full(path, info, preset, k, weighted, progress):
    """Full-length export.

    Draws grain content from get_plate()'s already-decoded, budget-capped
    array (looped via modulo) instead of running a second live
    `-stream_loop` ffmpeg process — that cut a whole concurrent ffmpeg
    process out of export, which mattered more for staying inside
    Community Cloud's 1GB container than any single buffer size did (see
    the x264-params below for the other half of that fix: default x264
    lookahead/reference-frame buffering alone was costing several hundred
    MB per encoder instance at 4K). It also means preview and export now
    draw from identical plate content, not just similarly-calibrated ones.
    """
    w, h, fps = info["width"], info["height"], info["fps"]
    dur, has_audio = info["duration"], info["has_audio"]
    py_all, mean, sigma = get_plate(preset, w, h)
    plate_n = py_all.shape[0]
    total_frames = max(1, round(dur * fps))

    fd, out_path = tempfile.mkstemp(suffix="_grain.mp4")
    os.close(fd)
    src_p = enc_p = None
    try:
        src_p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-threads", "2", "-i", path,
             "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"], stdout=subprocess.PIPE)

        enc_args = ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "-"]
        if has_audio:
            enc_args += ["-i", path, "-map", "0:v:0", "-map", "1:a:0?"]
        enc_args += ["-c:v", "libx264", "-preset", "fast", "-tune", "grain",
                     "-crf", "18", "-pix_fmt", "yuv420p",
                     "-x264-params", "rc-lookahead=10:ref=1:bframes=0",
                     "-threads", "2"]
        if has_audio:
            enc_args += ["-c:a", "aac", "-b:a", "192k"]
        enc_args += ["-movflags", "+faststart", "-shortest", out_path, "-y"]
        enc_p = subprocess.Popen(enc_args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

        fsz = frame_size(w, h)
        chunk_frames = frames_for_budget(EXPORT_CHUNK_BUDGET, w, h, min_frames=1, max_frames=8)
        done = 0
        while True:
            sbuf = read_exact(src_p.stdout, fsz * chunk_frames)
            got = len(sbuf) // fsz
            if got == 0:
                break
            sbuf = sbuf[:got * fsz]
            sarr = np.frombuffer(sbuf, dtype=np.uint8).reshape(got, fsz)
            sy, su, sv = planes(sarr, w, h)
            py = py_all[np.arange(done, done + got) % plate_n]
            oy = composite(sy, py, mean, sigma, k, weighted)
            enc_p.stdin.write(pack(oy, su, sv).tobytes())

            done += got
            progress(min(0.999, done / total_frames))
            if got < chunk_frames:
                break

        src_p.stdout.close()
        enc_p.stdin.close()
        src_p.wait(timeout=30)
        enc_err = enc_p.stderr.read()
        if enc_p.wait(timeout=180) != 0:
            raise RuntimeError(enc_err.decode(errors="replace")[-2000:])
        return out_path
    except Exception:
        for p in (src_p, enc_p):
            if p and p.poll() is None:
                p.kill()
        raise


# ---- Streamlit UI ---------------------------------------------------------

st.set_page_config(page_title="Grain Studio", page_icon="🎞️", layout="wide")
st.title("Grain Studio")
st.caption("Same calibrated 35mm plate and numpy compositing as the local Grain Bench — import your own video, tune it live, export.")

uploaded = st.file_uploader("Import a video", type=["mp4", "mov", "m4v", "webm", "mkv"])

if uploaded is None:
    st.info("Upload a video to begin.")
    st.stop()

sig = (uploaded.name, uploaded.size)
if st.session_state.get("upload_sig") != sig:
    tmp_dir = tempfile.mkdtemp(prefix="grain_")
    video_path = os.path.join(tmp_dir, uploaded.name)
    with open(video_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.session_state["upload_sig"] = sig
    st.session_state["video_path"] = video_path
    st.session_state.pop("win_key", None)

video_path = st.session_state["video_path"]

try:
    info = ffprobe_info(video_path)
except Exception as e:
    st.error(f"Couldn't read that file: {e}")
    st.stop()

st.caption(f"{info['width']}×{info['height']} · {info['fps']:.2f}fps · "
           f"{info['duration']:.1f}s" + (" · has audio" if info["has_audio"] else " · no audio"))

# Measured directly (see the design conversation this shipped from): 1080p
# exports stayed comfortably under Community Cloud's 1GB container limit
# (peak concurrent memory across every process — the Python app plus every
# concurrently-running ffmpeg process — measured 633-832MB across a couple
# of real test clips); a 4K export measured 1339MB, well over it, mainly
# from x264's own frame buffers at that resolution rather than anything
# this app allocates. Rather than let a larger upload fail with a generic
# crash, say so upfront.
SAFE_PIXELS = 1920 * 1080
if info["width"] * info["height"] > SAFE_PIXELS:
    st.warning(
        f"{info['width']}×{info['height']} is above the ~1080p ceiling this free "
        "tier's 1GB memory limit can reliably handle for export (mainly the video "
        "encoder's own buffers, not this app). It may still work, especially for "
        "shorter clips — but if export fails, downscale the source first.")

col_stage, col_controls = st.columns([2, 1], gap="large")

with col_controls:
    st.subheader("Grain")
    strength = st.slider("Strength", 0, 200, 60, step=5, format="%d%%") / 100.0
    size = st.select_slider("Grain size", options=["native", "1.5x", "2x", "3x"], value="native")
    weighted = st.checkbox("Luma-weighted", value=False,
                            help="Grain rolls off in the blacks and highlights, like emulsion does.")

with col_stage:
    st.subheader("Preview")
    dur = info["duration"]
    scrub_max = max(0.1, dur - 0.5)
    t = st.slider("Scrub", 0.0, scrub_max, min(2.0, scrub_max), step=max(0.1, scrub_max / 200))

    win_key = (video_path, round(t, 1))
    if st.session_state.get("win_key") != win_key:
        win_dur = min(3.0, max(0.5, dur - t)) if dur > 0 else 3.0
        n_wanted = max(1, round(win_dur * info["fps"]))
        n = min(n_wanted, frames_for_budget(PREVIEW_BUDGET, info["width"], info["height"], min_frames=4))
        buf = decode_window(video_path, t, n, info["width"], info["height"])
        sy, su, sv = planes(buf, info["width"], info["height"])
        st.session_state["win_key"] = win_key
        st.session_state["window"] = dict(sy=sy, su=su, sv=sv, w=info["width"],
                                           h=info["height"], fps=info["fps"])

    win_secs = st.session_state["window"]["sy"].shape[0] / st.session_state["window"]["fps"]
    st.caption(f"Preview window: {win_secs:.2f}s "
               f"({st.session_state['window']['sy'].shape[0]} frames)")

    win = st.session_state["window"]
    n = win["sy"].shape[0]
    idx = n // 2
    py_all, mean, sigma = get_plate(size, win["w"], win["h"])
    pidx = idx % py_all.shape[0]
    oy = composite(win["sy"][idx], py_all[pidx], mean, sigma, strength, weighted)
    frame = pack(oy[None], win["su"][idx][None], win["sv"][idx][None])[0]
    st.image(encode_png_bytes(frame, win["w"], win["h"]), width="stretch")

    if st.button("Preview clip"):
        with st.spinner("Rendering clip…"):
            pidx_all = np.arange(n) % py_all.shape[0]
            oy_all = composite(win["sy"], py_all[pidx_all], mean, sigma, strength, weighted)
            out = pack(oy_all, win["su"], win["sv"])
            fd, clip_path = tempfile.mkstemp(suffix=".mp4")
            os.close(fd)
            p = subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-threads", "2", "-f", "rawvideo",
                 "-pix_fmt", "yuv420p", "-s", f"{win['w']}x{win['h']}", "-r", str(win["fps"]),
                 "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                 "-x264-params", "rc-lookahead=10:ref=1:bframes=0",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", clip_path],
                input=np.ascontiguousarray(out).tobytes(), capture_output=True)
            if p.returncode:
                st.error(p.stderr.decode(errors="replace")[-2000:])
            else:
                st.video(clip_path)

st.divider()
st.subheader("Export")
if st.button("Export full video", type="primary"):
    bar = st.progress(0.0, text="Exporting…")
    try:
        out_path = export_full(video_path, info, size, strength, weighted,
                                lambda pct: bar.progress(pct, text=f"Exporting… {pct*100:.0f}%"))
        bar.progress(1.0, text="Done")
        with open(out_path, "rb") as f:
            data = f.read()
        base, _ext = os.path.splitext(uploaded.name)
        st.download_button("Download graded video", data, file_name=f"{base}_grain.mp4",
                            mime="video/mp4")
    except Exception as e:
        st.error(f"Export failed: {e}")
