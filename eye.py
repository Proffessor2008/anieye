"""Eye -> Space — a generative ASCII-art animation.

The scene is built procedurally with NumPy (no image files) and then turned
into text characters: an eye opens, blinks, looks around, dives into the
pupil, flies through a starfield into a heart-shaped nebula and then comes
back out of the pupil, so the whole thing loops.

Three presentation back-ends produce the same picture:

* **gpu** (default when ``moderngl`` + ``glfw`` are installed) — one fragment
  shader draws every character from a glyph atlas, so the per-pixel work
  happens on the GPU.  Runs at 100 FPS by default and stays that fast while
  the window is resized (the grid has a fixed cell budget, extra window area
  magnifies the characters).
* **raster** (Pillow) — the whole character grid is composed into one RGB
  image and blitted to the window as a single bitmap, so Tk never has to lay
  out thousands of text items.  Full 8-bit colour and the CRT overlay.
* **text** (fallback, NumPy only) — one Tk canvas text item per character,
  redrawing only the cells that changed, within a time budget.

Keys: ``R`` restart, ``Space`` pause, ``F`` fullscreen, ``S`` save a frame,
``Esc`` quit; drag the window edges to resize it (the grid re-fits), and the
eye follows the mouse pointer.

Command line::

    python eye.py                        # GPU, 100 FPS, resizable window
    python eye.py --fps 0                # uncapped
    python eye.py --size 14 --fps 60     # bigger window, 60 FPS
    python eye.py --theme ice --seed 12
    python eye.py --once                 # no loop: stop in the nebula
    python eye.py --raster               # force the Tk bitmap renderer
    python eye.py --text                 # force the canvas renderer
"""

import math as m
import os
import random
import sys
import time

import numpy as np


def _cli_value(flag, default, cast=str):
    """Read ``--flag value`` or ``--flag=value`` straight from sys.argv.

    The seed and the palette have to be known before the textures are built at
    import time, i.e. before argparse would normally run.
    """
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == flag and i + 1 < len(argv):
            try:
                return cast(argv[i + 1])
            except ValueError:
                return default
        if arg.startswith(flag + "="):
            try:
                return cast(arg.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _cli_flag(flag):
    return flag in sys.argv[1:]


SIZE = _cli_value("--size", 8, int)
FPS = _cli_value("--fps", 100.0, float)      # 0 = uncapped
SS = 2
ASPECT = 1.40
MAXROWS = _cli_value("--rows", 30, int)      # rows of the first window
MAXROWS_WIN = 36                             # cell budget while the window is resized
TK_MS = 13.0

# Feature switches (CLI: --once turns the loop off, --no-crt the CRT overlay).
FX = {
    "loop": not _cli_flag("--once"),
    "pupil": True,
    "pointer": True,
    "crt": not _cli_flag("--no-crt"),
}

# --- timeline (seconds) -----------------------------------------------------
# The loop is front-loaded: the eye itself gets the longest act (opening, two
# double blinks, a single blink, two light flashes, a handful of glances),
# then the dive into the pupil, a short heart, and the flight back.
T_OPEN = 0.9                             # lids start opening
T_OPEN_D = 2.4                           # how long the opening takes
T_BLINK = 4.0                            # first blink of the first double blink
BLINKS = (4.0, 4.32, 7.55, 7.87, 10.75)  # two double blinks + a single one
FLASHES = (2.35, 8.4)                    # light flashes: the pupil constricts
T_Z0 = 12.0                              # the eye act ends, the camera dives
T_ZD = 16.0                              # duration of the dive
ZEND = 14.0                              # final zoom (the pupil fills the frame)
T_SPACE = T_Z0 + T_ZD                    # 28.0 - fully inside the pupil
T_HEART = T_Z0 + 2.5                     # 14.5 - the nebula starts becoming a heart
T_HEART_D = 10.0                         # ... and the heart is done at 24.5
T_RET = T_SPACE + 3.5                    # 31.5 - fly back out of the pupil
T_RET_D = 4.5                            # duration of the flight back
T_CLOSE = 36.0                           # lids close again before the loop point
T_LOOP = 37.5                            # animation period when FX["loop"] is on

WO = 0.78
HU, HL = 0.30, 0.20
RI = 0.27
RP0 = 0.098
GS0 = 0.085
GS_MAX = 1.5

RAMP = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NR = len(RAMP) - 1
CIDX = {c: i for i, c in enumerate(RAMP)}
QL = 24

# "auto" tries the GPU first, then the bitmap renderer, then the text one.
USE_BACKEND = ("gpu" if _cli_flag("--gpu")
               else "text" if _cli_flag("--text")
               else "raster" if _cli_flag("--raster")
               else "auto")
GLYPH_DY = 1.0              # baseline nudge, calibrated against Tk's text layout

SEED = _cli_value("--seed", 5, int)
THEME = _cli_value("--theme", "amber")

rng = np.random.RandomState(SEED)
prnd = random.Random(SEED + 4)

TAU = 2.0 * m.pi
BG_RGB = (1, 2, 3)                       # canvas "#010203"
SPACE_BG = np.array([0.004, 0.005, 0.012], np.float32)

# --- palette constants (hoisted out of the per-frame code) -------------------
AMBER = np.array([0.80, 0.45, 0.10], np.float32)
RUST = np.array([0.42, 0.20, 0.06], np.float32)
RUST_GLOW = np.array([0.22, 0.11, 0.00], np.float32)
SCLERA = np.array([0.80, 0.82, 0.86], np.float32)
SCLERA_WARM = np.array([0.86, 0.50, 0.46], np.float32)
VEIN = np.array([0.62, 0.15, 0.14], np.float32)
CATCHLIGHT = np.array([1.0, 0.98, 0.95], np.float32)
SKIN = np.array([0.62, 0.40, 0.29], np.float32)
LID_HL = np.array([0.68, 0.40, 0.31], np.float32)
BROW = np.array([0.10, 0.06, 0.04], np.float32)
LID_LINE = np.array([0.05, 0.03, 0.022], np.float32)
WATERLINE = np.array([0.74, 0.34, 0.30], np.float32)
GLOW = np.array([1.0, 0.85, 0.75], np.float32)
SPECK = np.array([1.0, 0.9, 0.8], np.float32)
CORNER = np.array([0.82, 0.42, 0.38], np.float32)
NEB_DEEP = np.array([0.92, 0.16, 0.40], np.float32)
NEB_LIGHT = np.array([1.0, 0.58, 0.78], np.float32)
NEB_HALO = np.array([0.78, 0.10, 0.30], np.float32)
NEB_RED = np.array([0.95, 0.16, 0.34], np.float32)
STAR_COOL = np.array([0.70, 0.82, 1.0], np.float32)
STAR_WARM = np.array([1.0, 0.80, 0.55], np.float32)

STREAK_COL = np.array([0.88, 0.94, 1.0], np.float32)
COMET_COL = np.array([0.70, 0.86, 1.0], np.float32)

UP_ROOT = np.array([0.16, 0.10, 0.07], np.float32)
UP_TIP = np.array([0.30, 0.19, 0.13], np.float32)
LO_ROOT = np.array([0.17, 0.11, 0.08], np.float32)
LO_TIP = np.array([0.30, 0.20, 0.14], np.float32)

CIDX_DOT = CIDX["."]
CIDX_PLUS = CIDX["+"]
CIDX_STAR = CIDX["*"]
CIDX_AT = CIDX["@"]
CIDX_DASH = CIDX["-"]
CIDX_BAR = CIDX["|"]

# --- palette themes ---------------------------------------------------------
# Only the "accent" colours are themed (iris, veins, nebula, stars); the skin
# and the sclera stay natural, so it looks like different eyes rather than a
# colour filter.  "noir" desaturates the finished frame instead.
_ACCENTS = {
    "amber": {},
    "ice": {
        "iris_in": (0.26, 0.60, 0.88), "iris_out": (0.08, 0.22, 0.42),
        "iris_glow": (0.02, 0.12, 0.24), "vein": (0.32, 0.44, 0.60),
        "waterline": (0.34, 0.48, 0.66), "lid_hl": (0.44, 0.52, 0.64),
        "neb_deep": (0.14, 0.42, 0.92), "neb_light": (0.60, 0.88, 1.0),
        "neb_halo": (0.08, 0.26, 0.78), "neb_red": (0.18, 0.54, 0.96),
        "star_cool": (0.72, 0.88, 1.0), "star_warm": (0.95, 0.99, 1.0),
    },
    "emerald": {
        "iris_in": (0.24, 0.66, 0.34), "iris_out": (0.06, 0.26, 0.12),
        "iris_glow": (0.02, 0.16, 0.06), "vein": (0.34, 0.52, 0.34),
        "waterline": (0.42, 0.60, 0.44), "lid_hl": (0.48, 0.58, 0.46),
        "neb_deep": (0.10, 0.60, 0.36), "neb_light": (0.62, 1.0, 0.76),
        "neb_halo": (0.06, 0.42, 0.24), "neb_red": (0.16, 0.78, 0.46),
        "star_cool": (0.76, 1.0, 0.88), "star_warm": (1.0, 0.96, 0.72),
    },
    "violet": {
        "iris_in": (0.56, 0.34, 0.86), "iris_out": (0.22, 0.10, 0.40),
        "iris_glow": (0.10, 0.02, 0.22), "vein": (0.46, 0.34, 0.60),
        "waterline": (0.56, 0.42, 0.66), "lid_hl": (0.56, 0.46, 0.64),
        "neb_deep": (0.56, 0.20, 0.90), "neb_light": (0.88, 0.66, 1.0),
        "neb_halo": (0.36, 0.10, 0.72), "neb_red": (0.72, 0.26, 0.96),
        "star_cool": (0.84, 0.80, 1.0), "star_warm": (1.0, 0.86, 0.98),
    },
}

MONO = THEME.lower() == "noir"


def _rgb(*v):
    return np.array(v, np.float32)


def _apply_theme(name):
    """Rebind the accent colours for the chosen palette."""
    global AMBER, RUST, RUST_GLOW, VEIN, WATERLINE, LID_HL
    global NEB_DEEP, NEB_LIGHT, NEB_HALO, NEB_RED, STAR_COOL, STAR_WARM
    acc = _ACCENTS.get(name.lower())
    if not acc:
        return name.lower()
    AMBER = _rgb(*acc["iris_in"])
    RUST = _rgb(*acc["iris_out"])
    RUST_GLOW = _rgb(*acc["iris_glow"])
    VEIN = _rgb(*acc["vein"])
    WATERLINE = _rgb(*acc["waterline"])
    LID_HL = _rgb(*acc["lid_hl"])
    NEB_DEEP = _rgb(*acc["neb_deep"])
    NEB_LIGHT = _rgb(*acc["neb_light"])
    NEB_HALO = _rgb(*acc["neb_halo"])
    NEB_RED = _rgb(*acc["neb_red"])
    STAR_COOL = _rgb(*acc["star_cool"])
    STAR_WARM = _rgb(*acc["star_warm"])
    return name.lower()


THEME = _apply_theme(THEME)


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def zp(t):
    return ease((t - T_Z0) / T_ZD)


def mix(a, b, k):
    return a + (b - a) * k


def _up2(b, n):
    g = b.shape[0]
    idx = np.arange(n) * g / n
    i0 = idx.astype(np.int64) % g
    i1 = (i0 + 1) % g
    f = idx - np.floor(idx)
    f = (f * f * (3 - 2 * f)).astype(np.float32)
    t = b[:, i0] * (1 - f) + b[:, i1] * f
    return t[i0, :] * (1 - f)[:, None] + t[i1, :] * f[:, None]


def _norm(n, k=4.0):
    return np.clip(0.5 + (n - n.mean()) / (n.std() * k + 1e-9), 0, 1).astype(np.float32)


def tex2d(n, grids):
    out = np.zeros((n, n), np.float32)
    amp = 1.0
    for g in grids:
        out += amp * _up2(rng.rand(g, g).astype(np.float32), n)
        amp *= 0.55
    return _norm(out)


def noise1d(n, g):
    base = rng.rand(g)
    idx = np.arange(n) * g / n
    i0 = idx.astype(np.int64) % g
    i1 = (i0 + 1) % g
    f = idx - np.floor(idx)
    f = f * f * (3 - 2 * f)
    return (base[i0] * (1 - f) + base[i1] * f).astype(np.float32)


def _pad_wrap(tex):
    """Append the first row/column so bilinear taps never need wrap-around."""
    n = tex.shape[0]
    out = np.empty((n + 1, n + 1), np.float32)
    out[:n, :n] = tex
    out[n, :n] = tex[0]
    out[:n, n] = tex[:, 0]
    out[n, n] = tex[0, 0]
    return out


def bil(tex, u, v):
    """Bilinear sample of a padded texture at (u, v).

    Textures carry one wrapped row/column (see `_pad_wrap`), so the four texels
    of a sample are simply base, base+1, base+stride and base+stride+1: one
    index array, no integer modulo and a single flat gather per tap.  That is
    ~1.5x faster than the classic modulo version and gives identical results.
    """
    n = tex.shape[0] - 1            # real texture size (always a power of two)
    msk = n - 1
    stride = n + 1
    uf = np.floor(u)
    vf = np.floor(v)
    fu = u - uf
    fv = v - vf
    i0 = uf.astype(np.int32) & msk
    j0 = vf.astype(np.int32) & msk
    base = j0 * stride + i0
    flat = tex.reshape(-1)
    a = np.take(flat, base)
    b = np.take(flat, base + 1)
    c = np.take(flat, base + stride)
    d = np.take(flat, base + stride + 1)
    wu = 1 - fu
    wv = 1 - fv
    lo = a * wu + b * fu
    hi = c * wu + d * fu
    return lo * wv + hi * fv


def _rgb3(m):
    """Copy a 2-D field into all three colour channels.

    Colour maths written as ``field[..., None] * rgb_array`` makes NumPy walk
    the innermost (3 element) axis pixel by pixel, which is ~30x slower than a
    contiguous multiply.  Expanding the field once is far cheaper.
    """
    out = np.empty(m.shape + (3,), np.float32)
    out[..., 0] = m
    out[..., 1] = m
    out[..., 2] = m
    return out


def _block_mean(img, f):
    """Average f x f blocks (strided slices are much faster than reshape+mean)."""
    out = img[0::f, 0::f].copy()
    for i in range(f):
        for j in range(f):
            if i or j:
                out += img[i::f, j::f]
    out *= 1.0 / (f * f)
    return out


def _downsample(img):
    return _block_mean(img, SS)


print("Preparing textures...")
TS = _pad_wrap(tex2d(1024, (6, 20, 64, 200)))
TG = _pad_wrap(tex2d(1024, (8, 24, 64, 160)))
TW = _pad_wrap(rng.rand(512, 512).astype(np.float32))
_tv = tex2d(512, (5, 12, 30))
TV = _pad_wrap(1 - np.abs(2 * _tv - 1))
_tv2 = tex2d(512, (3, 7, 16))
TV2 = _pad_wrap(1 - np.abs(2 * _tv2 - 1))

NA = 4096
A1 = _norm(0.5 * noise1d(NA, 900) + 0.3 * noise1d(NA, 300) + 0.2 * noise1d(NA, 120), 3.0)
A2 = _norm(0.5 * noise1d(NA, 700) + 0.3 * noise1d(NA, 200) + 0.2 * noise1d(NA, 80), 3.0)
A3 = _norm(noise1d(NA, 13), 3.0)
A4 = _norm(noise1d(NA, 48), 3.0)

NS = 3200
SX = rng.uniform(-1, 1, NS).astype(np.float32)
SY = rng.uniform(-1, 1, NS).astype(np.float32)
SD = rng.uniform(0.06, 1.0, NS).astype(np.float32)
SB = (0.10 + 0.90 * rng.rand(NS) ** 3.5).astype(np.float32)
_tm = rng.rand(NS, 1).astype(np.float32)
SCOL = mix(STAR_COOL, STAR_WARM, _tm ** 1.5)
SPH = rng.uniform(0, 6.28, NS).astype(np.float32)
SF = rng.uniform(1.0, 3.5, NS).astype(np.float32)

GZ = [(0.0, 0.0, 0.0)]
_tc = 1.7
_gr = random.Random(4)
while _tc < 60:
    GZ.append((_tc, _gr.uniform(-0.045, 0.045), _gr.uniform(-0.022, 0.02)))
    _tc += _gr.uniform(0.9, 2.4)
GZ.append((T_Z0 - 0.8, 0.0, 0.0))     # look straight ahead just before the dive
GZ.sort(key=lambda g: g[0])

_lr = random.Random(21)


def _prep_lash(s0, lf_):
    """Everything about a lash that does not change from frame to frame."""
    edge = m.sqrt(max(0.0, 1 - s0 * s0))
    a_open = m.radians(-90 + 26 * s0)
    a_closed = m.radians(90 - 20 * s0) if s0 >= 0 else m.radians(-270 - 20 * s0)
    return (s0, s0 * WO, lf_, a_open, a_closed, m.radians(-22 * s0),
            (0.070 + 0.050 * edge) * lf_)


NUP = 44
UP_L = [_prep_lash(-0.92 + 1.84 * i / (NUP - 1), _lr.uniform(0.92, 1.08)) for i in range(NUP)]
NLO = 20
LO_L = [_prep_lash(-0.70 + 1.40 * i / (NLO - 1), _lr.uniform(0.90, 1.10)) for i in range(NLO)]


def grid_for(width, height, cw, ch, cap=None):
    """Largest aspect-correct grid of cw x ch cells that fits in width x height.

    Keeps ASPECT so the picture is never stretched; the caller centres the
    resulting W x H area inside the window.
    """
    cap = max(MAXROWS_WIN, MAXROWS) if cap is None else cap
    rows = int(max(2, height // ch))
    cols = int(rows * ch * ASPECT / cw) // 2 * 2
    if cols * cw > width:
        cols = int(max(4, width // cw)) // 2 * 2
        rows = int(max(2, cols * cw / ASPECT / ch))
    if rows > cap:
        rows = cap
        cols = int(rows * ch * ASPECT / cw) // 2 * 2
    return max(2, rows // 2 * 2), max(4, cols)


def init(cw, ch, rows, aspect=None, reset_state=True):
    global CW, CH, ROWS, COLS, W, H, GW, GH, X, Y, RS, XC, YC, RSC, HSC, SPIMAGE
    CW, CH = cw, ch
    aspect = ASPECT if aspect is None else aspect
    COLS = int(rows * ch * aspect / cw) // 2 * 2
    ROWS = rows // 2 * 2
    W, H = COLS * CW, ROWS * CH
    GW, GH = COLS * SS, ROWS * SS
    xs = (np.arange(GW, dtype=np.float32) + 0.5) * (CW / SS)
    ys = (np.arange(GH, dtype=np.float32) + 0.5) * (CH / SS)
    X = np.broadcast_to(((xs - W / 2) / H)[None, :], (GH, GW)).astype(np.float32)
    Y = np.broadcast_to(((ys - H / 2) / H)[:, None], (GH, GW)).astype(np.float32)
    RS = np.sqrt(X * X + Y * Y)
    cx = (np.arange(COLS, dtype=np.float32) + 0.5) * CW
    cy = (np.arange(ROWS, dtype=np.float32) + 0.5) * CH
    XC = np.broadcast_to(((cx - W / 2) / H)[None, :], (ROWS, COLS)).astype(np.float32)
    YC = np.broadcast_to(((cy - H / 2) / H)[:, None], (ROWS, COLS)).astype(np.float32)
    RSC = np.sqrt(XC * XC + YC * YC)
    rr, cc = np.mgrid[0:ROWS, 0:COLS]
    HSC = (((rr * 73856093) ^ (cc * 19349663) ^ 0x5bd1e995) % 1009 / 1009.0).astype(np.float32)
    SPIMAGE = np.empty((GH, GW, 3), np.float32)
    if reset_state:
        reset()


def blink_c(t, times=None):
    """Eyelid closure from one or several blinks (0 = open, 1 = closed)."""
    if times is None:
        times = BLINKS
    best = 0.0
    for tb in times:
        d = t - tb
        if d < 0 or d > 0.36:
            continue
        if d < 0.09:
            k = ease(d / 0.09)
        elif d < 0.12:
            k = 1.0
        else:
            k = 1 - ease((d - 0.12) / 0.24)
        if k > best:
            best = k
    return best


def lid_open(t):
    k = ease((t - T_OPEN) / T_OPEN_D)
    k = k * (1 + 0.035 * m.sin(m.pi * k))
    op = min(1.0, k) * (1 - 0.985 * blink_c(t))
    if FX["loop"] and t > T_CLOSE:
        op *= 1.0 - ease((t - T_CLOSE) / max(0.2, T_LOOP - T_CLOSE - 0.3))
    return op


def zoom_amount(t):
    """Camera zoom at animation time t (1 = whole eye, ZEND = deep in the pupil)."""
    base = m.exp(m.log(ZEND) * zp(t))
    if FX["loop"] and t > T_RET:
        # fly back out: exponential interpolation from ZEND towards 1
        base = m.exp(m.log(ZEND) * (1.0 - ease((t - T_RET) / T_RET_D)))
    idle = ease(t / T_Z0)
    return base * (1 + 0.05 * idle + 0.004 * m.sin(t * 0.9))


def star_spread(t, z):
    """How far the starfield is spread out (stars rush past while flying in)."""
    if t < T_SPACE:
        return z
    spread = ZEND * m.exp(0.03 * (min(t, T_RET) - T_SPACE))
    if FX["loop"] and t > T_RET:
        spread = mix(spread, z, ease((t - T_RET) / T_RET_D))
    return spread


def pupil_scale(t):
    """Pupil constriction after each light flash, then recovery."""
    if not FX["pupil"]:
        return 1.0
    k = 1.0
    for tf in FLASHES:
        d = t - tf
        if d < -0.15 or d > 1.8:
            continue
        if d < 0.0:
            s = 1.0 - 0.10 * ease((d + 0.15) / 0.15)
        else:
            s = 1.0 - 0.34 * m.exp(-((d / 0.42) ** 2))
        k = min(k, s)
    return k


POINTER = {"active": False, "x": 0.0, "y": 0.0, "gx": 0.0, "gy": 0.0}
GAZE_GAIN = 0.075       # how far the gaze follows the pointer, in eye units


def pointer_gaze(t, dt):
    """Gaze that smoothly follows the mouse pointer (eye-space units)."""
    amp = 1 - ease((t - T_Z0) / 2.5)
    tx = max(-1.0, min(1.0, POINTER["x"])) * GAZE_GAIN * amp
    ty = max(-1.0, min(1.0, POINTER["y"])) * GAZE_GAIN * amp
    k = 1.0 - m.exp(-max(dt, 0.0) / 0.07)
    POINTER["gx"] += (tx - POINTER["gx"]) * k
    POINTER["gy"] += (ty - POINTER["gy"]) * k
    return POINTER["gx"], POINTER["gy"]


def gaze(t, prog):
    prev = (0.0, 0.0)
    cur = (0.0, 0.0)
    te = 0.0
    for (t_, gx_, gy_) in GZ:
        if t_ <= t:
            prev = cur
            cur = (gx_, gy_)
            te = t_
        else:
            break
    k = ease((t - te) / 0.09)
    gx = prev[0] + (cur[0] - prev[0]) * k
    gy = prev[1] + (cur[1] - prev[1]) * k
    gx += 0.003 * m.sin(t * 1.3)
    gy += 0.002 * m.sin(t * 1.1 + 1)
    amp = 1 - ease((t - T_Z0) / 2.5)
    gy -= 0.03 * blink_c(t)
    return gx * amp, gy * amp


def lid_edges(u0s, ou):
    """Position of the upper/lower lid edge for one or many lash roots."""
    s = u0s / WO
    wu = np.where(s < -0.12, 0.88, 1.12)
    cu = np.maximum(0.0, 1 - ((s + 0.12) / wu) ** 2) ** 0.8
    cl = np.maximum(0.0, 1 - (s - 0.08) ** 2) ** 0.9
    vc = 0.03 * cl
    vl = vc + (HL * cl - vc) * (0.35 + 0.65 * ou)
    vu = vl + (-HU * cu - vl) * ou
    return vu, vl, cl


def heart_nebula(xs, ys, sz, hm, t):
    hx = xs / sz
    hy = -ys / sz

    nC = bil(TG, hx * 16 + 400, hy * 16 + 100)
    nD = bil(TG, hx * 16 + 90, hy * 16 + 530)
    nA = bil(TG, hx * 42 + 300, hy * 42 + 700 + t * 0.8)
    nB = bil(TG, hx * 110 + 80, hy * 110 + 200)

    wob = 1 - 0.65 * hm
    cp = wob * 0.40
    na = nA - 0.5
    hxw = hx + cp * (nC - 0.5) + 0.05 * na
    hyw = hy + cp * (nD - 0.5) + 0.05 * na

    hxx = hxw * hxw
    qc = np.sqrt(hxx + hyw * hyw) / 0.80 - 1.0

    hyh = hyw + 0.12
    fh = (hxx + hyh * hyh - 1.0) ** 3 - hxx * hyh ** 3
    qh = 0.5 * np.cbrt(fh)
    q = qc * (1 - hm) + qh * hm

    qpos = np.clip(q, 0, 10)
    inside = q < 0
    outside = q > 0
    ins = np.clip(-q / (0.80 - 0.35 * hm), 0, 1)
    k = 3.0 + 2.5 * hm
    body = np.where(inside, 0.50 + 0.50 * np.sqrt(ins), np.exp(-qpos * k) * 0.50)

    tex = 1 + (0.62 + 0.80 * nB - 1) * 0.8
    col = NEB_DEEP + (NEB_LIGHT - NEB_DEEP) * _rgb3(ins ** 0.8)
    col *= _rgb3(body * tex)

    halo = np.exp(-qpos * 1.6) * 0.20 * (0.5 + nA)
    col += _rgb3(halo * outside) * NEB_HALO

    rim = np.exp(-((q + 0.04) / 0.14) ** 2)
    rid = np.clip((bil(TV, hxw * 120 + 50, hyw * 120 + 300 + t * 0.6) - 0.86) * 7, 0, 1)
    col += _rgb3(rim * (0.20 + 0.9 * rid) * (0.5 + nA)
                 + rid * np.clip(-q, 0, 1) * 0.25) * NEB_RED
    return col


def eye_scene(t, z, ou, gx, gy, Rp, sp):
    iz = 1.0 / z
    px = iz / GH
    u = X * iz
    v = Y * iz
    near = z < 3.4
    pm_u = u - gx
    pm_v = v - gy
    r = np.sqrt(pm_u * pm_u + pm_v * pm_v)
    angf = (np.arctan2(pm_v, pm_u) / TAU) % 1.0
    ia = (angf * NA).astype(np.int32) & (NA - 1)

    rn = np.clip((r - Rp) / (RI - Rp), 0, 1.3)
    rb = rn * 14
    i0 = np.floor(rb).astype(np.int32)
    fr = rb - i0.astype(np.float32)        # int32 subtraction would upcast to float64
    fib = 0.55 * A1[ia] + 0.45 * (A2[(ia + i0 * 131) & (NA - 1)] * (1 - fr)
                                 + A2[(ia + (i0 + 1) * 131) & (NA - 1)] * fr)

    zig = np.abs((angf * 34) % 1.0 - 0.5) * 2 - 0.5
    cr = 0.34 + 0.20 * (A3[ia] - 0.5) + 0.035 * zig

    ir = AMBER + (RUST - AMBER) * _rgb3(ss(-0.10, 0.22, rn - cr))
    ir *= _rgb3(0.45 + 1.05 * fib)
    ir += _rgb3(np.exp(-((rn - cr) / 0.07) ** 2) * (0.2 + fib)) * RUST_GLOW
    cry = (A4[ia] > 0.72) * ss(0.65, 0.18, rn)
    ir *= _rgb3(1 - 0.55 * cry)
    ir *= _rgb3(1 - 0.82 * ss(0.62, 1.0, rn))
    ir *= _rgb3(0.55 + 0.45 * ss(0, 0.07, rn))

    if near:
        s = u / WO
        sa = np.clip(s, -1.3, 1.3)
        wu = np.where(sa < np.float32(-0.12), np.float32(0.88), np.float32(1.12))
        cu = np.clip(1 - ((sa + 0.12) / wu) ** 2, 0, 1) ** 0.8
        cl = np.clip(1 - (sa - 0.08) ** 2, 0, 1) ** 0.9
        vt = v + 0.055 * s
        vc = 0.03 * cl
        vl = vc + (HL * cl - vc) * (0.35 + 0.65 * ou)
        vu = vl + (-HU * cu - vl) * ou
        du_in = vt - vu
        dl_in = vl - vt
        d_in = np.minimum(du_in, dl_in)
        m_in = np.clip(d_in / (1.6 * px) + 0.5, 0, 1)
        ir *= _rgb3(0.30 + 0.70 * ss(0, 0.16, du_in))

        cw = ss(0.40, 0.95, np.abs(s))
        sh = (0.38 + 0.62 * ss(0, 0.22, du_in)) * (0.72 + 0.28 * ss(0, 0.10, dl_in)) * (1 - 0.30 * cw)
        sh3 = _rgb3(sh)
        scl = SCLERA * sh3
        scl += (SCLERA_WARM * sh3 - scl) * _rgb3(cw * 0.35)
        vein = np.clip((bil(TV2, u * 85 + 40, v * 85 + 70) - 0.925) * 14, 0, 1) * \
            (0.25 + 0.75 * cw) * ss(0.12, 0.32, np.abs(pm_u))
        scl += (VEIN - scl) * _rgb3(vein * 0.65)
        inside = scl + (ir - scl) * _rgb3(np.clip((RI - r) / (1.6 * px) + 0.5, 0, 1))
    else:
        inside = ir

    Rps = Rp * z
    vig = 0.25 + 0.75 * ss(0, 0.30 * Rps, Rps - RS)
    pm = np.clip((Rp - r) / (1.6 * px) + 0.5, 0, 1)
    inside += (sp * _rgb3(vig) - inside) * _rgb3(pm)

    fade = 1 - (z - 1.6) / 2.4
    if fade > 0:
        fade = min(1.0, fade)
        dd = np.sqrt((pm_u - 0.075) ** 2 + (pm_v + 0.040) ** 2)
        win = np.clip(1 - (dd - 0.030) / (1.4 * px + 0.004), 0, 1)
        dd2 = np.sqrt((pm_u + 0.045) ** 2 + (pm_v - 0.050) ** 2)
        win2 = np.clip(1 - (dd2 - 0.009) / (1.4 * px + 0.003), 0, 1) * 0.5
        inside += _rgb3((win + win2) * fade) * (CATCHLIGHT * 1.4)

    if not near:
        return inside, pm

    if z < 2.6:
        mid = 0.5 * (vu + vl)
        upper = vt < mid
        du = np.maximum(vu - vt, 0)
        dl = np.maximum(vt - vl, 0)
        dd = np.where(upper, du, dl)
        tx = bil(TS, u * 120 + 100, v * 120 + 100)
        skin = SKIN * _rgb3(0.90 + 0.20 * tx)
        sh = 0.50 + 0.50 * ss(0, 0.30, dd)
        sh *= 1 + 0.10 * np.exp(-((v + 0.20) / 0.09) ** 2)
        cr_pos = 0.075 + 0.03 * (1 - ou)
        sh *= 1 - 0.42 * np.exp(-((du - cr_pos) / 0.020) ** 2) * upper
        sh *= 1 + 0.08 * np.exp(-((dl - 0.09) / 0.04) ** 2) * (~upper)
        sh3 = _rgb3(sh)
        skin *= sh3
        lidzone = ss(cr_pos, cr_pos * 0.4, du) * upper
        skin += (LID_HL * sh3 - skin) * _rgb3(lidzone * 0.5)

        sb = np.clip(1 - np.abs(s), 0, 1)
        eb = np.exp(-((vt + 0.42 + 0.10 * s * s) / (0.040 + 0.012 * sb)) ** 2)
        hair = 0.55 + 0.9 * bil(TW, u * 90, v * 30)
        skin += (BROW - skin) * _rgb3(np.clip(eb * hair, 0, 0.92))

        line = np.exp(-(du / 0.010) ** 2) * upper
        skin += (LID_LINE - skin) * _rgb3(np.clip(line * (1.0 - 0.3 * ou) * 0.9, 0, 1))
        wl = ss(0.014, 0.0, dl) * (~upper)
        skin += (WATERLINE - skin) * _rgb3(wl * 0.7)
        gl = np.exp(-((dl - 0.016) / 0.006) ** 2) * (~upper) * \
            (0.35 + 0.65 * bil(TW, u * 60, v * 60))
        skin += _rgb3(gl * 0.55) * (GLOW * ss(0.12, 0.5, ou))
        sp_ = (bil(TW, u * 150 + 7, v * 150 + 3) > 0.985) * 0.35 * (0.4 + tx) * ss(0.0, 0.3, dd)
        skin += _rgb3(sp_) * SPECK
        col = skin + (inside - skin) * _rgb3(m_in)

        cg = np.exp(-(((u + WO * 0.90) / 0.045) ** 2 + ((vt - 0.025) / 0.032) ** 2)) * \
            ss(0.1, 0.5, ou)
        col += (CORNER - col) * _rgb3(cg * 0.85)
        return col, pm
    return inside, pm


streaks = []
S = {"last": 0.0, "next_shoot": 1.0e9, "comet": False, "tlocal": 0.0}


class Streak:
    def __init__(self, x, y, vx, vy, life, tail, col, soft=False):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life, self.tail, self.col, self.soft = life, tail, col, soft
        self.age = 0.0


def reset():
    S.update(last=0.0, next_shoot=1.0e9, comet=False, tlocal=0.0)
    streaks.clear()
    SD[:] = rng.uniform(0.06, 1.0, NS).astype(np.float32)
    POINTER.update(gx=0.0, gy=0.0)


def update_world(t, dt, prog):
    vf = 0.010 + 0.05 * prog + (0.015 if t > T_SPACE else 0.0)
    SD[:] -= vf * dt
    dead = SD < 0.06
    if dead.any():
        SD[dead] += 0.94
        n = int(dead.sum())
        SX[dead] = rng.uniform(-1, 1, n)
        SY[dead] = rng.uniform(-1, 1, n)
    if S["next_shoot"] > 1e8 and t > T_SPACE - 4.0:
        S["next_shoot"] = t + 1.0
    if t > S["next_shoot"]:
        S["next_shoot"] = t + prnd.uniform(2.2, 4.5)
        ang = m.radians(prnd.uniform(18, 62))
        sgn = prnd.choice((-1, 1))
        spd = prnd.uniform(0.7, 1.2)
        streaks.append(Streak(prnd.uniform(-0.7, 0.7), prnd.uniform(-0.45, -0.15),
                              sgn * m.cos(ang) * spd, m.sin(ang) * spd,
                              prnd.uniform(0.9, 1.4), prnd.uniform(0.14, 0.26), STREAK_COL))
    if t > T_SPACE + 1.5 and not S["comet"]:
        S["comet"] = True
        streaks.append(Streak(-0.95, -0.28, 0.030, 0.008, 70.0, 0.22, COMET_COL, soft=True))
    for s_ in streaks:
        s_.age += dt
    streaks[:] = [s_ for s_ in streaks if s_.age < s_.life]


def put(idx, fg, r, c, ch, color, a=1.0):
    if 0 <= r < ROWS and 0 <= c < COLS:
        idx[r, c] = CIDX[ch]
        fg[r, c] = color * a


def stamp_streak(s_, idx, fg, pmc, vigc):
    hx = s_.x + s_.vx * s_.age
    hy = s_.y + s_.vy * s_.age
    sp = m.hypot(s_.vx, s_.vy)
    dx, dy = s_.vx / sp, s_.vy / sp
    fade = m.sin(m.pi * min(1.0, s_.age / s_.life)) ** 0.7
    if fade < 0.02:
        return
    if abs(dy) < 0.38 * abs(dx):
        tc = "-"
    elif abs(dx) < 0.38 * abs(dy):
        tc = "|"
    elif dx * dy > 0:
        tc = "\\"
    else:
        tc = "/"
    n = max(6, int(s_.tail * H / (CW * 0.8)))
    w2h = W / 2
    h2h = H / 2
    tail = s_.tail
    soft = s_.soft
    col = s_.col
    for k in range(n):
        f = k / n
        c = int((w2h + (hx - dx * tail * f) * H) / CW)
        r = int((h2h + (hy - dy * tail * f) * H) / CH)
        b = (1 - f) ** 1.3 * fade
        if b < 0.08 or not (0 <= r < ROWS and 0 <= c < COLS) or pmc[r, c] < 0.5:
            continue
        if k == 0:
            ch = "@" if not soft else "*"
        elif soft:
            ch = ":" if b > 0.35 else "."
        else:
            ch = tc if b > 0.30 else "."
        put(idx, fg, r, c, ch, col, min(1.0, 0.3 + b) * vigc[r, c])


def stamp_stars(idx, fg, L, t, gs, amt, pmc, vigc, offx, offy):
    if amt <= 0.003:
        return
    d = SD
    c = ((W * 0.5 + (SX * gs / d + offx) * H) / CW).astype(np.int32)
    r = ((H * 0.5 + (SY * gs / d + offy) * H) / CH).astype(np.int32)
    alpha = ss(1.0, 0.82, d) * (0.80 + 0.20 * np.sin(t * SF + SPH)) * amt
    ok = (c >= 0) & (c < COLS) & (r >= 0) & (r < ROWS) & (alpha > 0.04)
    if not ok.any():
        return
    ii = np.flatnonzero(ok)
    c, r = c[ii], r[ii]
    b = SB[ii] * alpha[ii] * (1 + 0.05 / d[ii])
    msk = pmc[r, c] > 0.5
    ii, c, r, b = ii[msk], c[msk], r[msk], b[msk]
    keep = b > L[r, c] * 0.7
    ii, c, r, b = ii[keep], c[keep], r[keep], b[keep]
    if len(b) == 0:
        return
    ch = np.where(b < 0.20, CIDX_DOT, np.where(b < 0.42, CIDX_PLUS, CIDX_STAR))
    idx[r, c] = ch
    fg[r, c] = SCOL[ii] * np.clip(0.35 + 0.9 * b, 0, 1)[:, None] * vigc[r, c][:, None]

    big = np.flatnonzero(b > 0.55)
    for j in big[:40]:
        cc_, rr_, bb = int(c[j]), int(r[j]), float(b[j])
        col = SCOL[ii[j]] * vigc[rr_, cc_]
        idx[rr_, cc_] = CIDX_AT if bb > 0.85 else CIDX_STAR
        ln = int(2 + 6 * (bb - 0.5))
        for k in range(1, ln + 1):
            a = (1 - k / (ln + 1)) * bb
            kk = k // 2 + (1 if k > 2 else 0)
            chh = CIDX_DASH if a > 0.18 else CIDX_DOT
            chv = CIDX_BAR if a > 0.18 else CIDX_DOT
            for (dr, dc, chx) in ((0, k, chh), (0, -k, chh), (kk, 0, chv), (-kk, 0, chv)):
                if dr == 0 and dc == 0:
                    continue        # short rays collapse onto the star itself
                r2, c2 = rr_ + dr, cc_ + dc
                if 0 <= r2 < ROWS and 0 <= c2 < COLS and pmc[r2, c2] > 0.5 and L[r2, c2] < a * 0.8:
                    idx[r2, c2] = chx
                    fg[r2, c2] = col * min(1.0, 0.25 + a)


_SEG_IDS = np.array([CIDX["-"], CIDX["/"], CIDX["|"], CIDX["\\"]], np.int16)
_KMAX = 14                      # longest upper lash, in steps


def stamp_lashes(idx, fg, t, z, ou, gx, gy):
    """Stamp both rows of eyelashes.

    All lashes of a row are traced at once: the polyline of every hair is a
    cumulative sum of its step vectors, so the whole row becomes a handful of
    array operations and one bounds-checked, order-preserving scatter.
    """
    if z > 2.6 or ou < 0.0:
        return
    cl = 1 - ou ** 0.8
    zh = z * H
    w2 = W * 0.5
    h2 = H * 0.5
    cid_tip = (CIDX["'"], CIDX["."])
    k = np.arange(_KMAX, dtype=np.float64)

    for kind, (lashes, upper) in enumerate(((UP_L, True), (LO_L, False))):
        if not lashes:
            continue
        s0 = np.array([l[0] for l in lashes], np.float64)
        u0 = np.array([l[1] for l in lashes], np.float64)
        lf = np.array([l[2] for l in lashes], np.float64)
        a_open = np.array([l[3] for l in lashes], np.float64)
        a_closed = np.array([l[4] for l in lashes], np.float64)
        curl0 = np.array([l[5] for l in lashes], np.float64)
        Lh0 = np.array([l[6] for l in lashes], np.float64)

        vu, vl, cl_ = lid_edges(u0, ou)
        if upper:
            bv = vu - 0.055 * s0
            a0 = a_open + (a_closed - a_open) * cl
            Lh = Lh0
            curl = curl0 * (1 - cl)
            n_max, root, tip = 14, UP_ROOT, UP_TIP
        else:
            bv = vl - 0.055 * s0
            a0 = np.radians(90 - 20 * s0)
            Lh = 0.034 * (0.5 + 0.8 * cl_) * lf
            curl = np.radians(12 * np.where(s0 > 0, 1.0, -1.0))
            n_max, root, tip = 6, LO_ROOT, LO_TIP

        n = np.maximum(2.0, np.minimum(n_max, Lh * z * H / (CW * 0.85))).astype(np.int32)
        step = Lh / n
        valid = k[None, :] < n[:, None]
        tt = np.where(valid, k[None, :] / np.maximum(1, n - 1)[:, None], 0.0)
        aa = a0[:, None] + curl[:, None] * tt * tt
        du = np.cos(aa) * step[:, None]
        dv = np.sin(aa) * step[:, None]
        ue = u0[:, None] + np.cumsum(du, axis=1)
        ve = bv[:, None] + np.cumsum(dv, axis=1)
        c = ((w2 + ue * zh) / CW).astype(np.int32)
        r = ((h2 + ve * zh) / CH).astype(np.int32)
        valid &= (r >= 0) & (r < ROWS) & (c >= 0) & (c < COLS)

        d = np.degrees(np.arctan2(-np.sin(aa), np.cos(aa))) % 180.0
        code = np.where((d < 25.0) | (d >= 155.0), 0,
                        np.where(d < 70.0, 1, np.where(d < 110.0, 2, 3)))
        ids = _SEG_IDS[code]
        last = (k[None, :] == (n - 1)[:, None]) & (n >= 4)[:, None]
        ids = np.where(last, cid_tip[kind], ids)

        cols = root + (tip - root) * tt[..., None]
        rr = r[valid].ravel()
        cc = c[valid].ravel()
        chs = ids[valid].ravel().astype(np.int16)
        colv = cols[valid].reshape(-1, 3)
        if len(rr) == 0:
            continue
        # Later strokes win where two lashes cross: keep the last duplicate.
        order = np.argsort(rr * COLS + cc, kind="stable")
        key = (rr * COLS + cc)[order]
        last_of = np.empty(len(key), bool)
        last_of[:-1] = key[1:] != key[:-1]
        last_of[-1] = True
        sel = order[last_of]
        idx[rr[sel], cc[sel]] = chs[sel]
        fg[rr[sel], cc[sel]] = colv[sel]


def bloom(img):
    """Glow: blur the bright parts at two scales and add them back."""
    b = img - 0.45
    np.clip(b, 0, None, out=b)
    for f, w_ in ((2, 0.40), (4, 0.30)):
        s_ = _block_mean(b, f)
        for _ in range(2):
            acc = s_ * 2
            acc[1:] += s_[:-1]
            acc[:1] += s_[-1:]
            acc[:-1] += s_[1:]
            acc[-1:] += s_[:1]
            acc[:, 1:] += s_[:, :-1]
            acc[:, :1] += s_[:, -1:]
            acc[:, :-1] += s_[:, 1:]
            acc[:, -1:] += s_[:, :1]
            s_ = acc / 6
        up = np.repeat(np.repeat(s_, f, 0), f, 1)
        img += w_ * up
    return img


def render_cells(t, dt):
    """Compute one frame: (character indices, quantized colours, float colours)."""
    if FX["loop"]:
        t_local = t % T_LOOP
        if t_local < S["tlocal"]:
            reset()                     # crossed the loop point: fresh world
        S["tlocal"] = t_local
        t = t_local
    S["last"] = t

    prog = zp(t)
    z = zoom_amount(t)
    zsp = star_spread(t, z)
    gs = min(6.0, GS0 * zsp ** 1.25)
    gh = min(1.15, GS0 * z ** 1.25)
    ou = lid_open(t)
    if FX["pointer"] and POINTER["active"]:
        gx, gy = pointer_gaze(t, dt)
    else:
        gx, gy = gaze(t, prog)
    Rp = RP0 * pupil_scale(t) * (0.76 + 0.24 * ease((t - 1.8) / 3.5)) * \
        (1 + 0.025 * m.sin(1.9 * t)) * (1 + 0.30 * prog)
    Rps = Rp * z
    full = Rps * 0.72 > 1.03

    star_amt = ease((t - (T_Z0 + 1.5)) / 4.5)
    neb_amt = ease((t - (T_Z0 + 3.5)) / 8.0)
    # how much of the space behind the pupil is actually visible through the
    # eyelids: stars and streaks are stamped straight into the character grid,
    # so they must fade out while the eye closes (and the nebula is skipped
    # entirely once it is hidden behind the skin)
    eye_vis = ss(0.10, 0.45, ou)
    update_world(t, dt, prog)

    offx, offy = gx * z, gy * z
    sp = SPIMAGE
    sp[:] = SPACE_BG
    if neb_amt > 0.002 and (full or eye_vis > 0.002):
        drift = ease((t - T_SPACE) / 20.0)
        hm = ease((t - T_HEART) / T_HEART_D)
        beat = 1 + 0.03 * m.sin(t * 2.6) * hm * hm
        neb = heart_nebula(X - offx - 0.03 * drift * m.sin(t * 0.07), Y - offy,
                           0.17 * gh * beat, hm, t)
        sp += neb * (1.25 * neb_amt ** 1.3)

    if full:
        img = sp
        pmc = None
    else:
        img, pm = eye_scene(t, z, ou, gx, gy, Rp, sp)
        pmc = _downsample(pm)

    img = bloom(img)
    img = 1 - np.exp(-2.0 * np.clip(img, 0, None))
    img *= _rgb3((1 - 0.30 * ss(0.4, 1.0, RS)) * ease(t / 1.2))

    cells = _downsample(img)
    lum = 0.30 * cells[..., 0] + 0.60 * cells[..., 1] + 0.10 * cells[..., 2]
    li = np.clip(lum, 0, 1) ** 0.72
    ri = np.clip((li + (HSC - 0.5) * 0.07) * NR, 0, NR).astype(np.int32)
    if pmc is None:
        idx = ri
        pmc = np.ones((ROWS, COLS), np.float32)
    else:
        idx = np.where(pmc < 0.5, np.maximum(ri, 3), ri)
    fg = np.clip(cells * 1.15, 0, 1).astype(np.float32)

    vigc = (0.25 + 0.75 * ss(0, 0.30 * Rps, Rps - RSC)).astype(np.float32)
    visc = pmc * eye_vis
    stamp_stars(idx, fg, li, t, gs, star_amt, visc, vigc, offx, offy)
    for s_ in streaks:
        stamp_streak(s_, idx, fg, visc, vigc)
    if not full:
        stamp_lashes(idx, fg, t, z, ou, gx, gy)

    if MONO:
        mono = 0.30 * fg[..., 0] + 0.60 * fg[..., 1] + 0.10 * fg[..., 2]
        fg = np.stack((mono, mono, mono), axis=-1)

    q = np.clip((np.clip(fg, 0, 1) * (QL - 0.01)).astype(np.int16), 0, QL - 1)
    return idx, q, fg


def plan_updates(idx, q, d_idx, d_q, budget):
    same = idx == d_idx
    delta = np.abs(q - d_q).max(axis=2)
    blank_both = (idx == 0) & (d_idx == 0)
    need = (~blank_both) & ((~same) | (delta >= 2))
    cells = np.flatnonzero(need)
    if len(cells) > budget:
        prio = ((~same).astype(np.int16) * 6 + delta).ravel()[cells]
        top = np.argpartition(-prio, budget)[:budget]
        cells = cells[top]
    return cells


# ---------------------------------------------------------------------------
# Presentation layer
# ---------------------------------------------------------------------------

_FAMILY_FILES = {
    "cascadia mono": (("CascadiaMono.ttf", "CascadiaMonoPL.ttf", "CascadiaCode.ttf"), ()),
    "cascadia code": (("CascadiaCode.ttf", "CascadiaMono.ttf"), ()),
    "consolas": (("consola.ttf",), ("consolab.ttf",)),
    "courier new": (("cour.ttf",), ("courbd.ttf",)),
    "courier": (("cour.ttf",), ("courbd.ttf",)),
    "lucida console": (("lucon.ttf",), ()),
    "dejavu sans mono": (("DejaVuSansMono.ttf",), ("DejaVuSansMono-Bold.ttf",)),
    "liberation mono": (("LiberationMono-Regular.ttf",), ("LiberationMono-Bold.ttf",)),
    "ubuntu mono": (("UbuntuMono-R.ttf",), ("UbuntuMono-B.ttf",)),
    "menlo": (("Menlo.ttc",), ()),
    "monaco": (("Monaco.ttf",), ()),
    "andale mono": (("AndaleMono.ttf",), ()),
}
_GENERIC_MONO = ("CascadiaMono.ttf", "consola.ttf", "cour.ttf", "DejaVuSansMono.ttf",
                 "LiberationMono-Regular.ttf", "UbuntuMono-R.ttf", "lucon.ttf", "Monaco.ttf")


def _font_dirs():
    dirs = []
    win = os.environ.get("WINDIR")
    if win:
        dirs.append(os.path.join(win, "Fonts"))
    dirs += ["/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.local/share/fonts"), os.path.expanduser("~/.fonts"),
             "/System/Library/Fonts", "/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    return [d for d in dirs if os.path.isdir(d)]


def _find_font_files(family, bold):
    """Return (regular, bold) font-file paths for a font family, best first."""
    fam = (family or "").strip().lower()
    reg, boldfiles = _FAMILY_FILES.get(fam, ((), ()))
    names = [n for n in reg] + [n for n in _GENERIC_MONO if n not in reg]
    boldnames = [n for n in boldfiles]
    out_reg, out_bold = [], []
    dirs = _font_dirs()
    for d in dirs:
        for name in names:
            p = os.path.join(d, name)
            if os.path.exists(p) and p not in out_reg:
                out_reg.append(p)
        for name in boldnames:
            p = os.path.join(d, name)
            if os.path.exists(p) and p not in out_bold:
                out_bold.append(p)
        if out_reg and (out_bold or not bold):
            return out_reg, out_bold

    # fonts are often nested in per-family sub-directories on Linux/macOS
    wanted = {n.lower(): i for i, n in enumerate(names)}
    wanted_bold = {n.lower(): i for i, n in enumerate(boldnames)}
    found, found_bold = [], []
    for d in dirs:
        for root, _subdirs, files in os.walk(d):
            for fname in files:
                low = fname.lower()
                if low in wanted and os.path.join(root, fname) not in found:
                    found.append((wanted[low], os.path.join(root, fname)))
                elif low in wanted_bold and os.path.join(root, fname) not in found_bold:
                    found_bold.append((wanted_bold[low], os.path.join(root, fname)))
    out_reg = [p for _i, p in sorted(found)] or out_reg
    out_bold = [p for _i, p in sorted(found_bold)] or out_bold
    return out_reg, out_bold


def _fit_font(make_font, cw, ch):
    """Pixel size whose advance is exactly one cell wide and fits the row."""
    lo = max(4, int(ch * 0.5))
    hi = int(ch * 1.6) + 1
    best = None
    for px in range(lo, hi):
        try:
            f = make_font(px)
        except Exception:
            continue
        adv = f.getlength("M" * 10) / 10.0
        asc, desc = f.getmetrics()
        if round(adv) != cw or asc + desc > ch:
            continue
        score = (asc + desc) - ch          # prefer the tallest glyph that fits
        if best is None or score > best[0]:
            best = (score, px, f)
    if best is not None:
        return best[1], best[2]

    best = None
    for px in range(lo, hi):
        try:
            f = make_font(px)
        except Exception:
            continue
        adv = f.getlength("M" * 10) / 10.0
        asc, desc = f.getmetrics()
        if asc + desc > ch or adv > cw + 0.5:
            continue
        score = -abs(adv - cw) * 4 - abs((asc + desc) - ch)
        if best is None or score > best[0]:
            best = (score, px, f)
    if best is None:
        return None, None
    return best[1], best[2]


def _open_tk_font(family, bold, cw, ch):
    """Find a Pillow font that matches the Tk font actually in use."""
    from PIL import ImageFont

    reg, boldf = _find_font_files(family, bold)
    makers = []
    for p in (boldf if bold else ()):
        makers.append(lambda px, p=p: ImageFont.truetype(p, px))
    for p in reg:
        def mk(px, p=p):
            f = ImageFont.truetype(p, px)
            if bold:
                try:
                    f.set_variation_by_name("Bold")
                except Exception:
                    raise ValueError("%s is not a variable font" % p)
            return f
        makers.append(mk)
    for mk in makers:
        try:
            size_px, font = _fit_font(mk, cw, ch)
        except Exception:
            continue
        if font is not None:
            return font, size_px
    return None, None


def build_glyph_masks(font, ramp, cw, ch, asc, desc):
    """Rasterise every character of the ramp into a cw x ch coverage mask."""
    from PIL import Image, ImageDraw

    cw, ch = int(round(cw)), int(round(ch))
    masks = np.zeros((len(ramp), ch, cw), np.uint8)
    baseline = int(round((ch + asc - desc) / 2.0 + GLYPH_DY))
    tile = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(tile)
    for i, chx in enumerate(ramp):
        draw.rectangle((0, 0, cw, ch), fill=0)
        draw.text((cw / 2.0, baseline), chx, font=font, fill=255, anchor="ms")
        masks[i] = np.asarray(tile)
    return masks


def _shot_path(ext):
    return time.strftime("eye_%Y%m%d_%H%M%S.") + ext


class RasterView:
    """Compose the character grid into a single bitmap and blit it.

    Uses the full 8-bit colour that the scene produces (the text renderer has
    to quantize colours to keep its hex cache small) and applies an optional
    CRT overlay: scanlines plus a soft vignette, baked into one mask image.
    """

    def __init__(self, canvas, cols, rows, cw, ch, qlevels, family, bold, asc, desc):
        from PIL import Image, ImageChops, ImageTk

        cw, ch = int(round(cw)), int(round(ch))
        self.canvas = canvas
        self.cw, self.ch = cw, ch
        self.cols, self.rows = cols, rows
        self.width, self.height = cols * cw, rows * ch
        self.Image = Image
        self.ImageChops = ImageChops
        font, size_px = _open_tk_font(family, bold, cw, ch)
        if font is None:
            raise RuntimeError("no usable TrueType font found")
        self.font, self.size_px = font, size_px
        self.masks = build_glyph_masks(font, RAMP, cw, ch, asc, desc)
        self.palette = (np.arange(qlevels, dtype=np.uint16) * 255 // (qlevels - 1)).astype(np.uint8)
        self.bg = Image.new("RGB", (self.width, self.height), BG_RGB)
        if canvas is not None:                 # None = off-screen (save only)
            self.photo = ImageTk.PhotoImage(self.bg)
            canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.photo = None
        self.color = np.empty((rows, cols, 3), np.uint8)
        self.crt = self._crt_mask() if FX["crt"] else None
        self.frame = self.bg

    def _crt_mask(self):
        """Scanlines + vignette as one multiply mask."""
        w, h = self.width, self.height
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        scan = np.where((ys % 2) < 1, np.float32(0.84), np.float32(1.0))
        vy = (ys - (h - 1) * 0.5) / (h * 0.5)
        vx = (xs - (w - 1) * 0.5) / (w * 0.5)
        r2 = vx[None, :] ** 2 + vy[:, None] ** 2
        vig = 1.0 - 0.16 * np.clip(r2 * 0.5, 0.0, 1.0) ** 1.4
        m = (scan[:, None] * vig * 255.0).astype(np.uint8)
        return self.Image.fromarray(np.repeat(m[..., None], 3, axis=2), "RGB")

    def draw(self, idx, q, fg=None):
        Image = self.Image
        if fg is not None:
            self.color = np.clip(fg * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            np.take(self.palette, q, out=self.color)
        big = Image.fromarray(self.color, "RGB").resize((self.width, self.height), Image.NEAREST)
        mask = np.take(self.masks, idx, axis=0).transpose(0, 2, 1, 3)
        mask_img = Image.fromarray(np.ascontiguousarray(mask).reshape(self.height, self.width), "L")
        out = Image.composite(big, self.bg, mask_img)
        if self.crt is not None:
            out = self.ImageChops.multiply(out, self.crt)
        self.frame = out
        if self.photo is not None:
            self.photo.paste(out)

    def save(self, path=None):
        path = path or _shot_path("png")
        self.frame.save(path)
        return path


class TextView:
    """Fallback: one canvas text item per cell, with an adaptive update budget."""

    def __init__(self, canvas, font, cols, rows, cw, ch, qlevels):
        self.canvas = canvas
        self.font = font
        self.cw, self.ch = cw, ch
        self.cols, self.rows = cols, rows
        self.items = np.zeros((rows, cols), dtype=object)
        self.items[:] = 0
        self.d_idx = np.zeros((rows, cols), np.int32)
        self.d_q = np.zeros((rows, cols, 3), np.int16)
        levels = np.arange(qlevels, dtype=np.int32) * 255 // (qlevels - 1)
        self.hex = np.array(["#%02x%02x%02x" % (r, g, b)
                             for r in levels for g in levels for b in levels], dtype=object)
        self.hex = self.hex.reshape(qlevels, qlevels, qlevels)
        self.budget = 2500
        self.cost = 0.012
        self.xs = ((np.arange(cols) + 0.5) * cw).tolist()
        self.ys = ((np.arange(rows) + 0.5) * ch).tolist()
        self.ramp = [c for c in RAMP]
        self.idx_last = None

    def draw(self, idx, q, fg=None):
        self.idx_last = idx
        cells = plan_updates(idx, q, self.d_idx, self.d_q, self.budget)
        t1 = time.perf_counter()
        items = self.items
        d_idx, d_q = self.d_idx, self.d_q
        create = self.canvas.create_text
        config = self.canvas.itemconfig
        ramp, hexs, xs, ys = self.ramp, self.hex, self.xs, self.ys
        cols = self.cols
        for f in cells:
            r_, c_ = divmod(int(f), cols)
            i_ = idx[r_, c_]
            ch = ramp[i_]
            rgb = q[r_, c_]
            fill = hexs[rgb[0], rgb[1], rgb[2]]
            it = items[r_, c_]
            if it == 0:
                if ch == " ":
                    d_idx[r_, c_] = 0
                    d_q[r_, c_] = rgb
                    continue
                items[r_, c_] = create(xs[c_], ys[r_], text=ch, fill=fill,
                                       font=self.font, anchor="center")
            else:
                config(it, text=ch, fill=fill)
            d_idx[r_, c_] = i_
            d_q[r_, c_] = rgb
        n = len(cells)
        if n > 50:
            per = (time.perf_counter() - t1) / n
            self.cost = 0.7 * self.cost + 0.3 * per
            self.budget = int(min(12000, max(400, (TK_MS / 1000.0) / max(self.cost, 1e-6))))
        return n

    def save(self, path=None):
        """Save the frame as plain text (the text renderer has no bitmap)."""
        path = path or _shot_path("txt")
        rows = ["".join(self.ramp[i] for i in row) for row in self.idx_last]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        return path


# ---------------------------------------------------------------------------
# Optional OpenGL (GPU) back-end
#
# The scene stays on the CPU (it is launch-bound at these sizes), but the
# expensive part -- turning thousands of characters into pixels -- becomes one
# textured quad: the fragment shader looks the glyph mask up in an atlas and
# tints it with the per-cell colour.  That is the part whose cost grows with
# the window area (7.8 ms for 624x450, 23 ms for 1034x750 on the CPU path).
# ---------------------------------------------------------------------------

GPU_VERT = """
#version 330
in vec2 in_pos;
uniform vec2 u_scale;        // grid size / framebuffer size
void main() { gl_Position = vec4(in_pos * u_scale, 0.0, 1.0); }
"""

GPU_FRAG = """
#version 330
uniform sampler2D u_atlas;   // glyph coverage masks, laid out horizontally
uniform sampler2D u_idx;     // ramp index per cell (0..1, multiply by 255)
uniform sampler2D u_col;     // colour per cell
uniform vec2 u_cells;        // cols, rows
uniform vec2 u_cell;         // cell size on screen (pixels, may be scaled up)
uniform vec2 u_tile;         // glyph tile size inside the atlas (pixels)
uniform vec2 u_size;         // size of the drawn character grid in pixels
uniform vec2 u_off;          // where that area starts inside the framebuffer
uniform vec3 u_bg;           // window background
uniform float u_chars;       // number of characters in the ramp
uniform int u_crt;           // 1 = scanlines + vignette
out vec4 f_colour;

void main() {
    // gl_FragCoord is pixel-centred and bottom-up; u_off centres the grid
    vec2 frag = gl_FragCoord.xy - u_off;
    if (frag.x < 0.0 || frag.y < 0.0 || frag.x >= u_size.x || frag.y >= u_size.y) {
        f_colour = vec4(u_bg, 1.0);          // letterbox around the grid
        return;
    }
    // pixel-centred coordinates with the origin at the top-left, like NumPy
    vec2 px = vec2(frag.x, u_size.y - frag.y);
    vec2 cell = floor(px / u_cell);
    vec2 uv_cell = (cell + 0.5) / u_cells;
    float c = floor(texture(u_idx, uv_cell).r * 255.0 + 0.5);
    vec3 rgb = texture(u_col, uv_cell).rgb;
    vec2 local = px - cell * u_cell;          // 0.5 .. cell-0.5
    vec2 texel = local * (u_tile / u_cell);   // nearest-neighbour upscale of the tile
    float a = texture(u_atlas, vec2(c * u_tile.x + texel.x, texel.y)
                                / vec2(u_chars * u_tile.x, u_tile.y)).r;
    vec3 col = rgb * a + u_bg * (1.0 - a);
    if (u_crt == 1) {
        float scan = (mod(floor(px.y), 2.0) < 1.0) ? 0.84 : 1.0;
        vec2 v = px / u_size * 2.0 - 1.0;
        float vig = 1.0 - 0.16 * pow(clamp(dot(v, v) * 0.5, 0.0, 1.0), 1.4);
        col *= scan * vig;
    }
    f_colour = vec4(col, 1.0);
}
"""


def font_metrics():
    """Cell metrics measured with Tk, so every back-end uses the same grid."""
    import tkinter as tk
    import tkinter.font as tkf

    root = tk.Tk()
    root.withdraw()
    fams = set(tkf.families())
    family = next((f for f in ("Cascadia Mono", "Consolas", "Courier New",
                               "DejaVu Sans Mono", "Courier") if f in fams), "Courier")
    fnt = tkf.Font(root=root, family=family, size=SIZE, weight="bold")
    cw = float(fnt.measure("M"))
    ch = float(fnt.metrics("linespace"))
    asc, desc = fnt.metrics("ascent"), fnt.metrics("descent")
    rows = min(MAXROWS, (root.winfo_screenheight() - 110) // int(ch))
    root.destroy()
    return family, cw, ch, asc, desc, rows


class GpuView:
    """Compose the character grid in a fragment shader (one draw call)."""

    def __init__(self, cols, rows, cw, ch, family, bold, asc, desc):
        import glfw
        import moderngl

        self.glfw = glfw
        self.moderngl = moderngl
        icw, ich = int(round(cw)), int(round(ch))
        self.tile = (float(icw), float(ich))       # atlas tile size
        self.cw, self.ch = float(icw), float(ich)  # current on-screen cell size
        self.family, self.bold, self.asc, self.desc = family, bold, asc, desc
        if not glfw.init():
            raise RuntimeError("glfw.init() failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)     # the window can be dragged
        self.cols, self.rows = cols, rows
        self.zoom = 1.0
        self.width, self.height = cols * icw, rows * ich
        self.fb = (self.width, self.height)
        self.off = (0, 0)
        self.window = glfw.create_window(self.width, self.height, "Eye -> Space", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create an OpenGL window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)                 # precise pacing instead of vsync
        self.ctx = moderngl.create_context()

        font, _size = _open_tk_font(family, bold, cw, ch)
        if font is None:
            glfw.terminate()
            raise RuntimeError("no usable TrueType font found")
        masks = build_glyph_masks(font, RAMP, icw, ich, asc, desc)
        atlas = np.zeros((ich, len(RAMP) * icw), np.uint8)
        for i, m in enumerate(masks):
            atlas[:, i * icw:(i + 1) * icw] = m
        self.n_chars = len(RAMP)

        self.prog = self.ctx.program(vertex_shader=GPU_VERT, fragment_shader=GPU_FRAG)
        quad = np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], np.float32)
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.tex_atlas = self._texture((self.n_chars * icw, ich), 1, atlas.tobytes())
        self.tex_idx = self._texture((cols, rows), 1, bytes(cols * rows))
        self.tex_col = self._texture((cols, rows), 3, bytes(cols * rows * 3))
        self.ctx.disable(self.moderngl.BLEND)
        self._use_fb(self.fb)

    # -- layout -------------------------------------------------------------
    def _use_fb(self, fb):
        """Point the view at a framebuffer of this size and centre the grid in it."""
        self.fb = (int(fb[0]), int(fb[1]))
        self.off = ((self.fb[0] - self.width) // 2, (self.fb[1] - self.height) // 2)
        self.ctx.viewport = (0, 0, self.fb[0], self.fb[1])
        self._set_uniforms()

    def relayout(self, fb_w, fb_h):
        """Fit the grid to a new window size.

        The number of cells is capped (the CPU scene grows with it), so a bigger
        window ends up magnifying the ASCII instead of adding cells: the frame
        budget, and with it the frame rate, stays where it was.
        """
        rows, cols = grid_for(fb_w, fb_h, self.tile[0], self.tile[1])
        aspect = (cols * self.tile[0]) / (rows * self.tile[1])
        init(self.tile[0], self.tile[1], rows, aspect=aspect, reset_state=False)
        self.cols, self.rows = COLS, ROWS
        gw = COLS * self.tile[0]
        gh = ROWS * self.tile[1]
        self.zoom = max(1.0, min(fb_w / gw, fb_h / gh))
        self.cw = self.tile[0] * self.zoom
        self.ch = self.tile[1] * self.zoom
        self.width = int(round(gw * self.zoom))
        self.height = int(round(gh * self.zoom))
        for tex in (self.tex_idx, self.tex_col):
            tex.release()
        self.tex_idx = self._texture((self.cols, self.rows), 1, bytes(self.cols * self.rows))
        self.tex_col = self._texture((self.cols, self.rows), 3,
                                     bytes(self.cols * self.rows * 3))
        self._use_fb((fb_w, fb_h))
        return self.cols, self.rows

    # -- moderngl helpers ---------------------------------------------------
    def _texture(self, size, components, data):
        tex = self.ctx.texture(size, components, data, dtype="f1")
        tex.filter = (self.moderngl.NEAREST, self.moderngl.NEAREST)
        tex.repeat_x = tex.repeat_y = False
        return tex

    def _set_uniforms(self):
        p = self.prog
        p["u_atlas"] = 0
        p["u_idx"] = 1
        p["u_col"] = 2
        p["u_cells"] = (float(self.cols), float(self.rows))
        p["u_cell"] = (self.cw, self.ch)
        p["u_tile"] = self.tile
        p["u_size"] = (float(self.width), float(self.height))
        p["u_off"] = (float(self.off[0]), float(self.off[1]))
        p["u_scale"] = (self.width / float(self.fb[0] or 1),
                        self.height / float(self.fb[1] or 1))
        p["u_bg"] = (BG_RGB[0] / 255.0, BG_RGB[1] / 255.0, BG_RGB[2] / 255.0)
        p["u_chars"] = float(self.n_chars)
        p["u_crt"] = 1 if FX["crt"] else 0

    # -- frame --------------------------------------------------------------
    def draw(self, idx, q, fg=None):
        if fg is not None:
            col = np.clip(fg * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            col = (np.arange(QL, dtype=np.uint16) * 255 // (QL - 1)).astype(np.uint8)[q]
        self.tex_idx.write(idx.astype(np.uint8).tobytes())
        self.tex_col.write(col.tobytes())
        self.ctx.clear(BG_RGB[0] / 255.0, BG_RGB[1] / 255.0, BG_RGB[2] / 255.0, 1.0)
        self.tex_atlas.use(0)
        self.tex_idx.use(1)
        self.tex_col.use(2)
        self.vao.render(self.moderngl.TRIANGLES, 6)

    def present(self):
        self.glfw.swap_buffers(self.window)

    def close(self):
        try:
            self.glfw.destroy_window(self.window)
            self.glfw.terminate()
        except Exception:
            pass

    def save(self, path=None):
        """Render the character grid off-screen at 1:1 (no zoom, no letterbox)."""
        from PIL import Image

        path = path or _shot_path("png")
        gw = int(round(self.cols * self.tile[0]))
        gh = int(round(self.rows * self.tile[1]))
        fbo = self.ctx.simple_framebuffer((gw, gh), components=3)
        try:
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            self.ctx.viewport = (0, 0, gw, gh)
            self.prog["u_off"] = (0.0, 0.0)
            self.prog["u_scale"] = (1.0, 1.0)
            self.prog["u_cell"] = self.tile
            self.prog["u_size"] = (float(gw), float(gh))
            self.tex_atlas.use(0)
            self.tex_idx.use(1)
            self.tex_col.use(2)
            self.vao.render(self.moderngl.TRIANGLES, 6)
            data = fbo.read(components=3, alignment=1)
        finally:
            self._use_fb(self.fb)
        img = Image.frombytes("RGB", (gw, gh), data)
        img.transpose(Image.FLIP_TOP_BOTTOM).save(path)
        return path


def make_view(canvas, tk_font, cols, rows, cw, ch):
    """Pick the fastest available presentation back-end."""
    backend = USE_BACKEND
    if backend in ("auto", "raster"):
        try:
            asc = tk_font.metrics("ascent")
            desc = tk_font.metrics("descent")
            return RasterView(canvas, cols, rows, cw, ch, QL,
                              tk_font.actual("family"), tk_font.actual("weight") == "bold",
                              asc, desc)
        except Exception as exc:
            if backend == "raster":
                raise
            print("raster backend unavailable (%s: %s), using text" % (type(exc).__name__, exc))
    return TextView(canvas, tk_font, cols, rows, cw, ch, QL)


class Pacer:
    """Keep the frame period close to 1/FPS.

    Tk redraws the window *after* the timer callback returns, so the paint
    time is added to every frame.  Measuring that lateness and subtracting it
    from the delay is what actually holds 30 FPS instead of ~25.
    """

    def __init__(self, fps, paint=0.004):
        self.period = 1.0 / fps
        self.paint = paint
        self.due = time.perf_counter() + self.period
        self.behind = False

    def start_frame(self):
        now = time.perf_counter()
        late = now - self.due
        # Only trust the lateness when the previous frame did finish in time --
        # otherwise it also contains the frame's own work and the estimate
        # would run away (and the delay would collapse to zero).
        if not self.behind and 0.0 <= late < min(0.25, self.period):
            self.paint = min(0.05, max(0.0, self.paint + 0.3 * (late - self.paint)))
        self.behind = False
        return now

    def next_delay_ms(self, frame_start):
        now = time.perf_counter()
        delay = self.period - (now - frame_start) - self.paint
        if delay <= 0.0:
            self.behind = True
        self.due = now + max(0.0, delay)
        return max(1, int(round(delay * 1000.0)))


def _timer_resolution(ms=1):
    """Raise the Windows timer resolution so time.sleep() is accurate (~1 ms).

    Only needed by the GL back-end, which paces itself with sleep() instead of
    Tk's `after()`.  Returns a function that restores the previous setting.
    """
    try:
        import ctypes

        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(ms)

        def restore():
            try:
                winmm.timeEndPeriod(ms)
            except Exception:
                pass
        return restore
    except Exception:
        return lambda: None


def main():
    """Start the animation with the best available back-end.

    ``auto`` (the default) tries the GPU first; ``main_gpu`` falls back to Tk
    when OpenGL or the optional packages are missing.
    """
    if USE_BACKEND in ("auto", "gpu"):
        return main_gpu()
    return main_tk()


def toggle_fullscreen(glfw, win, ui):
    """Borderless fullscreen on/off; ``ui`` remembers the windowed geometry."""
    if ui.get("full"):
        glfw.set_window_attrib(win, glfw.DECORATED, glfw.TRUE)
        w, h = ui.get("size", (800, 600))
        x, y = ui.get("pos", (120, 120))
        glfw.set_window_size(win, int(w), int(h))
        glfw.set_window_pos(win, int(x), int(y))
    else:
        ui["pos"] = glfw.get_window_pos(win)
        ui["size"] = glfw.get_window_size(win)
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        glfw.set_window_attrib(win, glfw.DECORATED, glfw.FALSE)
        glfw.set_window_pos(win, 0, 0)
        glfw.set_window_size(win, mode.size.width, mode.size.height)
    ui["full"] = not ui.get("full")


def main_gpu():
    """GL back-end: the scene stays on the CPU, one shader draws every cell."""
    try:
        import glfw
    except Exception as exc:
        print("GPU back-end unavailable (%s: %s) -- falling back to Tk"
              % (type(exc).__name__, exc))
        return main_tk()

    family, cw, ch, asc, desc, rows = font_metrics()
    init(cw, ch, int(rows))
    try:
        view = GpuView(COLS, ROWS, cw, ch, family, True, asc, desc)
    except Exception as exc:
        print("GPU back-end unavailable (%s: %s) -- falling back to Tk"
              % (type(exc).__name__, exc))
        return main_tk()

    restore_timer = _timer_resolution(1)
    pacer = Pacer(FPS) if FPS > 0 else None
    ui = {"view": view, "full": False, "resize": None}
    state = {"t": 0.0, "last_t": 0.0, "prev": time.perf_counter(),
             "frames": 0, "fps": FPS, "mark": time.perf_counter(), "paused": False}

    def restart():
        state["t"] = 0.0
        state["last_t"] = 0.0
        state["prev"] = time.perf_counter()
        reset()

    def toggle_full(win):
        """Borderless fullscreen (no exclusive mode, so coming back is easy)."""
        toggle_fullscreen(glfw, win, ui)

    def on_key(win, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_R:
            restart()
        elif key == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            state["prev"] = time.perf_counter()
        elif key == glfw.KEY_S:
            print("saved", ui["view"].save())
        elif key == glfw.KEY_F:
            toggle_full(win)

    glfw.set_key_callback(ui["view"].window, on_key)

    try:
        while not glfw.window_should_close(ui["view"].window):
            window = ui["view"].window
            t0 = pacer.start_frame() if pacer else time.perf_counter()
            now = time.perf_counter()
            if not state["paused"]:
                state["t"] += min(max(now - state["prev"], 0.0), 0.1)
            state["prev"] = now
            glfw.poll_events()
            # a drag, fullscreen or a DPI change all end up here
            fb = glfw.get_framebuffer_size(window)
            if fb != ui["view"].fb and fb[0] >= 64 and fb[1] >= 64:
                ui["view"].relayout(fb[0], fb[1])
            if state["paused"]:
                time.sleep(0.02)
                continue

            t = state["t"]
            dt = min(max(t - state["last_t"], 0.0), 0.1)
            state["last_t"] = t
            if FX["pointer"]:
                mx, my = glfw.get_cursor_pos(window)
                fw, fh = glfw.get_window_size(window)
                POINTER["active"] = bool(0 <= mx < fw and 0 <= my < fh)
                if POINTER["active"]:
                    # keep the gaze in the same units as the Tk back-end
                    POINTER["x"] = (mx - fw / 2) / max(1, fh)
                    POINTER["y"] = (my - fh / 2) / max(1, fh)

            idx, q, fg = render_cells(t, dt)
            ui["view"].draw(idx, q, fg)
            ui["view"].present()

            state["frames"] += 1
            if t0 - state["mark"] >= 0.5:
                state["fps"] = state["frames"] / (t0 - state["mark"])
                state["frames"] = 0
                state["mark"] = t0
                glfw.set_window_title(
                    window, "Eye -> Space    %d fps    %dx%d  x%.2f" % (
                        round(state["fps"]), ui["view"].cols, ui["view"].rows,
                        ui["view"].zoom))
            if pacer:
                time.sleep(pacer.next_delay_ms(t0) / 1000.0)
    finally:
        ui["view"].close()
        restore_timer()


def main_tk():
    """Tk back-end (bitmap or per-character canvas rendering)."""
    import tkinter as tk
    import tkinter.font as tkf

    root = tk.Tk()
    root.title("Eye -> Space")
    root.resizable(True, True)                 # the grid re-fits after a resize
    root.configure(bg="#010203")

    fams = set(tkf.families())
    family = next((f for f in ("Cascadia Mono", "Consolas", "Courier New",
                               "DejaVu Sans Mono", "Courier") if f in fams), "Courier")
    fnt = tkf.Font(root=root, family=family, size=SIZE, weight="bold")
    ch_ = fnt.metrics("linespace")
    cw_ = fnt.measure("M")
    scr_h = root.winfo_screenheight()
    rows = min(MAXROWS, (scr_h - 110) // ch_)
    init(cw_, ch_, rows)
    root.geometry(f"{W}x{H}+{(root.winfo_screenwidth() - W) // 2}+{max(0, (scr_h - H) // 2 - 20)}")
    cv = tk.Canvas(root, width=W, height=H, bg="#010203", highlightthickness=0)
    cv.place(relx=0.5, rely=0.5, anchor="center")
    ui = {"view": make_view(cv, fnt, COLS, ROWS, cw_, ch_), "job": None}

    # Tk paints the window *after* the timer fires, which used to add ~10-15%
    # to the frame period.  Pacer tracks that latency and subtracts it.
    pacer = Pacer(FPS) if FPS > 0 else None
    state = {"t": 0.0, "last_t": 0.0, "prev": time.perf_counter(),
             "frames": 0, "fps": FPS, "mark": time.perf_counter(), "paused": False}

    def restart(_=None):
        state["t"] = 0.0
        state["last_t"] = 0.0
        state["prev"] = time.perf_counter()
        reset()

    def toggle_pause(_=None):
        state["paused"] = not state["paused"]
        state["prev"] = time.perf_counter()

    def toggle_full(_=None):
        root.attributes("-fullscreen", 0 if root.attributes("-fullscreen") else 1)

    def save_frame(_=None):
        print("saved", ui["view"].save())

    def on_motion(event):
        POINTER["x"] = (event.x - W / 2) / H
        POINTER["y"] = (event.y - H / 2) / H
        POINTER["active"] = True

    def rebuild():
        """Re-fit the character grid to the current window size."""
        ui["job"] = None
        w, h = root.winfo_width(), root.winfo_height()
        if w < 64 or h < 64:
            return
        grow, gcol = grid_for(w, h, cw_, ch_)
        if (grow, gcol) == (ROWS, COLS):
            return
        init(cw_, ch_, grow, aspect=(gcol * cw_) / (grow * ch_), reset_state=False)
        cv.delete("all")
        cv.configure(width=W, height=H)
        ui["view"] = make_view(cv, fnt, COLS, ROWS, cw_, ch_)

    def on_configure(event):
        if event.widget is not root:
            return
        if ui["job"] is not None:
            root.after_cancel(ui["job"])
        ui["job"] = root.after(180, rebuild)   # wait until the drag is over

    root.bind("<Escape>", lambda e: root.destroy())
    root.bind("r", restart)
    root.bind("R", restart)
    root.bind("<space>", toggle_pause)
    root.bind("f", toggle_full)
    root.bind("F", toggle_full)
    root.bind("s", save_frame)
    root.bind("S", save_frame)
    root.bind("<Configure>", on_configure)
    cv.bind("<Motion>", on_motion)
    cv.bind("<Leave>", lambda e: POINTER.update(active=False))

    def tick():
        t0 = pacer.start_frame() if pacer else time.perf_counter()
        now = time.perf_counter()
        if not state["paused"]:
            state["t"] += min(max(now - state["prev"], 0.0), 0.1)
        state["prev"] = now
        if state["paused"]:
            root.after(50, tick)          # frozen: keep the last frame, idle CPU
            return
        t = state["t"]
        dt = min(max(t - state["last_t"], 0.0), 0.1)
        state["last_t"] = t
        idx, q, fg = render_cells(t, dt)
        ui["view"].draw(idx, q, fg)
        state["frames"] += 1
        if t0 - state["mark"] >= 0.5:
            state["fps"] = state["frames"] / (t0 - state["mark"])
            state["frames"] = 0
            state["mark"] = t0
            root.title("Eye -> Space    %d fps    %dx%d" % (
                round(state["fps"]), COLS, ROWS))
        root.after(pacer.next_delay_ms(t0) if pacer else 1, tick)

    root.after(100, tick)
    root.mainloop()


if __name__ == "__main__":
    main()

