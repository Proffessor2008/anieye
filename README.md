# Eye → Space 👁️

A generative ASCII-art animation: an eye opens, blinks, glances around, then the camera
dives into the pupil, flies through a starfield into a heart-shaped nebula — and comes
back out of the pupil, so the whole thing loops. Every frame is computed procedurally
with NumPy and drawn as text characters; there are no image files, fonts aside.

![the eye](docs/preview.png)

## What it does

- **Procedural scene.** Skin, iris fibres, sclera veins, the eyelid crease, individually
  traced eyelashes and a pupil that works as a portal are all built from noise textures
  and maths each frame, then mapped to a dark-to-bright character ramp.
- **One 37.5-second loop in five acts**: the eye act (opening, blinks, glances) lasts 12 s,
  the dive 16 s, the heart morph just 10 s, the flight back 4.5 s. See [Timeline](#timeline).
- **GPU renderer by default.** A single fragment shader composes the whole character grid
  at **100 FPS**, and keeps that rate while the window is resized.
- **Full colour, five palettes, CRT overlay.** `amber`, `ice`, `emerald`, `violet`, `noir`;
  optional scanlines + vignette; `--seed N` for a different iris and starfield.
- **Interactive.** The eye follows the mouse pointer; pause, fullscreen, save a frame;
  drag the window edges to resize.
- **Portable.** Three renderers share the same picture — GPU (OpenGL), a Tk bitmap
  renderer, and a plain per-character canvas renderer that needs nothing but NumPy.

## Requirements

```bash
pip install numpy            # required
pip install pillow           # Tk bitmap renderer
pip install moderngl glfw    # GPU renderer (default when available)
```

Everything beyond NumPy is optional: without `moderngl`/`glfw` the app runs on the Tk
bitmap renderer, and without `pillow` it falls back to per-character canvas text.
Tkinter ships with Python on Windows and macOS; on Linux install `python3-tk`.

## Run

```bash
python eye.py                   # GPU renderer, 100 FPS, resizable window
python eye.py --fps 0           # uncapped frame rate
python eye.py --fps 60          # calmer
python eye.py --size 12         # bigger characters
python eye.py --rows 40         # taller first window
python eye.py --theme ice --seed 12
python eye.py --no-crt          # flat image, no scanlines
python eye.py --once            # no loop: the camera dives in and stays in the nebula
python eye.py --raster          # force the Tk bitmap renderer
python eye.py --text            # force the per-character canvas renderer
```

### Keys

| key | action |
| --- | --- |
| move mouse | the eye looks at the pointer |
| drag window edges | resize: the grid re-fits, then the characters themselves grow |
| `R` | restart the animation |
| `Space` | pause / resume |
| `F` | toggle borderless fullscreen |
| `S` | save the current frame at 1:1 — `eye_YYYYmmdd_HHMMSS.png` (bitmap/GPU) or `.txt` (text) |
| `Esc` | quit |

The title bar shows the live frame rate, the grid size and the current magnification,
for example `100 fps    104x30  x1.00`.

### Options

| option | meaning |
| --- | --- |
| `--fps N` | target frame rate (default **100**, `0` = uncapped) |
| `--size N` | font size, i.e. the size of one character |
| `--rows N` | character rows in the first window |
| `--theme NAME` | `amber` (default), `ice`, `emerald`, `violet`, `noir` |
| `--seed N` | random seed for the iris fibres, starfield and shooting stars |
| `--once` | disable the loop (no flight back, no restart) |
| `--no-crt` | disable the scanlines + vignette overlay |
| `--gpu` / `--raster` / `--text` | force one renderer |

## Renderers

All three draw the same characters in the same colours.

| renderer | how it draws | when it is used |
| --- | --- | --- |
| **gpu** | Uploads the character indices and the per-cell colours as two small textures and draws one quad; a fragment shader looks each glyph up in an atlas, tints it, applies the CRT overlay and writes the pixel. | default, when `moderngl` + `glfw` are installed |
| **raster** | Composes the whole grid into one RGB image with NumPy + Pillow (glyph atlas, CRT mask) and blits it as a single canvas image. | GPU back-end unavailable, or `--raster` |
| **text** | Creates one Tk canvas item per character and updates only the cells that changed, inside a time budget. | no Pillow, or `--text` |

The GPU and bitmap renderers use full 8-bit colour; the text renderer quantizes colours to
24 levels per channel to keep its hex cache small. The GPU frame matches the bitmap frame
to within ±2 out of 255. The character grid is measured with Tk, so all three renderers
line up on the same cells, and a monospace TrueType font is located and set to exactly one
cell of advance.

## GPU renderer

The scene itself stays on the CPU on purpose: at the default size it shades ~12,500
sub-pixels with ~150 small array operations, which is launch-bound — a GPU would spend
more on kernel launches and transfers than the CPU spends on the work. What *is*
expensive on the CPU is turning thousands of characters into pixels, so that is what the
shader does:

* once: the 70 characters of the ramp are rasterised into a glyph-atlas texture;
* every frame: `idx` (one byte per cell, 3 KB) and the per-cell colour (9 KB) are
  uploaded, then two triangles are drawn;
* the shader turns the pixel position into a cell, looks the glyph up in the atlas, tints
  it, applies scanlines/vignette and writes the pixel.

The result, written by the shader into its own framebuffer:

![the heart](docs/preview_heart.png)

### Resizing

A bigger window can pay for itself in two ways: more cells (the CPU scene is linear in
cells, so the frame rate drops) or bigger cells (a little fill rate, the CPU work stays
fixed). The grid therefore has a cell budget (`MAXROWS_WIN = 36`, ~4,500 cells): growing
the window adds cells up to that budget and magnifies the characters beyond it. The aspect
ratio is preserved — the grid is centred and the leftover area stays background — and `S`
always saves a 1:1 frame without magnification or letterbox.

Measured on one machine. All rows use the default 100 FPS target except the first one,
which shows the raw ceiling:

| window | grid | zoom | FPS |
| --- | --- | --- | --- |
| 624×450 (start), `--fps 0` | 104×30 | ×1.00 | 164 |
| 624×450 (start), default | 104×30 | ×1.00 | 100 |
| 1000×700 | 126×36 | ×1.19 | 100 |
| 1600×1000 | 126×36 | ×1.75 | 99–101 |
| 1400×500 | 100×28 | ×1.06 | 101 |
| fullscreen 2560×1440 | 126×36 | ×2.67 | 72–77 |

![a resized window](docs/preview_zoom.png)

### Notes

* The GL window has no vsync (`swap_interval(0)`); pacing is done by `Pacer`, so `--fps N`
  is respected exactly and `--fps 0` runs as fast as the scene allows.
* On a laptop with two GPUs the driver picks the one that opens the GL context (the
  numbers above are from an integrated Intel GPU, which is plenty for one textured quad);
  use the OS graphics settings to prefer the discrete one.
* Fullscreen is fill-rate bound on an integrated GPU; `--no-crt` removes the overlay if
  you want the last few frames.

## Timeline

One loop is `T_LOOP = 37.5` seconds. All times are constants at the top of `eye.py`.

| time (s) | what happens |
| --- | --- |
| 0.0 – 0.9 | eye closed, the frame fades in |
| 0.9 – 3.3 | the lids open |
| 2.35 | first light flash: the pupil contracts and recovers |
| 4.0 / 4.32 | first double blink |
| 7.55 / 7.87 | second double blink |
| 8.4 | second light flash |
| 10.75 | single blink, then the gaze settles on the centre |
| 0.0 – 12.0 | the eye glances around (a new glance every 0.9 – 2.4 s) |
| 12.0 – 28.0 | the camera dives into the pupil, stars rush past |
| 14.5 – 24.5 | the nebula morphs from a cloud into a heart |
| 24.5 – 31.5 | the heart beats in space |
| 31.5 – 36.0 | the camera flies back out of the pupil |
| 36.0 – 36.9 | the lids close again |
| 36.9 – 37.5 | fade to black |
| 37.5 | loop point: the starfield is re-seeded and the loop restarts |

The blinks live in `BLINKS`, the flashes in `FLASHES`, the glances in `GZ`; the acts start
at `T_Z0`, `T_SPACE`, `T_HEART`, `T_RET` and `T_CLOSE`.

## Performance

Scene cost on the CPU, per act, 30×104 cells (scene only, no window — `python bench/compare.py`
prints these numbers; they move a little with machine load):

| act | ms per frame |
| --- | --- |
| opening | 8.6 |
| glances | 9.0 |
| dive | 9.5 |
| heart | 4.9 |
| flight back | 10.8 |
| mean | 8.5 |

Whole window, uncapped (`--fps 0`, one machine, best of a few samples):

| renderer | FPS |
| --- | --- |
| text (per character) | 21 |
| bitmap (NumPy + Pillow) | 80 |
| GPU (one draw call) | 164 |

With the default `--fps 100` the GPU renderer holds 100 FPS at any window size; the bitmap
renderer reaches ~80 FPS at the default size and slows down as the window grows; the text
renderer is the portable fallback and is character-throughput bound.

## Project layout

```
eye.py                   the animation (run this)
Glossary.txt             every constant, function and variable, explained
README.md
docs/preview.png         the eye act, 1:1
docs/preview_heart.png   the heart in space, 1:1
docs/preview_zoom.png    a resized 1600x1000 window, characters magnified 1.85x
bench/compare.py         scene timings, parity check, window FPS
bench/reference.py       frozen snapshot used by the parity check
```

### Checking a change

```bash
python bench/compare.py                # scene ms per act + parity against the snapshot
python bench/compare.py --frames 1800  # more parity samples
python bench/compare.py --fps --fps-target 0   # frame rate of every renderer
```

The parity check renders the same moments of one loop with `eye.py` and
`bench/reference.py` and reports how many cells differ (zero when nothing changed). After
a deliberate change to the scene, refresh the snapshot with
`cp eye.py bench/reference.py`.
