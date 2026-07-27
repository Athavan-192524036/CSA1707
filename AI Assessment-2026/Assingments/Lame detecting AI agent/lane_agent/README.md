# Lane Detection AI Agent

A weather-robust lane-marking detection agent for road images and video,
built with classical computer vision (OpenCV) and a stateful decision
layer that behaves like an agent — not just a one-shot detector.

It:
- Detects lane markings (white and yellow) in single images or video/webcam streams
- Adapts its preprocessing to lighting/weather conditions (bright sun, overcast, fog, dusk, rain, low light)
- Fits smooth polynomial lane lines, computes road curvature and the vehicle's offset from lane center
- Tracks state across frames (temporal smoothing) so it survives brief occlusions or glare
- Emits decisions — `OK`, `WARNING`, `CRITICAL`, `LOW_CONFIDENCE`, `LOST` — with human-readable messages, like a real ADAS lane-departure module would
- Works out of the box with **no training data required**

## Project layout

```
lane_agent/
├── lane_agent/
│   ├── __init__.py        # package exports
│   ├── config.py          # all tunable parameters (AgentConfig)
│   ├── preprocessing.py   # weather-robust preprocessing (CLAHE, gamma, denoise)
│   ├── detector.py         # per-frame CV pipeline (LaneDetector)
│   ├── agent.py            # stateful agent: smoothing + decisions (LaneDetectionAgent)
│   └── utils.py            # I/O and debug-view helpers
├── run_image.py             # CLI: run on a single image
├── run_video.py              # CLI: run on a video file or webcam
├── tests/
│   └── test_detector.py     # pytest suite using a synthetic road (no dataset needed)
├── sample_data/              # example synthetic road image + output
├── requirements.txt
└── README.md
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Single image

```bash
python run_image.py --input path/to/road.jpg --output out.jpg --debug
```

`--debug` also saves `out_debug.jpg`, a side-by-side of the annotated frame
and the bird's-eye binarized lane mask — useful for tuning thresholds.

### Video file

```bash
python run_video.py --input path/to/drive.mp4 --output out.mp4 --log telemetry.jsonl
```

`--log` writes one JSON object per frame (state, offset, curvature,
confidence) — handy for downstream analysis or plotting.

### Webcam (live preview)

```bash
python run_video.py --input 0
```

Press `q` to quit the preview window. Add `--no-preview` on a headless
machine (e.g. a server without a display).

### Quick test that everything works

```bash
python -m pytest tests/ -v
```

The tests generate their own synthetic road images, so this runs with
zero external dependencies or datasets — good for CI or a first sanity
check right after cloning.

### Using it as a library

```python
import cv2
from lane_agent import LaneDetectionAgent, AgentConfig

agent = LaneDetectionAgent(AgentConfig())
frame = cv2.imread("road.jpg")

annotated_frame, status = agent.process_frame(frame)

print(status.state)        # "OK" / "WARNING" / "CRITICAL" / "LOW_CONFIDENCE" / "LOST"
print(status.message)      # human-readable explanation
print(status.offset_m)     # signed distance from lane center, in meters
print(status.curvature_m)  # estimated road curvature radius, in meters
```

## How it handles varying weather conditions

Real driving footage swings wildly in brightness, contrast and noise.
`preprocessing.py` adapts to that before any lane features are extracted:

| Condition | Technique | File |
|---|---|---|
| Night / underexposed | Auto gamma correction from measured mean brightness | `auto_gamma_correct` |
| Fog / overcast / low contrast | CLAHE (adaptive histogram equalization) on the LAB lightness channel | `apply_clahe` |
| Rain / sensor noise | Laplacian-variance noise estimate → conditional bilateral filtering | `auto_denoise` |
| Any of the above | Confidence penalty fed into the agent's decision layer, so it reports `LOW_CONFIDENCE` instead of a shaky false-confident lane | `estimate_condition_confidence_penalty` |

Color thresholding runs in **HLS** space rather than RGB because hue/lightness
separate lane-paint color from ambient lighting much better — a white line
stays "high lightness, low saturation" whether it's noon or dusk.

## Architecture

1. **Preprocess** the frame (gamma / CLAHE / denoise, auto-adapted to conditions)
2. **Threshold**: combine an HLS color mask (white + yellow paint) with a
   Sobel-x gradient mask (catches edges color alone misses, e.g. faded
   markings)
3. **Mask** to a trapezoidal region of interest (road area only)
4. **Warp** to a bird's-eye view via perspective transform, so lane lines
   become roughly vertical/parallel — easier to fit reliably
5. **Sliding-window search** finds lane pixel clusters bottom-to-top and
   fits a 2nd-order polynomial per lane line
6. **Curvature & offset**: polynomial coefficients converted from pixel to
   real-world units (meters), using a calibrated `xm_per_pix` / `ym_per_pix`
7. **Agent layer**: smooths fits over a rolling window of frames, tracks
   how long a lane has been lost, and converts offset/confidence into a
   decision (`OK` / `WARNING` / `CRITICAL` / `LOW_CONFIDENCE` / `LOST`)
8. **Overlay**: lane fill + boundary lines warped back onto the original
   frame, plus an on-screen status HUD

## Tuning for your own camera

The defaults in `config.py` assume a forward-facing dashcam around 720p.
For a different mount position/angle/resolution, adjust:

- `roi_vertices_ratio` — the trapezoid that isolates the road
- `perspective_src_ratio` / `perspective_dst_ratio` — points defining the
  bird's-eye warp; get these right by finding four points that form a
  rectangle on a flat, straight piece of road in a calibration photo
- `xm_per_pix`, `ym_per_pix` — real-world scale of the warped image, needed
  for accurate curvature/offset in meters
- `departure_offset_thresh_m` / `departure_offset_critical_m` — how far off
  center triggers `WARNING` / `CRITICAL`

## Limitations & next steps

This is a classical-CV pipeline — fast, dependency-light, and needs no
training data, but it can still struggle with: extreme occlusion (car
directly ahead blocking markings), completely unmarked roads, or very
sharp curves outside the ROI. To go further:

- **Swap in a deep-learning segmentation model** (e.g. ENet, SCNN,
  LaneNet, or a U-Net) trained on **TuSimple** or **CULane** for pixel-level
  lane segmentation that's more robust to occlusion and worn-out paint.
  The `LaneDetector.fit_lanes()` interface (input frame → `LaneResult`) is
  designed so you can drop a model-based detector in behind the same
  interface without touching `agent.py`.
- **Add lane-type classification** (solid vs. dashed, single vs. double)
  for more nuanced departure warnings (e.g. don't warn on a dashed line
  you're legally allowed to cross).
- **Fuse with IMU/GPS** for a more robust offset estimate when vision
  confidence is low.
