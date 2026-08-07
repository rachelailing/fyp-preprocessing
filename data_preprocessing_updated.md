# Data Preprocessing Plan

## Purpose

This pipeline converts raw YawDD videos into an event-level dataset for Random Forest classification.

Each row will represent one mouth-opening event, described by six engineered MAR-based temporal features. The classifier will use these events to separate three categories:

- Yawning
- Talking/singing
- Normal mouth movement

## Why Event-Level Data Is Needed

YawDD labels videos, not frames. A video labelled `Yawning` contains a yawn for only a few seconds out of its full duration; the rest is normal driving footage. A `Talking/Singing` video contains many separate mouth-opening events, not one continuous event.

A Random Forest model trained on whole-video labels would learn from mostly irrelevant frames. Segmenting each video into individual mouth-opening events, and labelling each event on its own, gives the model a training set where every row actually corresponds to the behaviour it is labelled as.

| Video-Level Label | What It Actually Contains |
|---|---|
| `Yawning video` | One or two genuine yawns, surrounded by neutral frames |
| `Talking/Singing video` | Multiple short mouth-opening events from speech |
| `Normal video` | Occasional small, incidental mouth movement |

## Small-Scale Pipeline Verification

Before running the full YawDD set, a small subset of six videos will confirm the pipeline works end to end:

- 2 normal videos
- 2 talking/singing videos
- 2 yawning videos

This step exists to catch landmark-detection failures, threshold miscalibration, or broken event segmentation early, before spending time processing the full dataset.

### Folder Structure

```text
YawDD_sample/
├── normal/
│   ├── normal_01.avi
│   └── normal_02.avi
├── talking/
│   ├── talking_01.avi
│   └── talking_02.avi
└── yawning/
    ├── yawning_01.avi
    └── yawning_02.avi
```

## Pipeline Overview

```text
YawDD video
↓
Read frame by frame (OpenCV)
↓
Detect face and mouth landmarks (MediaPipe Face Mesh)
↓
Calculate MAR per frame
↓
Smooth MAR signal (rolling average, reduces landmark jitter)
↓
Apply hysteresis thresholding to detect event start/end
↓
Discard events shorter than minimum duration (noise filtering)
↓
Extract six temporal features per event
↓
Label event using video-level label (single-behaviour videos)
↓
Save one row per mouth-opening event to CSV
```

## Detailed Steps

### 1. Load Video Input

Each selected YawDD video is loaded frame by frame using OpenCV. Frame-by-frame reading is necessary because MAR must be tracked as a continuous signal, not sampled at isolated points.

### 2. Detect Mouth Landmarks

MediaPipe Face Mesh detects facial landmarks per frame. Only the landmark points around the mouth are retained, since eye and other facial regions are outside this project's scope.

Frames where the face is not detected, the mouth is occluded, or landmark confidence is low are dropped rather than interpolated. Interpolating missing landmarks risks fabricating mouth movement that never happened, which would corrupt the MAR signal at exactly the points where detection is least reliable.

### 3. Calculate Mouth Aspect Ratio (MAR)

MAR is computed per frame as the ratio of vertical mouth opening to horizontal mouth width, using the standard landmark-distance formula:

```
MAR = (vertical mouth distance) / (horizontal mouth distance)
```

A closed mouth produces a low MAR. A wide-open mouth produces a high MAR. This project follows the MAR convention already established in Chapter 2 (Section: MAR-Based Yawning Detection) rather than introducing a new formulation.

### 4. Smooth the MAR Signal

Raw MAR values fluctuate frame to frame due to landmark jitter, even when the mouth is not actually moving. A short rolling average (e.g. 3–5 frames) is applied before threshold detection. Without this step, jitter alone can trigger spurious event boundaries, inflating the oscillation count feature for reasons unrelated to actual mouth behaviour.

### 5. Detect Mouth-Opening Events (Hysteresis Thresholding)

A single fixed threshold is not enough to segment events cleanly, because a MAR signal hovering near the threshold produces rapid start/stop flickering. This project uses two thresholds instead of one:

- **Rise threshold**: MAR must exceed this value to *start* an event.
- **Fall threshold** (set lower than the rise threshold): MAR must drop below this value to *end* an event.

This gap between the two thresholds (hysteresis) prevents a single mouth-opening event from being fragmented into several false micro-events when MAR oscillates near the boundary.

Events shorter than a minimum duration (e.g. 3–4 frames) are discarded, since these are more likely to be landmark noise than genuine mouth movement.

### 6. Extract Temporal Features

Each surviving event is reduced to six engineered features:

| Feature | Definition | Why It Separates Yawning From Talking |
|---|---|---|
| `peak_MAR` | Maximum MAR value reached during the event | Yawns typically reach a wider maximum opening than speech |
| `duration_sec` | Event length in seconds (end frame − start frame, divided by FPS) | Yawns sustain the open-mouth phase longer than a single spoken syllable |
| `opening_speed` | Rate of MAR increase from event start to peak | Yawns open gradually; speech opens and closes rapidly and repeatedly |
| `closing_speed` | Rate of MAR decrease from peak to event end | Yawns close more slowly than speech-driven mouth movements |
| `oscillation_count` | Number of local rises and falls in MAR within the event window | Talking/singing produces multiple small oscillations; a yawn is closer to one smooth arc |
| `baseline_deviation` | Peak MAR minus that subject's resting (closed-mouth) MAR baseline | Normalises for differences in mouth size and camera distance across subjects |

`baseline_deviation` requires a per-subject resting MAR, computed from a short window of neutral, mouth-closed frames at the start of each video before any event detection begins.

### 7. Label Events

Because each YawDD video in the sample set contains a single dominant behaviour, the video-level label (`yawning`, `talking/singing`, `normal`) is applied directly to every event extracted from that video. This is acceptable for the small-scale trial but will need per-event manual verification once mixed-behaviour videos are introduced, since a "talking" video can still contain an incidental non-speech mouth movement.

### 8. Save Output

One row per event is appended to a structured CSV. This is the direct input format for the Random Forest classifier — no further feature engineering happens after this stage.

## Expected Output Schema

| video_name | event_id | start_frame | end_frame | duration_sec | peak_MAR | opening_speed | closing_speed | oscillation_count | baseline_deviation | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| subject01_yawning.avi | 1 | 120 | 210 | 3.00 | 0.82 | 0.014 | 0.011 | 1 | 0.35 | yawning |
| subject02_talking.avi | 1 | 45 | 68 | 0.76 | 0.46 | 0.021 | 0.019 | 4 | 0.12 | talking/singing |
| subject03_normal.avi | 1 | 88 | 96 | 0.27 | 0.31 | 0.008 | 0.007 | 1 | 0.04 | normal |

## Known Limitations of the Small-Scale Trial

- Video-level labelling means events inside a "talking" video are assumed to all be talking. This is a simplification, not a guarantee.
- Hysteresis threshold values are provisional and will need tuning once more videos are processed and the MAR distribution across subjects is better understood.
- Baseline MAR is computed from a short neutral window per video; if a subject's mouth is not fully closed during that window, `baseline_deviation` will be miscalibrated for that subject.
- Six videos are enough to confirm the pipeline runs correctly, not enough to draw conclusions about feature separability between classes.

## How This Supports the Project

The Random Forest classifier learns to separate yawning from talking/singing and normal movement using these six features, rather than a single-frame MAR threshold. This is the mechanism by which the project moves from "is the mouth open" to "does this mouth-opening event behave like a yawn."

## Suggested Presentation Wording

> A small-scale preprocessing trial was conducted on six YawDD videos to verify that the pipeline can detect mouth landmarks, calculate MAR values, segment mouth-opening events using hysteresis thresholding, and generate structured event-level feature data for Random Forest training.
