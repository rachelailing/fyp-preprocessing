# Output Schema — `yawdd_event_features.csv`

Field reference for the CSV produced by `main.py`. Setup and run instructions
are in [README.md](README.md).

**One row = one detected mouth-opening event.** Not one row per video and not one
row per frame. A single video contributes as many rows as it has detected events:
a talking clip typically yields 10–16, a normal clip 0–1. Videos where no event
is detected contribute no rows at all.

All 13 columns are written in the fixed order below, defined by `OUTPUT_COLUMNS`
in `main.py`.

---

## Column reference

| # | Column | Type | Unit | Role |
|---|---|---|---|---|
| 1 | `video_name` | text | — | identifier |
| 2 | `event_id` | integer | — | identifier |
| 3 | `start_frame` | integer | frames | provenance |
| 4 | `end_frame` | integer | frames | provenance |
| 5 | `start_time_sec` | float | seconds | provenance |
| 6 | `end_time_sec` | float | seconds | provenance |
| 7 | `duration_sec` | float | seconds | **model feature** |
| 8 | `peak_MAR` | float | ratio | **model feature** |
| 9 | `opening_speed` | float | MAR/second | **model feature** |
| 10 | `closing_speed` | float | MAR/second | **model feature** |
| 11 | `oscillation_count` | integer | count | **model feature** |
| 12 | `baseline_deviation` | float | ratio | **model feature** |
| 13 | `label` | text | — | target |

Columns 7–12 are the six features used for classification. Columns 1–6 are for
traceability — locating an event in the source video — and must not be used as
model inputs. `start_frame` in particular correlates with when a behaviour
happens to occur in a clip, which is an artefact of how the dataset was recorded,
not a property of the behaviour.

---

## Field definitions

### `video_name`
Filename of the source video, including extension. Not unique — one row per event
means a video appears once per detected event. The unique key for a row is
`video_name` + `event_id`.

### `event_id`
Sequential number of the event **within that video**, in chronological order,
starting at 1. Restarts for each video, so it is not a global identifier. Because
it counts only detected events, the numbering does not correspond to the true
sequence of mouth movements in the clip — an event the segmenter missed leaves no
gap.

### `start_frame`, `end_frame`
First and last frame index of the event, zero-based, counted from the start of
the video. These are the frames where the smoothed MAR signal crossed the rise
and fall thresholds.

To locate an event in a video player: `time = frame ÷ fps`. YawDD videos run at
**29.971 fps**, not exactly 30.

### `start_time_sec`, `end_time_sec`
`start_frame ÷ fps` and `end_frame ÷ fps`, rounded to 4 decimals. Frame rate is
read from the video file, defaulting to 30.0 only if the file does not report one.

These mark **threshold crossings, not the visible limits of the movement.** An
event begins when MAR rises above `baseline + 0.15` — the mouth has already
started opening before this. It ends when MAR falls below `baseline + 0.12`, at
which point the mouth may still be slightly open. Expect roughly a tenth of a
second of offset at each end when comparing against manual observation.

### `duration_sec`
Event length: `(end_frame − start_frame + 1) ÷ fps`, rounded to 4 decimals.

The `+ 1` counts frames inclusively — an event spanning frames 15 to 306 contains
292 frames. This is why `duration_sec` is one frame (0.033 s) longer than
`end_time_sec − start_time_sec`.

*Discriminative value:* yawns sustain an open mouth longer than a spoken syllable.
*Caveat:* long uninterrupted speech can merge into one event, producing a
talking row with a yawn-like duration. Such rows carry a high `oscillation_count`.

### `peak_MAR`
Maximum smoothed MAR reached during the event, rounded to 6 decimals.

MAR is computed per frame as the mean of three vertical lip distances divided by
mouth width, using MediaPipe landmark indices:

```
MAR = mean(|13−14|, |81−178|, |311−402|) / |61−291|
```

Averaging three vertical pairs rather than one reduces sensitivity to a single
badly localised landmark. Distances are in pixels; the ratio is dimensionless.

*Note:* a manual 4-point measurement (single upper/lower lip pair plus corners)
gives a **higher** value than this column, because the two off-centre pairs sit
nearer the mouth corners and open less, pulling the mean down. That difference is
expected, not an error.

*Caveat:* absolute values are not comparable between drivers — face size and
camera distance shift them. Use `baseline_deviation` for cross-driver comparison.

### `opening_speed`
Rate of MAR increase from event start to peak, in MAR units per second:

```
(peak_MAR − MAR at start_frame) / max((peak_frame − start_frame) / fps, 1/fps)
```

The `max(..., 1/fps)` floor prevents division by zero when the peak falls on the
first frame.

*Discriminative value:* yawns open gradually; speech opens abruptly.
*Caveat:* understated when a clip is already mid-behaviour at frame 0, since the
opening phase occurred before recording started.

### `closing_speed`
Rate of MAR decrease from peak to event end, same units and same zero-division
floor:

```
(peak_MAR − MAR at end_frame) / max((end_frame − peak_frame) / fps, 1/fps)
```

*Discriminative value:* yawns close slowly; speech closes sharply.

### `oscillation_count`
Number of direction reversals in the smoothed MAR signal within the event. Each
switch from rising to falling, or falling to rising, counts as one. Frames where
MAR is unchanged are skipped. Returns 0 for events shorter than 3 samples.

*Discriminative value:* the single most reliable separator in practice. A yawn is
one smooth arc (typically 1–7). Continuous speech reverses constantly (20–35).
This feature still separates merged talking events from yawns even when
`duration_sec` does not.

### `baseline_deviation`
`peak_MAR − baseline_MAR`, rounded to 6 decimals, where `baseline_MAR` is that
driver's resting mouth position: the 5th percentile of the smoothed MAR signal
across the whole clip.

*Purpose:* normalises for mouth size and camera distance, making event magnitude
comparable between drivers in a way that raw `peak_MAR` is not.

*Note:* the baseline itself is not written to the CSV. Recover it as
`peak_MAR − baseline_deviation`. It is also printed per video in the run log.

### `label`
Ground-truth class, inferred from the filename, one of:

| Value | Filename contains |
|---|---|
| `normal` | `normal` |
| `talking/singing` | `talking` |
| `yawning` | `yawning` |

Files whose names contain `talking&yawning` are excluded entirely, and files
matching none of the patterns are skipped — including the dash-mounted YawDD
videos, whose filenames carry no behaviour keyword.

**This is a video-level label applied to every event extracted from that file.**
A clip named `...-Yawning.avi` may also contain talking and smiling; those events
are still labelled `yawning`. Manual verification of six videos found this
happens in practice. Treat `label` as a noisy target, not verified ground truth.

---

## Reading the file

```python
import pandas as pd

df = pd.read_csv('yawdd_event_features.csv')

FEATURE_COLS = [
    'duration_sec', 'peak_MAR', 'opening_speed',
    'closing_speed', 'oscillation_count', 'baseline_deviation',
]

X = df[FEATURE_COLS]
y = df['label']
```

Do not include `start_frame`, `end_frame`, `start_time_sec`, `end_time_sec`, or
`event_id` in `X`. They encode when an event happened rather than what it looked
like, and a model can exploit them to score well without learning the behaviour.

Because rows from the same video are not independent — one driver contributes
many events — a random train/test split lets the same person appear on both
sides. Splitting by participant gives a more honest estimate of how the model
generalises to unseen drivers.

---

## Known data quality issues

Confirmed by manually checking detected events against the source videos:

1. **Smiles are recorded as events.** Showing teeth raises vertical lip
   separation without opening the mouth. Roughly 3 of 25 events in the verified
   sample were smiles.
2. **Head rotation produces false events.** Turning the head shortens apparent
   mouth width; because width is the denominator, MAR rises with no mouth
   movement.
3. **Label noise from video-level labels**, as described under `label` above.
4. **Merged speech events.** Continuous talking can produce a single long event
   with an atypically large `duration_sec`; `oscillation_count` remains reliable
   for these.
5. **Events truncated at clip boundaries.** A behaviour already underway at frame
   0 yields an event with a distorted `opening_speed`.
