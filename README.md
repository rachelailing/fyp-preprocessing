# YawDD Preprocessing — Event-Level MAR Feature Extraction

Converts YawDD driver videos into an event-level dataset for Random Forest
classification. Each output row is one detected mouth-opening event described by
six temporal features, not one row per video or per frame.

Pipeline: read frames → detect mouth landmarks (MediaPipe) → compute MAR per
frame → smooth → segment events with hysteresis thresholds → extract six
features → label from filename → write CSV.

Field-by-field description of the output: [OUTPUT_SCHEMA.md](OUTPUT_SCHEMA.md).

---

## Running on Google Colab

Colab is the recommended environment for the full dataset. The run needs roughly
1.5–2.5 hours for ~307 videos.

### 1. Upload `main.py`

Put `main.py` in `MyDrive/dataset/` so it survives a runtime disconnect, or
upload it directly:

```python
from google.colab import files
files.upload()          # select main.py
```

### 2. Mount Drive and extract the dataset

Keep the dataset as an **archive** in Drive and extract it to `/content/`. Do not
extract into Drive and read videos from there — OpenCV seeks inside each file,
and that access pattern over the Drive mount is slow enough to add hours or fail
outright.

```python
import shutil
from google.colab import drive
drive.mount('/content/drive')

shutil.copy('/content/drive/MyDrive/dataset/YawDD.rar.gz', '/content/YawDD.rar.gz')
!pigz -d -k "/content/YawDD.rar.gz"
!apt-get install -y unar -qq
!unar -o /content/YawDD_Dataset "/content/YawDD.rar"
```

The archive is nested — there is a second `YawDD.rar` inside the first. Extract
it too:

```python
!unar -o /content/YawDD_Dataset "/content/YawDD_Dataset/YawDD.rar"
```

Confirm the extraction before starting a long run:

```python
from pathlib import Path
print(len(list(Path('/content/YawDD_Dataset').rglob('*.avi'))), "videos")   # expect ~349
```

### 3. Pin MediaPipe to 1.0.0

```python
!pip install -q mediapipe==1.0.0 opencv-python numpy
```

This matters. `main.py` chooses its landmark backend automatically: it uses
`mp.solutions.face_mesh` when that module exists, and the newer MediaPipe Tasks
`FaceLandmarker` otherwise. Colab's default MediaPipe provides `mp.solutions`,
which would silently switch backends and produce different MAR values from the
verified sample. Version 1.0.0 has no `mp.solutions`, so the Tasks backend is
used consistently.

**Restart the runtime** after installing (Runtime → Restart session), then
re-mount Drive.

### 4. Confirm the backend

```python
import mediapipe as mp
print("mediapipe", mp.__version__)
print("has solutions:", hasattr(mp, "solutions"))   # must be False
```

If this prints `True`, the pin did not take. Stop and fix it — the run would not
be comparable to previously verified output.

### 5. Download the landmark model

```python
!wget -q -O /content/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 6. Run

Write the output **to Drive**, not `/content/`, so a disconnect does not destroy
the partial CSV.

```python
!cp "/content/drive/MyDrive/dataset/main.py" /content/main.py   # skip if uploaded directly

!python -u /content/main.py --all \
  --input-dir "/content/YawDD_Dataset" \
  --output "/content/drive/MyDrive/dataset/yawdd_event_features.csv" \
  --face-landmarker-model "/content/face_landmarker.task"
```

`-u` gives live progress instead of a cell that appears frozen. All the tuned
settings are defaults, so no extra flags are needed.

### 7. If the runtime disconnects

Rows are written and flushed after every video, so work already done is kept.
Re-run the same command with `--resume` to continue:

```python
!python -u /content/main.py --all --resume \
  --input-dir "/content/YawDD_Dataset" \
  --output "/content/drive/MyDrive/dataset/yawdd_event_features.csv" \
  --face-landmarker-model "/content/face_landmarker.task"
```

Without `--resume` the output file is overwritten from empty.

Videos that produce zero events are not recorded in the CSV, so `--resume`
reprocesses them each time. This wastes a little time but cannot create
duplicates.

### 8. Verify the output

```python
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/dataset/yawdd_event_features.csv')
print(len(df), "rows")
print(df['label'].value_counts())
print(df[df.video_name == '1-FemaleNoGlasses-Yawning.avi'][['start_time_sec', 'duration_sec', 'peak_MAR']])
```

Two checks:

- All three classes present, with `normal` not near zero.
- That yawning row should read approximately **1.60 → 5.61 s, duration 4.04,
  peak_MAR 1.089859**. A six-decimal match on `peak_MAR` confirms the backend is
  correct and the run is comparable to the manually verified sample.

---

## Running locally

```bash
pip install -r requirements.txt
```

Place `face_landmarker.task` in `models/` (the download URL is in step 5).

```bash
# 6-video sample: 2 per label, for verification
python main.py --output outputs/sample.csv

# full dataset
python main.py --all --output outputs/yawdd_event_features.csv
```

Local runs need roughly 2 GB of free commit charge. `ImportError: DLL load
failed ... paging file is too small` means the machine is out of virtual memory,
not that anything is wrong with the code.

---

## Command-line options

| Option | Default | Purpose |
|---|---|---|
| `--input-dir` | `YawDD dataset` | Folder searched recursively for `.avi` files. |
| `--output` | `yawdd_event_features.csv` | Output CSV path. |
| `--all` | off | Process every labelled video. Without it, only `--sample-per-label` per class. |
| `--sample-per-label` | `2` | Videos per class when `--all` is not set. |
| `--resume` | off | Append to an existing CSV, skipping videos already recorded. |
| `--threshold-mode` | `adaptive` | `adaptive` = thresholds relative to each driver's baseline. `fixed` = one absolute threshold for everyone. |
| `--rel-rise` | `0.15` | Adaptive rise offset above baseline. |
| `--rel-fall` | `0.12` | Adaptive fall offset above baseline. |
| `--rise-threshold` | `0.30` | Absolute rise threshold (`--threshold-mode fixed` only). |
| `--fall-threshold` | `0.24` | Absolute fall threshold (`--threshold-mode fixed` only). |
| `--baseline-mode` | `global` | `global` = percentile over the whole clip. `early` = first `--baseline-frames` valid frames only. |
| `--baseline-percentile` | `5` | Percentile treated as the resting mouth position. |
| `--baseline-frames` | `30` | Frames used by `--baseline-mode early`. |
| `--smooth-window` | `5` | Rolling-average window over the MAR signal, in frames. |
| `--min-event-frames` | `4` | Events shorter than this frame span are discarded. |
| `--max-frames` | `0` | Debug limit per video. `0` reads the whole video. |
| `--face-landmarker-model` | `models/face_landmarker.task` | MediaPipe Tasks model, required when the installed MediaPipe has no `mp.solutions`. |

### Why `adaptive` is the default

Absolute MAR shifts with facial geometry, mouth size, and camera distance, so a
single fixed threshold does not transfer between drivers. Adaptive mode measures
each event against that driver's own resting mouth position. `fixed` mode is
retained so the adaptive approach can be compared against a conventional
MAR-threshold baseline.

### Why `--baseline-percentile` is low

Several YawDD clips show the labelled behaviour from the opening second, and some
drivers talk through most of their clip. A high percentile — or a baseline taken
only from early frames — can measure a mouth that is mid-yawn and record it as
the resting position, which inflates that driver's threshold and causes real
yawns to be missed. The 5th percentile over the whole signal avoids this.

---

## Known limitations

These are properties of MAR-based segmentation, confirmed by manual verification
against the source videos:

- **Smiles register as events.** Showing teeth increases vertical lip separation
  without the mouth opening. MAR cannot distinguish the two.
- **Head rotation inflates MAR.** Turning the head shortens the apparent mouth
  width, and since width is the denominator, MAR rises with no mouth movement.
- **Video-level labels are applied to every event in a file.** A clip named
  `...-Yawning.avi` can also contain talking and smiling; every event extracted
  from it is labelled `yawning`. Events from mixed `Talking&Yawning` clips are
  excluded for this reason, but single-label clips still carry some label noise.
- **Clips starting mid-behaviour understate `opening_speed`,** because the
  opening phase happened before the first frame.
- **Long continuous speech can merge into one event** when MAR never falls back
  below the fall threshold between syllables. Affected rows have high
  `oscillation_count`, which still separates them from yawns.
