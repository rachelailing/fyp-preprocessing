# Data Preprocessing Plan

## Purpose

The preprocessing pipeline is designed to convert raw YawDD videos into structured mouth-behaviour data that can be used for machine learning classification.

Instead of using raw video directly, the output of preprocessing will be an event-level dataset. Each row will represent one mouth-opening event with extracted MAR-based temporal features.

This prepares the data for contextual classification of:

- Yawning
- Talking/Singing
- Normal mouth movement

---

## Why Preprocessing is Needed

YawDD videos are mainly labelled at the video level. For example, a video may be labelled as `Yawning`, but the actual yawn may only happen during a few seconds of the video.

For this project, the Random Forest classifier needs event-level samples, not full videos. Therefore, the system needs to identify each mouth-opening event and extract features from it.

Example:

| Video-Level Label | Limitation |
|---|---|
| `Yawning video` | The whole video is labelled as yawning, but not every frame contains a yawn |
| `Talking/Singing video` | The whole video contains talking/singing, but mouth-opening events still need to be segmented |
| `Normal video` | Some natural small mouth movements may still appear |

---

## Small-Scale Preprocessing Trial

For the initial implementation, only a few videos can be processed first to verify that the pipeline works.

Suggested sample selection:

- 2 normal videos
- 2 talking/singing videos
- 2 yawning videos

This small trial can be used to show that mouth landmarks, MAR values, and event-level features can be extracted successfully before processing the full dataset.

---

## Suggested Folder Structure

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

---

## Preprocessing Workflow

```text
YawDD Video
↓
Read frame by frame
↓
Detect face and mouth landmarks
↓
Calculate MAR for every frame
↓
Detect MAR rise above threshold
↓
Mark start frame of mouth-opening event
↓
Detect MAR drop below threshold
↓
Mark end frame of mouth-opening event
↓
Extract temporal features
↓
Save one row per mouth-opening event
```

---

## Detailed Steps

### 1. Load Video Input

Each selected YawDD video will be loaded using OpenCV.

The video will be read frame by frame so that the mouth movement can be analysed over time.

---

### 2. Detect Mouth Landmarks

MediaPipe Face Mesh will be used to detect facial landmarks.

Only the mouth landmark points are needed for this preprocessing stage because the project focuses on mouth behaviour and yawning classification.

---

### 3. Calculate Mouth Aspect Ratio (MAR)

For each frame, the Mouth Aspect Ratio will be calculated using mouth landmark distances.

MAR represents how wide the mouth is open.

- Higher MAR = mouth is more open
- Lower MAR = mouth is closed or slightly open

---

### 4. Detect Mouth-Opening Events

A mouth-opening event starts when the MAR value rises above a selected threshold.

The event ends when the MAR value drops below the threshold.

This helps identify complete mouth-opening patterns instead of analysing only individual frames.

---

### 5. Extract Temporal Features

For each mouth-opening event, temporal features will be extracted.

Temporal features describe how the mouth-opening pattern changes over time.

Examples:

| Feature | Description | Why It Is Useful |
|---|---|---|
| `duration_sec` | How long the mouth stays open | Yawning is usually longer than talking |
| `peak_MAR` | Highest MAR value during the event | Yawning usually has a wider opening |
| `opening_speed` | How fast the mouth opens | Talking may involve faster repeated movements |
| `closing_speed` | How fast the mouth closes | Yawning may close more gradually |
| `oscillation_count` | Number of MAR rises and drops during the event | Talking/singing may show repeated mouth movements |
| `baseline_deviation` | Difference from normal mouth-opening baseline | Helps adapt to different users |

---

## Expected Output

The output will be a structured CSV dataset.

Each row represents one mouth-opening event.

Example output table:

| video_name | event_id | start_frame | end_frame | duration_sec | peak_MAR | opening_speed | closing_speed | oscillation_count | baseline_deviation | label |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| subject01_yawning.avi | 1 | 120 | 210 | 3.00 | 0.82 | 0.014 | 0.011 | 1 | 0.35 | yawning |
| subject02_talking.avi | 1 | 45 | 68 | 0.76 | 0.46 | 0.021 | 0.019 | 4 | 0.12 | talking/singing |
| subject03_normal.avi | 1 | 88 | 96 | 0.27 | 0.31 | 0.008 | 0.007 | 1 | 0.04 | normal |

---

## How This Supports the Project

The preprocessing output will be used as input for the Random Forest classifier.

The classifier will learn from the extracted temporal features to distinguish:

- Genuine yawning
- Talking/singing
- Normal mouth movement

This approach improves the basic MAR threshold method because the system does not classify yawning based only on mouth-opening size. Instead, it analyses the full mouth-opening behaviour over time.

---

## Initial Finding to Mention in Presentation

A small-scale preprocessing test can be conducted on selected YawDD videos to verify that:

- Mouth landmarks can be detected successfully
- MAR values can be calculated frame by frame
- Mouth-opening events can be segmented
- Event-level feature rows can be generated
- The output can be saved as a structured CSV file for Random Forest classification

Suggested presentation wording:

> An initial preprocessing test will be conducted on selected YawDD videos to verify that the system can extract mouth landmarks, calculate MAR values, segment mouth-opening events, and generate structured event-level feature data for model training.
