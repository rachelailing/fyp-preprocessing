from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

cv2 = None
mp = None
np = None
base_options_module = None
face_landmarker_module = None
image_module = None
running_mode_module = None


# Output order is kept explicit so the generated CSV always matches the
# expected Random Forest training schema from the preprocessing plan.
OUTPUT_COLUMNS = [
    "video_name",
    "event_id",
    "start_frame",
    "end_frame",
    "start_time_sec",
    "end_time_sec",
    "duration_sec",
    "peak_MAR",
    "opening_speed",
    "closing_speed",
    "oscillation_count",
    "baseline_deviation",
    "label",
]

# MediaPipe Face Mesh/Face Landmarker lip landmark indices used for MAR.
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
VERTICAL_MOUTH_PAIRS = [(13, 14), (81, 178), (311, 402)]


@dataclass(frozen=True)
class VideoJob:
    """A single video to process together with its inferred class label."""

    path: Path
    label: str


@dataclass(frozen=True)
class Event:
    """A detected mouth-opening event and its frame-level smoothed MAR values."""

    start_frame: int
    end_frame: int
    values: list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    """Read command-line settings for sample/full preprocessing runs."""

    parser = argparse.ArgumentParser(
        description="Convert YawDD videos into event-level MAR feature rows."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("YawDD dataset"),
        help="Folder containing YawDD .avi videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("yawdd_event_features.csv"),
        help="CSV file to write event-level features.",
    )
    parser.add_argument(
        "--sample-per-label",
        type=int,
        default=2,
        help="Number of videos per label for verification. Use 0 with --all for full processing.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all single-behaviour labelled videos instead of the small sample.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Append to an existing output CSV and skip videos already recorded "
            "in it. Use this to continue a run that was cut short."
        ),
    )
    parser.add_argument(
        "--threshold-mode",
        choices=("adaptive", "fixed"),
        default="adaptive",
        help=(
            "adaptive: thresholds are offsets above each driver's resting MAR "
            "baseline, so mouth size and camera distance do not shift them. "
            "fixed: one absolute threshold for every driver, kept for the "
            "baseline MAR-threshold comparison."
        ),
    )
    # Adaptive thresholds. The fall offset is deliberately well above the
    # baseline so continuous speech drops below it between syllables instead of
    # merging a whole talking sequence into one long event.
    parser.add_argument("--rel-rise", type=float, default=0.15)
    parser.add_argument("--rel-fall", type=float, default=0.12)
    # Absolute thresholds, used only when --threshold-mode fixed.
    parser.add_argument("--rise-threshold", type=float, default=0.30)
    parser.add_argument("--fall-threshold", type=float, default=0.24)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--min-event-frames", type=int, default=4)
    parser.add_argument(
        "--min-mouth-width-ratio",
        type=float,
        default=0.85,
        help=(
            "Drop frames whose mouth width falls below this fraction of the "
            "clip's median, which removes turned-head frames that inflate MAR. "
            "Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--yawn-dominance-ratio",
        type=float,
        default=0.6,
        help=(
            "In yawning clips, keep only events whose baseline_deviation reaches "
            "this fraction of the strongest event in the same clip. Prevents "
            "smiles and speech inside a yawning file being labelled as yawns. "
            "Set 0 to keep every event."
        ),
    )
    parser.add_argument(
        "--baseline-mode",
        choices=("global", "early"),
        default="global",
        help=(
            "global: resting MAR is a low percentile over the whole clip, which "
            "survives videos that start mid-behaviour. early: percentile over "
            "the first --baseline-frames valid frames only."
        ),
    )
    parser.add_argument(
        "--baseline-percentile",
        type=float,
        default=5.0,
        help=(
            "Percentile treated as the resting mouth position. Kept low because "
            "some drivers talk or yawn through most of their clip, which leaves "
            "only a small fraction of genuinely resting frames."
        ),
    )
    parser.add_argument(
        "--baseline-frames",
        type=int,
        default=30,
        help="Early valid MAR frames used for the baseline when --baseline-mode early.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debugging limit per video. 0 means read the whole video.",
    )
    parser.add_argument(
        "--face-landmarker-model",
        type=Path,
        default=Path("models/face_landmarker.task"),
        help=(
            "MediaPipe Tasks face landmarker model. Required only when the "
            "installed mediapipe package does not provide mp.solutions.face_mesh."
        ),
    )
    return parser.parse_args()


def load_dependencies() -> None:
    """Import heavy video/landmark libraries only when processing is requested."""

    global cv2, mp, np
    global base_options_module, face_landmarker_module, image_module, running_mode_module

    try:
        import cv2 as cv2_module
        import mediapipe as mp_module
        import numpy as np_module
    except ModuleNotFoundError as exc:
        print(
            f"Missing dependency: {exc.name}. "
            "Install project dependencies with: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    cv2 = cv2_module
    mp = mp_module
    np = np_module

    # Python 3.14 currently installs a MediaPipe package that exposes the newer
    # Tasks API instead of the older mp.solutions.face_mesh API. The code below
    # loads the Tasks modules only when that newer API is required.
    if not hasattr(mp, "solutions"):
        from mediapipe.tasks.python.core import base_options as imported_base_options
        from mediapipe.tasks.python.vision import face_landmarker as imported_face_landmarker
        from mediapipe.tasks.python.vision.core import image as imported_image
        from mediapipe.tasks.python.vision.core import (
            vision_task_running_mode as imported_running_mode,
        )

        base_options_module = imported_base_options
        face_landmarker_module = imported_face_landmarker
        image_module = imported_image
        running_mode_module = imported_running_mode


class FaceLandmarkDetector:
    """Small wrapper that hides differences between MediaPipe landmark APIs."""

    def __init__(self, detector, backend: str, fps: float | None = None) -> None:
        self.detector = detector
        self.backend = backend
        self.fps = fps or 30.0

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.detector.close()

    def detect(self, rgb_frame, frame_index: int):
        """Return one face's landmarks for a frame, or None when no face is found."""

        if self.backend == "solutions":
            result = self.detector.process(rgb_frame)
            if not result.multi_face_landmarks:
                return None
            return result.multi_face_landmarks[0]

        # Tasks VIDEO mode requires monotonically increasing timestamps.
        timestamp_ms = int((frame_index / self.fps) * 1000)
        mp_image = image_module.Image(
            image_format=image_module.ImageFormat.SRGB,
            data=rgb_frame,
        )
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]


def create_face_landmark_detector(model_path: Path, fps: float | None = None) -> FaceLandmarkDetector:
    """Create a face landmark detector for either supported MediaPipe backend."""

    if hasattr(mp, "solutions"):
        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return FaceLandmarkDetector(detector, backend="solutions", fps=fps)

    if not model_path.exists():
        raise RuntimeError(
            "This mediapipe install uses the Tasks API and needs a face landmarker "
            f"model at {model_path}. Download face_landmarker.task and pass it with "
            "--face-landmarker-model. Official model URL: "
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )

    options = face_landmarker_module.FaceLandmarkerOptions(
        base_options=base_options_module.BaseOptions(
            model_asset_path=str(model_path),
            delegate=base_options_module.BaseOptions.Delegate.CPU,
        ),
        running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
        num_faces=1,
    )
    detector = face_landmarker_module.FaceLandmarker.create_from_options(options)
    return FaceLandmarkDetector(detector, backend="tasks", fps=fps)


def infer_label(path: Path) -> str | None:
    """Infer the event label from the YawDD filename."""

    name = path.stem.lower()
    # Mixed talking+yawning videos are skipped because video-level labels are
    # not reliable enough for event-level training rows.
    if "talking&yawning" in name or "talking& yawning" in name:
        return None
    if "yawning" in name:
        return "yawning"
    if "talking" in name:
        return "talking/singing"
    if "normal" in name:
        return "normal"
    return None


def discover_videos(input_dir: Path, sample_per_label: int, process_all: bool) -> list[VideoJob]:
    """Find labelled AVI videos and optionally keep only a small balanced sample."""

    jobs_by_label: dict[str, list[VideoJob]] = {
        "normal": [],
        "talking/singing": [],
        "yawning": [],
    }

    for video_path in sorted(input_dir.rglob("*.avi")):
        label = infer_label(video_path)
        if label is not None:
            jobs_by_label[label].append(VideoJob(video_path, label))

    jobs: list[VideoJob] = []
    for label in ("normal", "talking/singing", "yawning"):
        label_jobs = jobs_by_label[label]
        if process_all:
            jobs.extend(label_jobs)
        else:
            jobs.extend(label_jobs[:sample_per_label])

    return jobs


def landmark_xy(landmarks, index: int, width: int, height: int) -> np.ndarray:
    """Convert one normalized landmark into pixel coordinates."""

    point = landmarks[index]
    return np.array([point.x * width, point.y * height], dtype=np.float64)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate straight-line distance between two 2D points."""

    return float(np.linalg.norm(a - b))


def calculate_mar(face_landmarks, width: int, height: int) -> float:
    """Calculate Mouth Aspect Ratio from mouth width and vertical opening."""

    # Classic FaceMesh returns an object with .landmark, while Tasks returns a
    # plain list. This line normalizes both formats.
    landmarks = getattr(face_landmarks, "landmark", face_landmarks)
    left = landmark_xy(landmarks, LEFT_MOUTH, width, height)
    right = landmark_xy(landmarks, RIGHT_MOUTH, width, height)
    horizontal = euclidean(left, right)

    if horizontal <= 1e-6:
        return math.nan

    # Averaging several vertical lip distances reduces sensitivity to a single
    # noisy landmark point.
    vertical_distances = [
        euclidean(
            landmark_xy(landmarks, upper, width, height),
            landmark_xy(landmarks, lower, width, height),
        )
        for upper, lower in VERTICAL_MOUTH_PAIRS
    ]
    return float(np.mean(vertical_distances) / horizontal)


def mouth_width(face_landmarks, width: int, height: int) -> float:
    """Return the pixel distance between the two mouth corners."""

    landmarks = getattr(face_landmarks, "landmark", face_landmarks)
    return euclidean(
        landmark_xy(landmarks, LEFT_MOUTH, width, height),
        landmark_xy(landmarks, RIGHT_MOUTH, width, height),
    )


def drop_rotated_frames(
    samples: list[tuple[int, float, float]],
    min_width_ratio: float,
) -> list[tuple[int, float]]:
    """Discard frames where the head is turned away from the camera.

    Mouth width is the denominator of MAR, and turning the head shortens it in
    projection, so a rotated head raises MAR with no mouth movement at all.
    Yawning barely changes mouth width, so a frame whose width falls well below
    the clip's median is far more likely to be a turned head than an open mouth.
    """

    if min_width_ratio <= 0 or not samples:
        return [(frame, mar) for frame, mar, _ in samples]

    median_width = float(np.median([width for _, _, width in samples]))
    if median_width <= 0:
        return [(frame, mar) for frame, mar, _ in samples]

    minimum = min_width_ratio * median_width
    return [(frame, mar) for frame, mar, width in samples if width >= minimum]


def smooth_mar(raw_values: Iterable[tuple[int, float]], window_size: int) -> list[tuple[int, float]]:
    """Apply a rolling average to reduce frame-to-frame landmark jitter."""

    if window_size <= 1:
        return list(raw_values)

    window: deque[float] = deque(maxlen=window_size)
    smoothed: list[tuple[int, float]] = []
    for frame_index, mar in raw_values:
        window.append(mar)
        smoothed.append((frame_index, float(np.mean(window))))
    return smoothed


def extract_mar_signal(
    video_path: Path,
    model_path: Path,
    smooth_window: int,
    max_frames: int,
    min_width_ratio: float = 0.0,
) -> tuple[list[tuple[int, float]], float]:
    """Read a video frame by frame and return its smoothed MAR signal plus FPS."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    # Mouth width is carried alongside MAR so turned-head frames can be removed
    # once the clip's median width is known.
    raw_values: list[tuple[int, float, float]] = []
    frame_index = 0

    # The detector is created per video so Tasks VIDEO timestamps restart from 0.
    with create_face_landmark_detector(model_path, fps=fps) as detector:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if max_frames and frame_index >= max_frames:
                break

            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            face_landmarks = detector.detect(rgb_frame, frame_index)

            # Frames with no detected face are dropped instead of interpolated,
            # matching the plan's choice to avoid fabricating mouth movement.
            if face_landmarks:
                mar = calculate_mar(face_landmarks, width, height)
                if not math.isnan(mar):
                    raw_values.append(
                        (frame_index, mar, mouth_width(face_landmarks, width, height))
                    )

            frame_index += 1

    capture.release()
    frontal_values = drop_rotated_frames(raw_values, min_width_ratio)
    return smooth_mar(frontal_values, smooth_window), float(fps)


def estimate_baseline(
    mar_signal: list[tuple[int, float]],
    baseline_mode: str,
    baseline_frames: int,
    percentile: float,
) -> float:
    """Estimate the driver's resting (closed-mouth) MAR for one video.

    Global mode reads the percentile across the whole clip. This is deliberate:
    several YawDD clips already show the labelled behaviour in their opening
    second, so sampling only early frames can measure a mouth that is mid-yawn
    and report it as the resting position. Because a driver's mouth is closed
    for most of any clip, a low percentile over the full signal lands on the
    true resting value no matter when the behaviour occurs.
    """

    if not mar_signal:
        return 0.0

    if baseline_mode == "early":
        values = [mar for _, mar in mar_signal[:baseline_frames]]
    else:
        values = [mar for _, mar in mar_signal]

    if not values:
        return 0.0

    return float(np.percentile(values, percentile))


def resolve_thresholds(baseline_mar: float, args: argparse.Namespace) -> tuple[float, float]:
    """Return the (rise, fall) MAR thresholds to segment one video's events.

    Adaptive mode measures each event against that driver's own resting mouth
    position, which is what the methodology requires: facial geometry, mouth
    size, and camera distance all shift the absolute MAR value, so a single
    fixed threshold does not transfer across drivers.
    """

    if args.threshold_mode == "fixed":
        return args.rise_threshold, args.fall_threshold

    return baseline_mar + args.rel_rise, baseline_mar + args.rel_fall


def detect_events(
    mar_signal: list[tuple[int, float]],
    rise_threshold: float,
    fall_threshold: float,
    min_event_frames: int,
) -> list[Event]:
    """Segment mouth-opening events using hysteresis thresholding."""

    events: list[Event] = []
    in_event = False
    event_values: list[tuple[int, float]] = []

    for frame_index, mar in mar_signal:
        # Event starts only when MAR rises above the higher threshold.
        if not in_event and mar >= rise_threshold:
            in_event = True
            event_values = [(frame_index, mar)]
            continue

        if in_event:
            event_values.append((frame_index, mar))
            # Event ends only when MAR falls below the lower threshold. This
            # prevents one event from being split by small threshold flickers.
            if mar <= fall_threshold:
                start_frame = event_values[0][0]
                end_frame = event_values[-1][0]
                if end_frame - start_frame + 1 >= min_event_frames:
                    events.append(Event(start_frame, end_frame, event_values.copy()))
                in_event = False
                event_values = []

    # Keep a final event that reaches the end of the video without dropping
    # below the fall threshold.
    if in_event and event_values:
        start_frame = event_values[0][0]
        end_frame = event_values[-1][0]
        if end_frame - start_frame + 1 >= min_event_frames:
            events.append(Event(start_frame, end_frame, event_values.copy()))

    return events


def count_oscillations(values: list[float]) -> int:
    """Count local MAR direction changes inside an event."""

    if len(values) < 3:
        return 0

    direction_changes = 0
    previous_direction = 0

    for previous, current in zip(values, values[1:]):
        diff = current - previous
        direction = 1 if diff > 0 else -1 if diff < 0 else 0
        if direction == 0:
            continue
        if previous_direction and direction != previous_direction:
            direction_changes += 1
        previous_direction = direction

    return direction_changes


def event_to_row(
    video_name: str,
    event_id: int,
    event: Event,
    fps: float,
    baseline_mar: float,
    label: str,
) -> dict[str, object]:
    """Convert one detected event into the final six-feature CSV row."""

    frame_numbers = [frame for frame, _ in event.values]
    mar_values = [mar for _, mar in event.values]
    peak_index = int(np.argmax(mar_values))
    peak_mar = float(mar_values[peak_index])

    start_frame = frame_numbers[0]
    peak_frame = frame_numbers[peak_index]
    end_frame = frame_numbers[-1]
    start_mar = float(mar_values[0])
    end_mar = float(mar_values[-1])

    # Avoid division by zero when the peak is at the first or final frame.
    opening_seconds = max((peak_frame - start_frame) / fps, 1.0 / fps)
    closing_seconds = max((end_frame - peak_frame) / fps, 1.0 / fps)
    duration_sec = (end_frame - start_frame + 1) / fps

    return {
        "video_name": video_name,
        "event_id": event_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_time_sec": round(start_frame / fps, 4),
        "end_time_sec": round(end_frame / fps, 4),
        "duration_sec": round(duration_sec, 4),
        "peak_MAR": round(peak_mar, 6),
        "opening_speed": round((peak_mar - start_mar) / opening_seconds, 6),
        "closing_speed": round((peak_mar - end_mar) / closing_seconds, 6),
        "oscillation_count": count_oscillations(mar_values),
        "baseline_deviation": round(peak_mar - baseline_mar, 6),
        "label": label,
    }


def process_video(job: VideoJob, args: argparse.Namespace) -> list[dict[str, object]]:
    """Run MAR extraction, event detection, and feature generation for one video."""

    mar_signal, fps = extract_mar_signal(
        job.path,
        args.face_landmarker_model,
        smooth_window=args.smooth_window,
        max_frames=args.max_frames,
        min_width_ratio=args.min_mouth_width_ratio,
    )
    baseline_mar = estimate_baseline(
        mar_signal,
        baseline_mode=args.baseline_mode,
        baseline_frames=args.baseline_frames,
        percentile=args.baseline_percentile,
    )
    rise_threshold, fall_threshold = resolve_thresholds(baseline_mar, args)
    events = detect_events(
        mar_signal,
        rise_threshold=rise_threshold,
        fall_threshold=fall_threshold,
        min_event_frames=args.min_event_frames,
    )

    # Logged per video so the segmentation of any single row can be traced back
    # to the thresholds that produced it.
    print(
        f"    baseline MAR {baseline_mar:.4f} | "
        f"rise {rise_threshold:.4f} | fall {fall_threshold:.4f}"
    )

    rows = [
        event_to_row(job.path.name, event_id, event, fps, baseline_mar, job.label)
        for event_id, event in enumerate(events, start=1)
    ]

    return keep_dominant_yawns(rows, job.label, args.yawn_dominance_ratio)


def keep_dominant_yawns(
    rows: list[dict[str, object]],
    label: str,
    dominance_ratio: float,
) -> list[dict[str, object]]:
    """Drop weak events from yawning clips instead of labelling them as yawns.

    A YawDD yawning clip also contains talking and smiling, and the video-level
    label would mark those as yawns. A real yawn opens far wider than either, so
    events well below the clip's strongest event are dropped rather than
    mislabelled. They are discarded instead of relabelled because their true
    behaviour cannot be determined automatically.
    """

    if dominance_ratio <= 0 or label != "yawning" or not rows:
        return rows

    strongest = max(float(row["baseline_deviation"]) for row in rows)
    minimum = dominance_ratio * strongest
    kept = [row for row in rows if float(row["baseline_deviation"]) >= minimum]

    # Renumber so event_id stays contiguous within the video.
    for new_id, row in enumerate(kept, start=1):
        row["event_id"] = new_id

    if len(kept) < len(rows):
        print(f"    dropped {len(rows) - len(kept)} sub-dominant event(s) from yawning clip")

    return kept


def completed_videos(output_path: Path) -> set[str]:
    """Return the video names already present in a partially written CSV."""

    if not output_path.exists():
        return set()

    with output_path.open("r", newline="", encoding="utf-8") as csv_file:
        return {row["video_name"] for row in csv.DictReader(csv_file)}


def main() -> None:
    """Coordinate the full preprocessing run from CLI arguments to CSV output."""

    args = parse_args()
    load_dependencies()

    if args.threshold_mode == "fixed":
        if args.fall_threshold >= args.rise_threshold:
            raise ValueError("--fall-threshold must be lower than --rise-threshold")
    elif args.rel_fall >= args.rel_rise:
        raise ValueError("--rel-fall must be lower than --rel-rise")

    print(f"Threshold mode: {args.threshold_mode} | baseline mode: {args.baseline_mode}")

    jobs = discover_videos(args.input_dir, args.sample_per_label, args.all)
    if not jobs:
        raise RuntimeError(f"No labelled .avi videos found under {args.input_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Videos already recorded are skipped so an interrupted run can be continued
    # instead of restarted. Without --resume the output is rewritten from empty.
    already_done = completed_videos(args.output) if args.resume else set()
    if already_done:
        jobs = [job for job in jobs if job.path.name not in already_done]
        print(f"Resuming: {len(already_done)} video(s) already recorded, {len(jobs)} left")

    # Rows are appended and flushed per video, so losing the process costs only
    # the video in flight rather than the whole run.
    open_mode = "a" if already_done else "w"
    written = 0

    with args.output.open(open_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_COLUMNS)
        if not already_done:
            writer.writeheader()

        for index, job in enumerate(jobs, start=1):
            print(f"[{index}/{len(jobs)}] Processing {job.path.name} ({job.label})")
            video_rows = process_video(job, args)
            writer.writerows(video_rows)
            csv_file.flush()
            written += len(video_rows)
            print(f"    detected {len(video_rows)} event(s) | {written} row(s) saved")

    print(f"Wrote {written} event row(s) to {args.output}")


if __name__ == "__main__":
    main()
