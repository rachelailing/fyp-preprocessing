"""Real-time yawn detection from a webcam.

The feature code is imported from main.py rather than reimplemented. A live
event must produce byte-identical features to the ones the classifier was
trained on: any drift between the offline and online paths would silently
invalidate the model, and that class of bug does not announce itself.

Run with:
    python realtime.py
    python realtime.py --model models/yawn_rf_model.pkl   # three-class model
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import main as pipeline

import train

# Taken from train.py rather than restated, because the classifier receives a
# bare array: a column list that drifted out of order here would be read as a
# different set of features entirely, silently and without error.
FEATURE_SETS = {
    "temporal": train.ABLATIONS["temporal only"],
    "all": train.FEATURE_COLS,
}

POSITIVE_CLASS = "yawning"

# YawDD was recorded at 30 fps, and every window in main.py is expressed in
# frames against that rate. A webcam running slower makes the same frame count
# span more real time, which over-smooths the signal: the dips between syllables
# flatten out, consecutive words merge into one long event, and talking starts
# looking like a yawn. Windows are therefore held constant in seconds here and
# converted using the measured frame rate.
TRAINING_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live webcam yawn detection.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yawn_rf_model_temporal.pkl"),
        help="Trained classifier. Must match --feature-set.",
    )
    parser.add_argument(
        "--feature-set",
        choices=tuple(FEATURE_SETS),
        default="temporal",
        help=(
            "temporal: durations, speeds and oscillation only, which transfer "
            "across camera setups. all: adds peak MAR and baseline deviation, "
            "which are larger on a desk webcam than in YawDD's mirror footage "
            "and bias the model toward calling wide speech a yawn."
        ),
    )
    parser.add_argument(
        "--face-landmarker-model",
        type=Path,
        default=Path("models/face_landmarker.task"),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=1,
        help="Capture device index. 1 is the Logitech C615; 0 is the built-in camera.",
    )
    parser.add_argument(
        "--capture-fps",
        type=int,
        default=30,
        help=(
            "Frame rate requested from the camera. The C615 drops to 15 fps in "
            "dim light regardless, which the seconds-based windows absorb."
        ),
    )
    parser.add_argument("--rel-rise", type=float, default=0.15)
    parser.add_argument("--rel-fall", type=float, default=0.12)
    parser.add_argument(
        "--smooth-seconds",
        type=float,
        default=5 / TRAINING_FPS,
        help="Rolling-average span. Matches main.py's 5 frames at 30 fps.",
    )
    parser.add_argument(
        "--min-event-seconds",
        type=float,
        default=4 / TRAINING_FPS,
        help="Shortest event kept. Matches main.py's 4 frames at 30 fps.",
    )
    parser.add_argument(
        "--min-yawn-seconds",
        type=float,
        default=1.0,
        help=(
            "Shortest event that may be called a yawn. A yawn is a slow reflex "
            "lasting several seconds; in YawDD only 5 percent of them finish "
            "inside one second, while 76 percent of non-yawn events do. An "
            "explicit floor is worth having alongside the model because it "
            "holds regardless of how the camera scales mouth openings."
        ),
    )
    parser.add_argument(
        "--yawn-threshold",
        type=float,
        default=0.5,
        help=(
            "Probability required before an event is called a yawn. Raising it "
            "trades recall for precision, which is the trade the alert wants: a "
            "false alert at a driver who was only talking costs trust."
        ),
    )
    parser.add_argument(
        "--baseline-percentile",
        type=float,
        default=5.0,
        help="Percentile of recent MAR treated as the driver's resting mouth.",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=300,
        help=(
            "Frames of history the resting baseline is measured over, about ten "
            "seconds at 30 fps. Offline the baseline spans a whole clip, which a "
            "live stream cannot do; a rolling window tracks the driver instead "
            "and re-adapts if they change posture or distance from the camera."
        ),
    )
    parser.add_argument(
        "--min-mouth-width-ratio",
        type=float,
        default=0.85,
        help=(
            "Drop frames whose mouth width falls below this fraction of the "
            "recent median, which removes turned-head frames that inflate MAR. "
            "Matches main.py's offline gate. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=45,
        help="Frames collected before the baseline is trusted and detection starts.",
    )
    parser.add_argument(
        "--alert-seconds",
        type=float,
        default=3.0,
        help="How long the yawn alert stays on screen after an event is classified.",
    )
    parser.add_argument(
        "--record-signal",
        type=Path,
        default=None,
        help=(
            "Write the per-frame MAR trace to this CSV. Segmentation thresholds "
            "can then be re-tuned offline against the real signal instead of "
            "being guessed at from the events they happened to produce."
        ),
    )
    parser.add_argument(
        "--no-sklearnex",
        action="store_true",
        help="Disable Intel Extension for Scikit-learn.",
    )
    return parser.parse_args()


def enable_intel_acceleration(enabled: bool) -> bool:
    """Patch scikit-learn before the classifier is unpickled."""

    if not enabled:
        return False
    try:
        from sklearnex import patch_sklearn
    except ImportError:
        return False
    patch_sklearn(verbose=False)
    return True


class EventTracker:
    """Segment mouth-opening events from a live MAR stream.

    Mirrors detect_events in main.py, but consumes one frame at a time and
    keeps a rolling baseline, because a live stream has no future frames to
    take a percentile over.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.recent = deque(maxlen=args.baseline_window)
        # Sized for the slowest frame rate worth supporting; how many of these
        # samples are actually averaged is decided per frame from measured fps.
        self.smoothing: deque[float] = deque(maxlen=32)
        self.event_values: list[tuple[int, float]] = []
        self.in_event = False
        self.baseline = 0.0
        self.smoothed_mar = 0.0
        self.widths: deque[float] = deque(maxlen=args.baseline_window)

    @property
    def ready(self) -> bool:
        return len(self.recent) >= self.args.warmup_frames

    def is_frontal(self, mouth_width_px: float) -> bool:
        """Reject frames where the head is turned away from the camera.

        Mouth width is the denominator of MAR, so turning the head shortens it
        in projection and raises MAR without the mouth moving at all. Yawning
        barely changes mouth width, so an unusually narrow mouth is far more
        likely to be a turned head than an open one. main.py applies this
        against the whole clip's median; live, the median comes from the same
        rolling window that feeds the baseline.
        """

        import numpy as np

        self.widths.append(mouth_width_px)
        if self.args.min_mouth_width_ratio <= 0 or len(self.widths) < self.args.warmup_frames:
            return True

        median = float(np.median(self.widths))
        if median <= 0:
            return True
        return mouth_width_px >= self.args.min_mouth_width_ratio * median

    def update(self, frame_index: int, raw_mar: float, fps: float) -> pipeline.Event | None:
        """Feed one frame's MAR; return an Event on the frame it completes."""

        import numpy as np

        self.recent.append(raw_mar)
        self.smoothing.append(raw_mar)

        span = max(1, min(round(self.args.smooth_seconds * fps), len(self.smoothing)))
        mar = float(np.mean(list(self.smoothing)[-span:]))
        self.smoothed_mar = mar

        if not self.ready:
            return None

        self.baseline = float(np.percentile(self.recent, self.args.baseline_percentile))
        rise = self.baseline + self.args.rel_rise
        fall = self.baseline + self.args.rel_fall

        if not self.in_event:
            if mar >= rise:
                self.in_event = True
                self.event_values = [(frame_index, mar)]
            return None

        self.event_values.append((frame_index, mar))
        if mar > fall:
            return None

        # Mouth has returned to rest, so the event is complete.
        start_frame = self.event_values[0][0]
        end_frame = self.event_values[-1][0]
        completed = None
        min_frames = max(2, round(self.args.min_event_seconds * fps))
        if end_frame - start_frame + 1 >= min_frames:
            completed = pipeline.Event(start_frame, end_frame, self.event_values.copy())

        self.in_event = False
        self.event_values = []
        return completed


def classify(
    model,
    event: pipeline.Event,
    fps: float,
    baseline: float,
    args: argparse.Namespace,
) -> tuple[str, float, float, dict]:
    """Score one completed event, returning (label, yawn probability, ms, features)."""

    import numpy as np

    # Reusing event_to_row is the point: it is the same function that produced
    # every training row, so the live feature vector cannot drift from it.
    row = pipeline.event_to_row("webcam", 0, event, fps, baseline, label="")
    # Derived the same way train.py derives it, from the same two columns.
    row["oscillation_rate"] = train.oscillation_rate(
        row["oscillation_count"], row["duration_sec"]
    )
    columns = FEATURE_SETS[args.feature_set]
    features = np.array([[float(row[column]) for column in columns]])

    start = time.perf_counter()
    probabilities = model.predict_proba(features)[0]
    latency_ms = (time.perf_counter() - start) * 1000

    classes = list(model.classes_)
    yawn_probability = float(probabilities[classes.index(POSITIVE_CLASS)])

    # Thresholding the probability instead of taking the argmax makes the
    # precision/recall trade a tunable parameter rather than one fixed at 0.5.
    # The duration floor is applied on top: an event too brief to be a yawn is
    # rejected however confident the model is about it.
    long_enough = float(row["duration_sec"]) >= args.min_yawn_seconds
    if yawn_probability >= args.yawn_threshold and long_enough:
        label = POSITIVE_CLASS
    else:
        alternatives = [
            (probability, name)
            for probability, name in zip(probabilities, classes)
            if name != POSITIVE_CLASS
        ]
        label = str(max(alternatives)[1])

    return label, yawn_probability, latency_ms, row


def draw_overlay(frame, tracker: EventTracker, state: dict, args: argparse.Namespace) -> None:
    """Render the live MAR signal, thresholds, and the last decision."""

    import cv2

    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 96), (24, 24, 24), -1)

    mar = tracker.smoothed_mar
    baseline = tracker.baseline
    rise = baseline + args.rel_rise

    if not tracker.ready:
        status, colour = "CALIBRATING", (0, 200, 255)
    elif tracker.in_event:
        status, colour = "MOUTH OPEN", (0, 200, 255)
    else:
        status, colour = "MONITORING", (0, 220, 120)

    cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
    cv2.putText(
        frame,
        f"MAR {mar:.3f}   baseline {baseline:.3f}   trigger {rise:.3f}",
        (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1,
    )
    cv2.putText(
        frame,
        f"{state['fps']:.1f} fps   inference {state['latency_ms']:.2f} ms   "
        f"events {state['events']}   off-angle skipped {state['skipped']}",
        (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1,
    )

    # MAR bar with the trigger threshold marked, so the decision is legible on
    # screen rather than only in the printed log.
    bar_x, bar_y, bar_w, bar_h = 12, height - 44, width - 24, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    span = max(rise * 1.6, 0.1)
    filled = int(min(mar / span, 1.0) * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), colour, -1)
    trigger_x = bar_x + int(min(rise / span, 1.0) * bar_w)
    cv2.line(frame, (trigger_x, bar_y - 4), (trigger_x, bar_y + bar_h + 4), (0, 0, 255), 2)

    if state["last_label"]:
        cv2.putText(
            frame,
            f"last event: {state['last_label']}  p(yawn) {state['last_confidence']:.2f}",
            (12, height - 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )

    if state["alert_until"] > time.time():
        cv2.rectangle(frame, (0, 104), (width, 168), (0, 0, 200), -1)
        cv2.putText(
            frame, "YAWN DETECTED - DROWSINESS ALERT",
            (16, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )


def main() -> None:
    args = parse_args()

    accelerated = enable_intel_acceleration(not args.no_sklearnex)
    import joblib

    pipeline.load_dependencies()
    import cv2

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run train.py first.")
    model = joblib.load(args.model)

    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Try --camera 0 for the built-in one."
        )

    # 640x480 matches YawDD's resolution, so the landmarks the model sees live
    # are framed the same way as the ones it was trained on.
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, args.capture_fps)

    print(f"Intel Extension for Scikit-learn: {'enabled' if accelerated else 'disabled'}")
    print(f"Model: {args.model}")
    print("Press q to quit.\n")

    tracker = EventTracker(args)
    signal_log: list[tuple] | None = [] if args.record_signal else None
    session_start = time.time()
    state = {
        "fps": 0.0,
        "latency_ms": 0.0,
        "events": 0,
        "skipped": 0,
        "last_label": "",
        "last_confidence": 0.0,
        "alert_until": 0.0,
    }

    # Webcams report unreliable CAP_PROP_FPS, and fps scales every speed and
    # duration feature, so it is measured from actual frame arrival instead.
    frame_times: deque[float] = deque(maxlen=30)
    measured_fps = 30.0
    frame_index = 0

    with pipeline.create_face_landmark_detector(args.face_landmarker_model, fps=30.0) as detector:
        while True:
            success, frame = capture.read()
            if not success:
                break

            now = time.perf_counter()
            frame_times.append(now)
            if len(frame_times) > 1:
                elapsed = frame_times[-1] - frame_times[0]
                if elapsed > 0:
                    measured_fps = (len(frame_times) - 1) / elapsed
                    state["fps"] = measured_fps

            frame = cv2.flip(frame, 1)  # mirror, so the preview reads naturally
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            landmarks = detector.detect(rgb_frame, frame_index)

            if landmarks:
                mar = pipeline.calculate_mar(landmarks, width, height)
                frontal = tracker.is_frontal(
                    pipeline.mouth_width(landmarks, width, height)
                )
                state["skipped"] += 0 if frontal else 1
                if mar == mar and frontal:  # drop NaN, as the offline pipeline does
                    if signal_log is not None:
                        signal_log.append((
                            frame_index, round(time.time() - session_start, 4),
                            round(measured_fps, 2), round(mar, 6),
                            round(tracker.smoothed_mar, 6), round(tracker.baseline, 6),
                            int(tracker.in_event),
                        ))
                    event = tracker.update(frame_index, mar, measured_fps)
                    if event is not None:
                        label, probability, latency_ms, row = classify(
                            model, event, measured_fps, tracker.baseline, args
                        )
                        state.update(
                            events=state["events"] + 1,
                            last_label=label,
                            last_confidence=probability,
                            latency_ms=latency_ms,
                        )
                        # The feature vector is printed alongside the verdict so a
                        # surprising call can be traced to the numbers behind it
                        # rather than guessed at.
                        print(
                            f"event {state['events']:3d} | {label:<16} p(yawn)={probability:.2f} | "
                            f"dur {row['duration_sec']:5.2f}s  peak {row['peak_MAR']:.3f}  "
                            f"dev {row['baseline_deviation']:.3f}  osc {row['oscillation_count']:2d} | "
                            f"{latency_ms:.2f} ms"
                        )
                        if label == POSITIVE_CLASS:
                            state["alert_until"] = time.time() + args.alert_seconds

            draw_overlay(frame, tracker, state, args)
            cv2.imshow("Drowsiness detection - yawn classifier", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_index += 1

    capture.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {frame_index} frames, classified {state['events']} event(s).")

    if signal_log is not None:
        import csv

        args.record_signal.parent.mkdir(parents=True, exist_ok=True)
        with args.record_signal.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "frame", "time_sec", "fps", "raw_mar",
                "smoothed_mar", "baseline", "in_event",
            ])
            writer.writerows(signal_log)
        print(f"Wrote {len(signal_log)} signal rows to {args.record_signal}")


if __name__ == "__main__":
    main()
