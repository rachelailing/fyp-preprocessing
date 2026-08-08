"""Train a Random Forest to classify mouth-opening events from YawDD.

Input is the event-level CSV produced by main.py. See OUTPUT_SCHEMA.md for the
field definitions.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

# The six engineered features. Identifier columns (video_name, event_id, frame
# and time columns) are deliberately excluded: they encode when an event
# happened rather than what it looked like, and a model can exploit them to
# score well without learning the behaviour.
FEATURE_COLS = [
    "duration_sec",
    "peak_MAR",
    "opening_speed",
    "closing_speed",
    "oscillation_count",
    "baseline_deviation",
]

POSITIVE_CLASS = "yawning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the yawn/non-yawn event classifier."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("yawdd_event_features.csv"),
        help="Event-level CSV produced by main.py.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/yawn_rf_model.pkl"),
        help="Where to save the trained classifier.",
    )
    parser.add_argument(
        "--no-sklearnex",
        action="store_true",
        help=(
            "Disable Intel Extension for Scikit-learn. Use this to measure the "
            "speedup the extension provides."
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=2000,
        help="Single-sample predictions used to measure inference latency.",
    )
    return parser.parse_args()


def enable_intel_acceleration(enabled: bool) -> bool:
    """Patch scikit-learn with Intel's oneDAL-backed implementations.

    Must run before any scikit-learn estimator is imported, which is why every
    sklearn import in this file is deferred until after this call.
    """

    if not enabled:
        print("Intel Extension for Scikit-learn: disabled (--no-sklearnex)")
        return False

    try:
        from sklearnex import patch_sklearn
    except ImportError:
        print(
            "Intel Extension for Scikit-learn: not installed, using stock "
            "scikit-learn. Install with: pip install scikit-learn-intelex"
        )
        return False

    patch_sklearn(verbose=False)
    print("Intel Extension for Scikit-learn: enabled")
    return True


def participant_id(video_name: str) -> str:
    """Identify the person in a clip, so one driver cannot span two splits.

    YawDD names files like '10-FemaleNoGlasses-Talking.avi'. The leading number
    plus gender identifies the participant: the same person is recorded across
    normal/talking/yawning clips and sometimes with and without glasses, while
    '1-Male' and '1-Female' are different people.
    """

    match = re.match(r"^(\d+)-(Male|Female)", video_name, re.IGNORECASE)
    if not match:
        # Fall back to the whole filename so an unparsed name is never silently
        # merged into another participant's group.
        return video_name
    return f"{match.group(1)}-{match.group(2).capitalize()}"


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Feature file not found: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in FEATURE_COLS + ["label", "video_name"] if column not in frame]
    if missing:
        raise SystemExit(f"Feature file is missing columns: {missing}")

    frame["participant"] = frame["video_name"].map(participant_id)
    return frame


def split_by_participant(frame: pd.DataFrame, random_state: int):
    """Split into train/validation/test with no participant in two splits.

    Splitting events at random would place the same driver on both sides, letting
    the model recognise individual mouths instead of learning behaviour, which
    inflates every score. Grouping by participant measures what actually matters:
    generalisation to a driver the model has never seen.
    """

    from sklearn.model_selection import StratifiedGroupKFold

    groups = frame["participant"]
    labels = frame["label"]

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    remainder_idx, test_idx = next(outer.split(frame, labels, groups))

    remainder = frame.iloc[remainder_idx]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=random_state)
    train_local, val_local = next(
        inner.split(remainder, remainder["label"], remainder["participant"])
    )

    return (
        remainder.iloc[train_local],
        remainder.iloc[val_local],
        frame.iloc[test_idx],
    )


def features_of(frame: pd.DataFrame) -> np.ndarray:
    """Extract the feature matrix as a plain array.

    Arrays are used on both sides of fit/predict so the model carries no pandas
    column metadata. A model fitted on a DataFrame warns on every prediction made
    from an array, which is exactly what a deployed real-time app would pass.
    """

    return frame[FEATURE_COLS].to_numpy()


def describe_split(name: str, part: pd.DataFrame) -> None:
    counts = part["label"].value_counts().to_dict()
    print(
        f"  {name:11} {len(part):5} events | "
        f"{part['participant'].nunique():3} participants | {counts}"
    )


def false_positive_rate(y_true, y_pred, positive: str) -> float:
    """Share of non-yawning events wrongly flagged as yawning.

    This is the metric the project exists to reduce: every false positive is an
    alert raised at a driver who was only talking.
    """

    actual_negatives = np.asarray(y_true) != positive
    if not actual_negatives.any():
        return float("nan")
    predicted_positive = np.asarray(y_pred) == positive
    return float((actual_negatives & predicted_positive).sum() / actual_negatives.sum())


def evaluate(model, X, y_true, title: str, positive: str) -> None:
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X)
    labels = sorted(pd.unique(y_true))

    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")
    print(classification_report(y_true, y_pred, zero_division=0))

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print("Confusion matrix")
    print(
        pd.DataFrame(
            matrix,
            index=[f"true:{label}" for label in labels],
            columns=[f"pred:{label}" for label in labels],
        )
    )

    if positive in labels:
        print(f"\nFalse positive rate ({positive}): {false_positive_rate(y_true, y_pred, positive):.4f}")


# Feature subsets used to quantify what the contextual approach adds over a
# conventional MAR threshold. "peak_MAR only" is the closest stand-in for a
# system that decides from mouth openness alone.
ABLATIONS: dict[str, list[str]] = {
    "peak_MAR only (MAR-threshold proxy)": ["peak_MAR"],
    "magnitude only": ["peak_MAR", "baseline_deviation"],
    "temporal only": ["duration_sec", "opening_speed", "closing_speed", "oscillation_count"],
    "all six features": FEATURE_COLS,
}


def run_ablation(train: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> None:
    """Compare feature subsets on identical splits.

    The project's claim is that temporal context separates yawning from talking
    better than mouth openness alone. Training the same model on openness-only
    and on the full feature set measures that difference instead of asserting it.
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_fscore_support

    print(f"\n{'=' * 62}\nABLATION - yawning class on held-out drivers\n{'=' * 62}")
    print(f"  {'feature set':<38} {'prec':>6} {'recall':>7} {'F1':>6} {'FPR':>8}")

    for name, columns in ABLATIONS.items():
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(train[columns].to_numpy(), train["label"])
        predictions = model.predict(test[columns].to_numpy())

        # average=None with a single label returns one-element arrays, which
        # scores the yawning class without collapsing the multiclass target.
        precision, recall, f1, _ = precision_recall_fscore_support(
            test["label"], predictions,
            labels=[POSITIVE_CLASS], average=None, zero_division=0,
        )
        precision, recall, f1 = precision[0], recall[0], f1[0]
        fpr = false_positive_rate(test["label"], predictions, POSITIVE_CLASS)
        print(f"  {name:<38} {precision:6.3f} {recall:7.3f} {f1:6.3f} {fpr:8.4f}")

    print("\n  FPR is the share of non-yawning events wrongly flagged; lower is better.")


def benchmark_latency(model, X, iterations: int) -> None:
    """Measure single-event inference latency, the real-time-relevant metric.

    Training time on a dataset this small is dominated by fixed overhead, so
    per-prediction latency is the honest way to report the effect of CPU
    acceleration and to support the claim that the system runs without a GPU.
    """

    # A deployed app holds one event's features in a plain array. Passing a
    # pandas DataFrame instead adds validation overhead that dwarfs the actual
    # tree traversal and would make the measurement meaningless.
    sample = X[:1]
    model.predict(sample)  # warm up caches and any lazy initialisation

    start = time.perf_counter()
    for _ in range(iterations):
        model.predict(sample)
    elapsed = time.perf_counter() - start

    per_call_ms = (elapsed / iterations) * 1000
    print(f"\n{'=' * 62}\nINFERENCE LATENCY\n{'=' * 62}")
    print(f"  {iterations} single-event predictions in {elapsed:.3f} s")
    print(f"  {per_call_ms:.4f} ms per event")
    # Classification runs once per completed mouth-opening event, not per frame,
    # so this is the budget that matters for responsiveness of the alert.
    print(f"  events classifiable per second: {1000 / per_call_ms:,.0f}")


def main() -> None:
    args = parse_args()

    # Must happen before any sklearn estimator import.
    accelerated = enable_intel_acceleration(not args.no_sklearnex)

    from sklearn.ensemble import RandomForestClassifier
    import joblib

    frame = load_features(args.features)
    print(f"\nLoaded {len(frame)} events from {frame['participant'].nunique()} participants")
    print(frame["label"].value_counts().to_string())

    train, validation, test = split_by_participant(frame, args.random_state)
    print("\nParticipant-disjoint splits:")
    for name, part in (("train", train), ("validation", validation), ("test", test)):
        describe_split(name, part)

    overlap = (
        set(train["participant"]) & set(test["participant"])
        | set(train["participant"]) & set(validation["participant"])
    )
    if overlap:
        raise SystemExit(f"Participant leaked across splits: {sorted(overlap)}")
    print("  no participant appears in more than one split")

    classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",  # counteracts the small normal/yawning classes
        random_state=args.random_state,
        n_jobs=-1,
    )

    start = time.perf_counter()
    classifier.fit(features_of(train), train["label"])
    print(f"\nTrained in {time.perf_counter() - start:.3f} s")

    evaluate(classifier, features_of(validation), validation["label"],
             "THREE-CLASS - VALIDATION", POSITIVE_CLASS)
    evaluate(classifier, features_of(test), test["label"],
             "THREE-CLASS - TEST (held-out drivers)", POSITIVE_CLASS)

    print(f"\n{'=' * 62}\nFEATURE IMPORTANCE\n{'=' * 62}")
    ranking = sorted(zip(FEATURE_COLS, classifier.feature_importances_),
                     key=lambda pair: pair[1], reverse=True)
    for feature, importance in ranking:
        print(f"  {feature:<20} {importance:.4f}  {'#' * int(importance * 60)}")

    run_ablation(train, test, args)

    # Secondary framing: the deployed system only ever needs to decide whether to
    # raise an alert, so yawn vs non-yawn is what the alert logic actually uses.
    binary_train = train["label"].where(train["label"] == POSITIVE_CLASS, "non-yawn")
    binary_test = test["label"].where(test["label"] == POSITIVE_CLASS, "non-yawn")

    binary_classifier = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=args.random_state,
        n_jobs=-1,
    )
    binary_classifier.fit(features_of(train), binary_train)
    evaluate(binary_classifier, features_of(test), binary_test,
             "BINARY yawn vs non-yawn - TEST (held-out drivers)", POSITIVE_CLASS)

    # Parallelism speeds up training but adds thread-pool overhead to every
    # single-event prediction, so the deployed model is switched to serial.
    classifier.n_jobs = 1
    binary_classifier.n_jobs = 1

    benchmark_latency(classifier, features_of(test), args.benchmark_iterations)
    print(f"  Intel Extension for Scikit-learn: {'enabled' if accelerated else 'disabled'}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_out)
    binary_path = args.model_out.with_name(args.model_out.stem + "_binary.pkl")
    joblib.dump(binary_classifier, binary_path)
    print(f"\nSaved {args.model_out} and {binary_path}")


if __name__ == "__main__":
    main()
