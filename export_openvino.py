"""Convert the trained Random Forest into ONNX and OpenVINO formats.

OpenVINO cannot read a Random Forest directly. The obvious route, skl2onnx,
emits ai.onnx.ml.TreeEnsembleClassifier, and OpenVINO's ONNX frontend has no
conversion rule for that operator: it fails with "No conversion rule found for
operations: ai.onnx.ml.TreeEnsembleClassifier".

The way round it is to stop expressing the forest as trees. Hummingbird
rewrites a tree ensemble as a sequence of tensor operations -- comparisons,
gathers and sums -- which are ordinary ONNX operators that OpenVINO does
support. The predictions are unchanged; only the arithmetic used to reach them
is different.

Two files are produced, because they are good at different things:

  models/<name>.onnx      via skl2onnx, for ONNX Runtime. Matches scikit-learn
                          to about 1e-07 and is the fastest backend measured.
  models/<name>.xml/.bin  via Hummingbird, the OpenVINO IR. Runs on Intel
                          runtimes and reaches the same verdicts, though the
                          tensor form accumulates in float32 so individual
                          probabilities can differ by a couple of percent.

Run:
    python export_openvino.py
    python export_openvino.py --model models/yawn_rf_model_binary.pkl
"""

from __future__ import annotations

import argparse
import copy
import warnings
from pathlib import Path

import numpy as np

# Hummingbird and torch are noisy on import and during export, and none of it
# is actionable here.
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the trained classifier to ONNX and OpenVINO IR."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/yawn_rf_model_temporal.pkl"),
        help="Trained scikit-learn classifier to convert.",
    )
    parser.add_argument(
        "--probe-samples",
        type=int,
        default=500,
        help="Random feature vectors used to check the converted models agree.",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=3000,
        help="Single-event predictions used to time each backend.",
    )
    return parser.parse_args()


def integer_label_copy(model):
    """Return a copy whose class labels are integers rather than strings.

    Hummingbird refuses to translate a classifier with string labels. Class
    names do not affect the trees: scikit-learn already stores classes as
    indices internally and maps them back only at predict time, so swapping the
    labels leaves the decision boundaries untouched. The caller keeps the
    original label order and applies it to the converted model's output.
    """

    relabelled = copy.deepcopy(model)
    indices = np.arange(len(model.classes_))
    relabelled.classes_ = indices
    for estimator in relabelled.estimators_:
        estimator.classes_ = indices
    return relabelled


def export_onnx(model, n_features: int, destination: Path) -> Path:
    """Export via skl2onnx, which preserves scikit-learn's arithmetic exactly."""

    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, n_features]))],
        # ZipMap would return a list of dictionaries; a plain probability
        # tensor is what both runtimes and the caller actually want.
        options={id(model): {"zipmap": False}},
        target_opset=15,
    )
    destination.write_bytes(onnx_model.SerializeToString())
    return destination


def export_openvino(model, n_features: int, destination: Path) -> Path:
    """Export via Hummingbird's tensor rewrite, which OpenVINO can read."""

    import torch
    import openvino as ov
    from hummingbird.ml import convert

    torch_model = convert(integer_label_copy(model), "torch").model.eval()

    class ProbabilitiesOnly(torch.nn.Module):
        """Drop Hummingbird's label output and keep the probability tensor."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, features):
            result = self.inner(features)
            return result[1] if isinstance(result, (tuple, list)) else result

    sample = torch.zeros(1, n_features, dtype=torch.float32)
    onnx_path = destination.with_name(destination.stem + "_ov.onnx")

    # dynamo=False selects the legacy exporter. Hummingbird 0.4.12 does not
    # work with the tracer that torch 2.13 uses by default.
    torch.onnx.export(
        ProbabilitiesOnly(torch_model), (sample,), str(onnx_path),
        input_names=["input"], output_names=["probabilities"],
        dynamic_axes={"input": {0: "batch"}, "probabilities": {0: "batch"}},
        opset_version=15, dynamo=False,
    )

    ov.save_model(ov.Core().read_model(str(onnx_path)), str(destination))
    return destination


def probe_features(n_samples: int, n_features: int) -> np.ndarray:
    """Random feature vectors spanning the range real events produce."""

    generator = np.random.default_rng(1)
    # Roughly the observed span of duration, the two speeds, and oscillation
    # rate, so the check covers the region the model is actually used in.
    scale = np.array([5.0, 2.0, 2.0, 5.0])[:n_features]
    return (generator.random((n_samples, n_features)) * scale).astype(np.float32)


def main() -> None:
    args = parse_args()

    import joblib

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}. Run train.py first.")

    model = joblib.load(args.model)
    model.n_jobs = 1
    n_features = model.n_features_in_
    labels = list(model.classes_)
    print(f"Loaded {args.model}")
    print(f"  {len(model.estimators_)} trees | {n_features} features | classes {labels}")

    onnx_path = export_onnx(model, n_features, args.model.with_suffix(".onnx"))
    print(f"\nONNX (skl2onnx)   -> {onnx_path}")

    ir_path = export_openvino(model, n_features, args.model.with_suffix(".xml"))
    print(f"OpenVINO IR       -> {ir_path} and {ir_path.with_suffix('.bin')}")

    # ---------------------------------------------------------------- checks
    import onnxruntime as ort
    import openvino as ov

    probe = probe_features(args.probe_samples, n_features)
    reference = model.predict_proba(probe.astype(np.float64))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_probabilities = session.run(None, {input_name: probe})[1]

    compiled = ov.Core().compile_model(str(ir_path), "CPU")
    request = compiled.create_infer_request()
    output_port = compiled.output(0)
    openvino_probabilities = np.vstack([
        request.infer({0: probe[index:index + 1]})[output_port]
        for index in range(len(probe))
    ])

    print(f"\n{'=' * 62}\nAGREEMENT WITH SCIKIT-LEARN ({len(probe)} random events)\n{'=' * 62}")
    for name, probabilities in (
        ("ONNX Runtime", onnx_probabilities),
        ("OpenVINO", openvino_probabilities),
    ):
        same = bool((reference.argmax(1) == probabilities.argmax(1)).all())
        difference = float(np.abs(reference - probabilities).max())
        print(f"  {name:<14} same verdict: {str(same):<5} | max probability gap: {difference:.2e}")

    # ------------------------------------------------------------- benchmark
    import time

    single_f32 = np.zeros((1, n_features), dtype=np.float32)
    single_f64 = single_f32.astype(np.float64)

    def measure(call) -> float:
        call()  # warm up
        start = time.perf_counter()
        for _ in range(args.benchmark_iterations):
            call()
        return (time.perf_counter() - start) / args.benchmark_iterations * 1000

    print(f"\n{'=' * 62}\nSINGLE-EVENT LATENCY\n{'=' * 62}")
    print(f"  {'backend':<26}{'ms/event':>10}{'events/sec':>14}")
    for name, call in (
        ("scikit-learn", lambda: model.predict_proba(single_f64)),
        ("ONNX Runtime", lambda: session.run(None, {input_name: single_f32})),
        ("OpenVINO (CPU)", lambda: request.infer({0: single_f32})),
    ):
        milliseconds = measure(call)
        print(f"  {name:<26}{milliseconds:>10.4f}{1000 / milliseconds:>14,.0f}")

    print("\nUse a backend from realtime.py with --backend sklearn|onnx|openvino")


if __name__ == "__main__":
    main()
