"""Command-Line Interface for VisionQC.

Provides fully headless, terminal-based execution for industrial quality inspection,
batch processing, model training, benchmarking, and self-contained demos.
"""

from typing import List, Optional
import argparse
import json
import os
import sys
import glob

from visionqc.preprocessor import ImagePreprocessor
from visionqc.classifier import DefectClassifier
from visionqc.pipeline import InspectionPipeline


def parse_arguments() -> argparse.Namespace:
    """Parses command-line configuration options."""
    parser = argparse.ArgumentParser(
        prog="visionqc",
        description="VisionQC: Automated Industrial Defect Detection & Quality Inspection Pipeline",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run self-contained end-to-end demo on benchmark industrial parts.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to an input image or directory of images for inspection.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="full",
        choices=["full", "preprocess", "morphology", "edge", "segment"],
        help="Inspection pipeline stage to execute.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save diagnostic overlays and output masks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON results to stdout.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the PCA + KNN defect classification model on sample data.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/defect_classifier.pkl",
        help="Path to save or load the trained classifier.",
    )
    return parser.parse_args()


def run_demo(save_dir: str, json_output: bool) -> int:
    """Executes a fully self-contained automated demonstration."""
    from data.synthetic_generator import generate_benchmark_dataset
    import numpy as np

    print("================================================================")
    print(" VisionQC: Automated Industrial Defect Inspection System Demo   ")
    print("================================================================")

    samples_dir = os.path.join("data", "samples")
    print(f"[*] Generating synthetic industrial component dataset in '{samples_dir}'...")
    generate_benchmark_dataset(output_dir=samples_dir, count_per_class=5)

    print("[*] Training PCA + KNN Defect Classifier on synthetic samples...")
    train_paths = sorted(glob.glob(os.path.join(samples_dir, "train", "*", "*.png")))
    if not train_paths:
        print("[!] No training images found in dataset. Aborting demo.")
        return 1

    model = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
    x_list, y_list = [], []
    for img_path in train_paths:
        cls_name = os.path.basename(os.path.dirname(img_path))
        mat = ImagePreprocessor.load_image(img_path)
        x_list.append(model.extract_image_features(mat))
        y_list.append(cls_name)

    metrics = model.fit(np.array(x_list), np.array(y_list))
    print(f"    -> Trained on {len(x_list)} samples across {len(set(y_list))} classes.")
    print(f"    -> PCA reduced feature vector to {metrics['pca_components']} components.")
    print(f"    -> Retained {metrics['explained_variance_ratio'] * 100:.1f}% explained variance.")

    print(f"\n[*] Running Inspection Pipeline on test components (saving to '{save_dir}')...")
    test_paths = sorted(glob.glob(os.path.join(samples_dir, "test", "*", "*.png")))
    pipeline = InspectionPipeline(classifier=model)

    records = []
    for test_file in test_paths:
        rep = pipeline.inspect(test_file, save_dir=save_dir)
        records.append(rep)
        badge = "✔ PASS" if rep["status"] == "PASS" else "✘ FAIL"
        print(
            f"    [{badge}] {rep['source']:<30} Class: {rep['defect_type']:<15} "
            f"Conf: {rep['confidence'] * 100:.1f}% ({rep['execution_time_ms']} ms)"
        )

    inspected_count = len(records)
    passed_count = sum(1 for r in records if r["status"] == "PASS")
    failed_count = inspected_count - passed_count

    print("\n----------------------------------------------------------------")
    print(f"Inspection Summary: {inspected_count} parts inspected | {passed_count} PASSED | {failed_count} DEFECTIVE")
    print(f"Diagnostic overlays and masks saved to: {os.path.abspath(save_dir)}")
    print("================================================================\n")

    if json_output:
        print(json.dumps(records, indent=2))

    return 0


def main() -> int:
    """CLI application entrypoint."""
    args = parse_arguments()

    if args.demo:
        return run_demo(args.save_dir, args.json)

    if args.train:
        from data.synthetic_generator import generate_benchmark_dataset
        import numpy as np

        samples_dir = os.path.join("data", "samples")
        if not os.path.exists(os.path.join(samples_dir, "train")):
            generate_benchmark_dataset(output_dir=samples_dir, count_per_class=10)

        train_files = sorted(glob.glob(os.path.join(samples_dir, "train", "*", "*.png")))
        clf = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
        feats, targets = [], []
        for path in train_files:
            lbl = os.path.basename(os.path.dirname(path))
            im = ImagePreprocessor.load_image(path)
            feats.append(clf.extract_image_features(im))
            targets.append(lbl)

        res = clf.fit(np.array(feats), np.array(targets))
        clf.save(args.model_path)
        print(f"Model saved to {args.model_path} with training accuracy: {res['accuracy'] * 100:.1f}%")
        return 0

    if args.input is None:
        print("Error: Specify --input <file_or_dir> or use --demo to run self-contained inspection.")
        return 1

    if os.path.isdir(args.input):
        targets = sorted(glob.glob(os.path.join(args.input, "*.png")) + glob.glob(os.path.join(args.input, "*.jpg")))
    else:
        targets = [args.input]

    if not targets:
        print(f"No image files found in {args.input}")
        return 1

    classifier: Optional[DefectClassifier] = None
    if os.path.isfile(args.model_path):
        classifier = DefectClassifier()
        classifier.load(args.model_path)

    pipe = InspectionPipeline(classifier=classifier)
    outputs = []

    for target_path in targets:
        rep = pipe.inspect(target_path, save_dir=args.save_dir)
        outputs.append(rep)
        if not args.json:
            print(f"Inspected: {rep['source']} -> Status: {rep['status']} ({rep['defect_type']}) [{rep['execution_time_ms']}ms]")

    if args.json:
        print(json.dumps(outputs if len(outputs) > 1 else outputs[0], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
