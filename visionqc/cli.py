"""Command-Line Interface for VisionQC.

Provides fully headless, terminal-based execution for industrial quality inspection,
batch processing, model training, benchmarking, and self-contained demos.
"""

import argparse
import json
import os
import sys
import glob
from typing import List

from visionqc.preprocessor import ImagePreprocessor
from visionqc.morphology import MorphologicalAnalyzer
from visionqc.edge_corner import FeatureDetector
from visionqc.segmentation import ImageSegmenter
from visionqc.classifier import DefectClassifier
from visionqc.pipeline import InspectionPipeline


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
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
    """Executes a self-contained demonstration."""
    from data.synthetic_generator import generate_benchmark_dataset

    print("================================================================")
    print(" VisionQC: Automated Industrial Defect Inspection System Demo   ")
    print("================================================================")

    # 1. Generate benchmark dataset
    samples_dir = os.path.join("data", "samples")
    print(f"[*] Generating synthetic industrial component dataset in '{samples_dir}'...")
    generate_benchmark_dataset(output_dir=samples_dir, count_per_class=5)

    # 2. Train classifier
    print("[*] Training PCA + KNN Defect Classifier on synthetic samples...")
    train_images = glob.glob(os.path.join(samples_dir, "train", "*", "*.png"))
    if not train_images:
        print("[!] No training images found. Aborting.")
        return 1

    classifier = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
    X_train = []
    y_train = []
    for p in train_images:
        label = os.path.basename(os.path.dirname(p))
        img = ImagePreprocessor.load_image(p)
        feat = classifier.extract_image_features(img)
        X_train.append(feat)
        y_train.append(label)

    import numpy as np
    train_res = classifier.fit(np.array(X_train), np.array(y_train))
    print(f"    -> Trained on {len(X_train)} samples across {len(set(y_train))} classes.")
    print(f"    -> PCA reduced feature vector to {train_res['pca_components']} components.")
    print(f"    -> Retained {train_res['explained_variance_ratio']*100:.1f}% explained variance.")

    # 3. Run inspection pipeline on test parts
    print(f"\n[*] Running Inspection Pipeline on test components (saving to '{save_dir}')...")
    test_images = sorted(glob.glob(os.path.join(samples_dir, "test", "*", "*.png")))
    pipeline = InspectionPipeline(classifier=classifier)

    results = []
    for test_path in test_images:
        report = pipeline.inspect(test_path, save_dir=save_dir)
        results.append(report)
        status_symbol = "✔ PASS" if report["status"] == "PASS" else "✘ FAIL"
        print(f"    [{status_symbol}] {report['source']:<30} Class: {report['defect_type']:<15} Conf: {report['confidence']*100:.1f}% ({report['execution_time_ms']} ms)")

    # 4. Summary Table
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = total - passed

    print("\n----------------------------------------------------------------")
    print(f"Inspection Summary: {total} parts inspected | {passed} PASSED | {failed} DEFECTIVE")
    print(f"Diagnostic overlays and masks saved to: {os.path.abspath(save_dir)}")
    print("================================================================\n")

    if json_output:
        print(json.dumps(results, indent=2))

    return 0


def main() -> int:
    """CLI entrypoint."""
    args = parse_arguments()

    if args.demo:
        return run_demo(args.save_dir, args.json)

    if args.train:
        from data.synthetic_generator import generate_benchmark_dataset
        samples_dir = os.path.join("data", "samples")
        if not os.path.exists(os.path.join(samples_dir, "train")):
            generate_benchmark_dataset(output_dir=samples_dir, count_per_class=10)

        train_images = glob.glob(os.path.join(samples_dir, "train", "*", "*.png"))
        classifier = DefectClassifier(classifier_type="knn", n_neighbors=3, n_components=0.95)
        X, y = [], []
        for p in train_images:
            label = os.path.basename(os.path.dirname(p))
            img = ImagePreprocessor.load_image(p)
            X.append(classifier.extract_image_features(img))
            y.append(label)

        import numpy as np
        res = classifier.fit(np.array(X), np.array(y))
        classifier.save(args.model_path)
        print(f"Model saved to {args.model_path} with training accuracy: {res['accuracy']*100:.1f}%")
        return 0

    if args.input is None:
        print("Error: Specify --input <file_or_dir> or use --demo to run self-contained inspection.")
        return 1

    # Single or batch processing
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.png")) + glob.glob(os.path.join(args.input, "*.jpg"))
    else:
        files = [args.input]

    if not files:
        print(f"No image files found in {args.input}")
        return 1

    # Load classifier if available
    classifier = None
    if os.path.exists(args.model_path):
        classifier = DefectClassifier()
        classifier.load(args.model_path)

    pipeline = InspectionPipeline(classifier=classifier)
    reports = []

    for f in files:
        report = pipeline.inspect(f, save_dir=args.save_dir)
        reports.append(report)
        if not args.json:
            print(f"Inspected: {report['source']} -> Status: {report['status']} ({report['defect_type']}) [{report['execution_time_ms']}ms]")

    if args.json:
        print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
