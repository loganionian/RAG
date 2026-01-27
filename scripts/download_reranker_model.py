"""Download and cache the cross-encoder reranker model locally for offline use."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download cross-encoder reranker model for offline use.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_RERANKER_MODEL,
        help=f"HuggingFace model name to download. Default: {DEFAULT_RERANKER_MODEL}",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the model. Default: models",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading cross-encoder model: {args.model_name}")
    print(f"Output directory: {output_dir.absolute()}")

    try:
        from sentence_transformers import CrossEncoder

        # Download and initialize model
        print("\nInitializing model (this will download if not cached)...")
        model = CrossEncoder(args.model_name)

        # Create a clean model name for the folder
        model_folder = args.model_name.replace("/", "_")
        save_path = output_dir / model_folder

        # Save model to local path
        print(f"Saving model to: {save_path}")
        model.save(str(save_path))
        print(f"\nModel saved to: {save_path}")

        # Test the saved model
        print("\nTesting saved model...")
        loaded_model = CrossEncoder(str(save_path))
        test_pairs = [
            ("What is machine learning?", "Machine learning is a type of AI."),
            ("What is machine learning?", "The weather is nice today."),
        ]
        scores = loaded_model.predict(test_pairs)
        print(f"Test scores: {scores}")
        print(f"  - Relevant pair score: {scores[0]:.4f}")
        print(f"  - Irrelevant pair score: {scores[1]:.4f}")

        if scores[0] > scores[1]:
            print("\nModel validation passed: relevant pair scored higher than irrelevant pair.")
        else:
            print("\nWarning: Model validation unexpected - relevant pair should score higher.")

        print(f"\nTo use this model, set RERANKER_MODEL_PATH environment variable:")
        print(f"  set RERANKER_MODEL_PATH={save_path}")
        print(f"\nOr the model will be auto-detected from: models/{model_folder}")

        return 0

    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Install sentence-transformers: pip install sentence-transformers", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
