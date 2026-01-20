"""Download tiktoken encoding files for offline use.

This script downloads the tiktoken encoding files and saves them with the correct
hash-based filenames that tiktoken expects in its cache directory.

Usage:
    python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache
"""

from __future__ import annotations

import argparse
import hashlib
import ssl
import sys
import urllib.request
from pathlib import Path

# Tiktoken encoding URLs (from tiktoken_ext/openai_public.py)
ENCODINGS = {
    "cl100k_base": "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
    "p50k_base": "https://openaipublic.blob.core.windows.net/encodings/p50k_base.tiktoken",
    "p50k_edit": "https://openaipublic.blob.core.windows.net/encodings/p50k_edit.tiktoken",
    "r50k_base": "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken",
    "o200k_base": "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
}

# Default encoding used by this project
DEFAULT_ENCODING = "cl100k_base"


def compute_cache_key(url: str) -> str:
    """Compute the cache key (filename) that tiktoken uses for a URL.

    Tiktoken uses SHA1 hash of the URL as the cache filename.
    """
    return hashlib.sha1(url.encode()).hexdigest()


def download_encoding(url: str, output_path: Path, skip_ssl: bool = False) -> None:
    """Download an encoding file from the given URL."""
    print(f"Downloading from: {url}")

    if skip_ssl:
        # Create SSL context that doesn't verify certificates
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(url, context=ctx)
    else:
        response = urllib.request.urlopen(url)

    data = response.read()
    output_path.write_bytes(data)
    print(f"Saved to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download tiktoken encoding files for offline use.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download default encoding (cl100k_base)
    python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache

    # Download all encodings
    python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache --all

    # Download specific encoding
    python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache --encoding p50k_base

    # Skip SSL verification (for corporate environments with SSL issues)
    python -m scripts.download_tiktoken_cache --output-dir models/tiktoken_cache --skip-ssl
""",
    )
    parser.add_argument(
        "--output-dir",
        default="models/tiktoken_cache",
        help="Directory to save the encoding files (default: models/tiktoken_cache).",
    )
    parser.add_argument(
        "--encoding",
        choices=list(ENCODINGS.keys()),
        default=DEFAULT_ENCODING,
        help=f"Encoding to download (default: {DEFAULT_ENCODING}).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available encodings.",
    )
    parser.add_argument(
        "--skip-ssl",
        action="store_true",
        help="Skip SSL certificate verification (use in corporate environments with SSL issues).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    encodings_to_download = list(ENCODINGS.keys()) if args.all else [args.encoding]

    print(f"Output directory: {output_dir.absolute()}")
    print(f"Encodings to download: {', '.join(encodings_to_download)}")
    if args.skip_ssl:
        print("WARNING: SSL verification disabled")
    print()

    success_count = 0
    for encoding_name in encodings_to_download:
        url = ENCODINGS[encoding_name]
        cache_key = compute_cache_key(url)
        output_path = output_dir / cache_key

        print(f"Downloading {encoding_name}...")
        print(f"  URL: {url}")
        print(f"  Cache key: {cache_key}")

        try:
            download_encoding(url, output_path, skip_ssl=args.skip_ssl)
            success_count += 1
            print(f"  Success!\n")
        except Exception as e:
            print(f"  Error: {e}\n", file=sys.stderr)
            if not args.all:
                return 1

    print(f"\nDownloaded {success_count}/{len(encodings_to_download)} encoding(s).")
    print(f"\nTo use offline, set environment variable:")
    print(f"  TIKTOKEN_CACHE_DIR={output_dir.absolute()}")
    print(f"\nOr the code will auto-detect if the cache is in models/tiktoken_cache/")

    return 0 if success_count == len(encodings_to_download) else 1


if __name__ == "__main__":
    sys.exit(main())
