"""Main entry point for the pychorus command-line tool."""

import argparse

from pychorus.helpers import find_chorus


def sec_to_iso(seconds: float) -> str:
    """Convert seconds to an ISO-like time string (MM:SS.mmm).

    Args:
        seconds (float): Time in seconds.

    Returns:
        str: Time formatted as MM:SS.mmm (minutes, seconds, milliseconds).

    """
    total_ms = round(seconds * 1000)
    minutes = total_ms // 60000
    remainder = total_ms % 60000
    seconds = remainder // 1000
    milliseconds = remainder % 1000
    return f"{str(minutes).zfill(2)}:{str(seconds).zfill(2)}.{str(milliseconds).zfill(3)}"


def main(args: argparse.Namespace) -> None:
    """Run chorus detection using parsed command-line arguments."""
    result = find_chorus(args.input_file, args.min_clip_length)
    print("Chorus found at:", sec_to_iso(result) if result is not None else "No chorus found")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the chorus of a piece of music")
    parser.add_argument("input_file", type=str, help="Path to input audio file")
    parser.add_argument(
        "--min_clip_length", default=15, type=float, help="Minimum length (in seconds) to be considered a chorus"
    )

    main(parser.parse_args())
