"""Helper functions for chorus detection in pychorus."""

from __future__ import annotations

from typing import Any, Optional, Union

import librosa
import numpy as np
import scipy.signal

from pychorus.constants import (
    LINE_THRESHOLD,
    MIN_LINES,
    N_FFT,
    NUM_ITERATIONS,
    OVERLAP_PERCENT_MARGIN,
    SMOOTHING_SIZE_SEC,
)
from pychorus.similarity_matrix import Line, TimeLagSimilarityMatrix, TimeTimeSimilarityMatrix


def local_maxima_rows(denoised_time_lag: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Find rows whose normalized sum is a local maxima.

    Args:
        denoised_time_lag (np.ndarray[Any, Any]): Denoised time-lag similarity matrix.

    Returns:
        np.ndarray[Any, Any]: Indices of local maxima rows.

    """
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]  # type: ignore[no-any-return]


def detect_lines(
    denoised_time_lag: np.ndarray[Any, Any], rows: np.ndarray[Any, Any], min_length_samples: int
) -> list[Line]:
    """Detect lines in the time lag matrix. Reduce the threshold until enough lines are found.

    Args:
        denoised_time_lag (np.ndarray[Any, Any]): Denoised time-lag similarity matrix.
        rows (np.ndarray[Any, Any]): Candidate row indices.
        min_length_samples (int): Minimum length of a line in samples.

    Returns:
        list[Line]: List of detected lines.

    """
    cur_threshold = LINE_THRESHOLD
    line_segments: list[Line] = []
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows, cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95

    return line_segments


def detect_lines_helper(
    denoised_time_lag: np.ndarray[Any, Any], rows: np.ndarray[Any, Any], threshold: float, min_length_samples: int
) -> list[Line]:
    """Detect lines where at least min_length_samples are above threshold.

    Args:
        denoised_time_lag (np.ndarray[Any, Any]): Denoised time-lag similarity matrix.
        rows (np.ndarray[Any, Any]): Candidate row indices.
        threshold (float): Threshold for line detection.
        min_length_samples (int): Minimum length of a line in samples.

    Returns:
        list[Line]: List of detected lines.

    """
    num_samples = denoised_time_lag.shape[0]
    line_segments: list[Line] = []
    cur_segment_start: Optional[int] = None
    for row in rows:
        if int(row) < min_length_samples:
            continue
        for col in range(int(row), num_samples):
            if denoised_time_lag[int(row), col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if (cur_segment_start is not None) and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, int(row)))
                cur_segment_start = None
    return line_segments


def count_overlapping_lines(lines: list[Line], margin: int, min_length_samples: int) -> dict[Line, int]:
    """Look at all pairs of lines and see which ones overlap vertically and diagonally.

    Args:
        lines (list[Line]): List of detected lines.
        margin (int): Margin for overlap detection.
        min_length_samples (int): Minimum length of a line in samples.

    Returns:
        dict[Line, int]: Dictionary mapping lines to their overlap scores.

    """
    line_scores: dict[Line, int] = dict.fromkeys(lines, 0)

    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                (line_2.start < (line_1.start + margin))
                and (line_2.end > (line_1.end - margin))
                and (abs(line_2.lag - line_1.lag) > min_length_samples)
            )

            lines_overlap_diagonally = (
                ((line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin))
                and ((line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin))
                and (abs(line_2.lag - line_1.lag) > min_length_samples)
            )

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1

    return line_scores


def best_segment(line_scores: dict[Line, int]) -> Line:
    """Return the best line, sorted first by chorus matches, then by duration.

    Args:
        line_scores (dict[Line, int]): Dictionary mapping lines to their overlap scores.

    Returns:
        Line: The best line segment.

    """
    lines_to_sort = [(line, score, line.end - line.start) for line, score in line_scores.items()]
    lines_to_sort.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return lines_to_sort[0][0]


def draw_lines(num_samples: int, sample_rate: int, lines: list[Line]) -> None:
    """Debugging function to draw detected lines in black.

    Args:
        num_samples (int): Number of samples in the song.
        sample_rate (int): Sample rate of the song.
        lines (list[Line]): List of detected lines.

    """
    lines_matrix = np.zeros((num_samples, num_samples))
    for line in lines:
        lines_matrix[line.lag : line.lag + 4, line.start : line.end + 1] = 1

    # Import here since this function is only for debugging
    import librosa.display  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    librosa.display.specshow(lines_matrix, y_axis="time", x_axis="time", sr=sample_rate / (N_FFT / 2048))
    plt.colorbar()
    plt.set_cmap("hot_r")
    plt.show()


def create_chroma(
    input_file: str, n_fft: int = N_FFT
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], Union[int, float], float]:
    """Generate the notes present in a song.

    Args:
        input_file (str): Path to the input audio file.
        n_fft (int, optional): FFT window size. Defaults to N_FFT.

    Returns:
        tuple: (chroma, song wav data, sample rate, song length in seconds)
            chroma (np.ndarray[Any, Any]): 12 x n chroma matrix.
            song wav data (np.ndarray[Any, Any]): Audio waveform data.
            sample rate (int): Sample rate of the audio.
            song length in seconds (float): Duration of the song in seconds.

    """
    y, sr = librosa.load(input_file)
    song_length_sec = y.shape[0] / float(sr)
    s = np.abs(librosa.stft(y, n_fft=n_fft)) ** 2
    chroma = librosa.feature.chroma_stft(S=s, sr=sr)

    return chroma, y, sr, song_length_sec


def find_chorus(input_file: str, clip_length: float) -> Optional[float]:
    """Find the most repeated chorus.

    Args:
        input_file (str): Path to the input audio file.
        clip_length (float): Minimum length in seconds for the chorus (at least 10-15s).

    Returns:
        Optional[float]: Time in seconds of the start of the best chorus, or None if not found.

    """
    chroma, song_wav_data, sr, song_length_sec = create_chroma(input_file)
    num_samples = chroma.shape[1]

    time_time_similarity = TimeTimeSimilarityMatrix(chroma, sr)
    time_lag_similarity = TimeLagSimilarityMatrix(chroma, sr)

    # Denoise the time lag matrix
    chroma_sr = num_samples / song_length_sec
    smoothing_size_samples = int(SMOOTHING_SIZE_SEC * chroma_sr)
    time_lag_similarity.denoise(time_time_similarity.matrix, smoothing_size_samples)

    # Detect lines in the image
    clip_length_samples = int(clip_length * chroma_sr)
    candidate_rows = local_maxima_rows(time_lag_similarity.matrix)
    lines = detect_lines(time_lag_similarity.matrix, candidate_rows, clip_length_samples)
    if not lines:
        return None
    line_scores = count_overlapping_lines(lines, int(OVERLAP_PERCENT_MARGIN * clip_length_samples), clip_length_samples)
    best_chorus = best_segment(line_scores)
    return best_chorus.start / chroma_sr
