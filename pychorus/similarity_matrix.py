"""Similarity matrix classes and utilities for pychorus."""

from abc import ABCMeta, abstractmethod
from math import sqrt
from typing import Any

import numpy as np
import scipy.signal

from pychorus.constants import N_FFT


class SimilarityMatrix(metaclass=ABCMeta):
    """Abstract class for time-time and time-lag similarity matrices.

    Args:
        chroma (np.ndarray[Any, Any]): 12 x n numpy array of musical notes present at every time step.
        sample_rate (float): Sample rate of the audio, almost always 22050.

    """

    def __init__(self, chroma: np.ndarray[Any, Any], sample_rate: float) -> None:
        """Initialize the similarity matrix.

        Args:
            chroma (np.ndarray[Any, Any]): 12 x n numpy array of musical notes present at every time step.
            sample_rate (float): Sample rate of the audio, almost always 22050.

        """
        self.sample_rate = sample_rate
        self.matrix = self.compute_similarity_matrix(chroma)

    @abstractmethod
    def compute_similarity_matrix(self, chroma: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute the specific type of similarity matrix.

        Args:
            chroma (np.ndarray[Any, Any]): 12 x n numpy array of musical notes present at every time step.

        Returns:
            np.ndarray[Any, Any]: The computed similarity matrix.

        """

    def display(self) -> None:
        """Display the similarity matrix using matplotlib and librosa."""
        import librosa.display  # noqa: PLC0415
        import matplotlib.pyplot as plt  # noqa: PLC0415

        librosa.display.specshow(self.matrix, y_axis="time", x_axis="time", sr=self.sample_rate / (N_FFT / 2048))
        plt.colorbar()
        plt.set_cmap("hot_r")
        plt.show()


class TimeTimeSimilarityMatrix(SimilarityMatrix):
    """Class for the time-time similarity matrix where sample (x, y) represents how similar
    the song frames x and y are.

    """

    def compute_similarity_matrix(self, chroma: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute the time-time similarity matrix using numpy broadcasting.

        Args:
            chroma (np.ndarray[Any, Any]): 12 x n numpy array of musical notes present at every time step.

        Returns:
            np.ndarray[Any, Any]: The computed time-time similarity matrix.

        """
        broadcast_x = np.expand_dims(chroma, 2)  # (12 x n x 1)
        broadcast_y = np.swapaxes(np.expand_dims(chroma, 2), 1, 2)  # (12 x 1 x n)
        return 1 - (np.linalg.norm((broadcast_x - broadcast_y), axis=0) / sqrt(12))  # type: ignore[no-any-return]


class TimeLagSimilarityMatrix(SimilarityMatrix, metaclass=ABCMeta):
    """Class to hold the time-lag similarity matrix where sample (x, y) represents
    the similarity of song frames x and (x-y).
    """

    def compute_similarity_matrix(self, chroma: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute the time-lag similarity matrix.

        Args:
            chroma (np.ndarray[Any, Any]): 12 x n numpy array of musical notes present at every time step.

        Returns:
            np.ndarray[Any, Any]: The computed time-lag similarity matrix.

        """
        num_samples = chroma.shape[1]
        broadcast_x = np.repeat(np.expand_dims(chroma, 2), num_samples + 1, axis=2)

        # We create the lag effect by tiling the samples but reshaping with an extra column
        # so that subsequent rows are offset by one each time
        circulant_y = np.tile(chroma, (1, num_samples + 1)).reshape(12, num_samples, num_samples + 1)
        time_lag_similarity = 1 - (np.linalg.norm((broadcast_x - circulant_y), axis=0) / sqrt(12))
        time_lag_similarity = np.rot90(time_lag_similarity, k=1, axes=(0, 1))
        return time_lag_similarity[:num_samples, :num_samples]

    def denoise(self, time_time_matrix: np.ndarray[Any, Any], smoothing_size: int) -> None:
        """Emphasize horizontal lines by suppressing vertical and diagonal lines.

        Looks at 6 moving averages (left, right, up, down, upper diagonal, lower diagonal).
        For lines, the left or right average should be much greater than the other ones.

        Args:
            time_time_matrix (np.ndarray[Any, Any]): n x n numpy array to quickly compute diagonal averages.
            smoothing_size (int): Smoothing size in samples (usually 1-2 sec is good).

        """
        n = self.matrix.shape[0]

        # Get the horizontal strength at every sample
        horizontal_smoothing_window = np.ones((1, smoothing_size)) / smoothing_size
        horizontal_moving_average = scipy.signal.convolve2d(self.matrix, horizontal_smoothing_window, mode="full")
        left_average = horizontal_moving_average[:, 0:n]
        right_average = horizontal_moving_average[:, smoothing_size - 1 :]
        max_horizontal_average = np.maximum(left_average, right_average)

        # Get the vertical strength at every sample
        vertical_smoothing_window = np.ones((smoothing_size, 1)) / smoothing_size
        vertical_moving_average = scipy.signal.convolve2d(self.matrix, vertical_smoothing_window, mode="full")
        down_average = vertical_moving_average[0:n, :]
        up_average = vertical_moving_average[smoothing_size - 1 :, :]

        # Get the diagonal strength of every sample from the time_time_matrix.
        # The key insight is that diagonal averages in the time lag matrix are horizontal
        # lines in the time time matrix
        diagonal_moving_average = scipy.signal.convolve2d(time_time_matrix, horizontal_smoothing_window, mode="full")
        ur_average = np.zeros((n, n))
        ll_average = np.zeros((n, n))
        for x in range(n):
            for y in range(x):
                ll_average[y, x] = diagonal_moving_average[x - y, x]
                ur_average[y, x] = diagonal_moving_average[x - y, x + smoothing_size - 1]

        non_horizontal_max = np.maximum.reduce([down_average, up_average, ll_average, ur_average])
        non_horizontal_min = np.minimum.reduce([up_average, down_average, ll_average, ur_average])

        # If the horizontal score is stronger than the vertical score, it is considered part of a line
        # and we only subtract the minimum average. Otherwise subtract the maximum average
        suppression = (max_horizontal_average > non_horizontal_max) * non_horizontal_min + (
            max_horizontal_average <= non_horizontal_max
        ) * non_horizontal_max

        # Filter it horizontally to remove any holes, and ignore values less than 0
        denoised_matrix = scipy.ndimage.filters.gaussian_filter1d(
            np.triu(self.matrix - suppression), smoothing_size, axis=1
        )
        denoised_matrix = np.maximum(denoised_matrix, 0)
        denoised_matrix[0:5, :] = 0

        self.matrix = denoised_matrix


class Line:
    """Represents a detected line segment in the similarity matrix.

    Args:
        start (int): Start index of the line.
        end (int): End index of the line.
        lag (int): Lag value of the line.

    """

    def __init__(self, start: int, end: int, lag: int) -> None:
        """Initialize the line segment in the similarity matrix.

        Args:
            start (int): Start index of the line.
            end (int): End index of the line.
            lag (int): Lag value of the line.

        """
        self.start = start
        self.end = end
        self.lag = lag

    def __repr__(self) -> str:
        """Return a string representation of the Line object.

        Returns:
            str: String representation.

        """
        return f"Line ({self.start} {self.end} {self.lag})"
