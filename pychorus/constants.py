"""Constants for pychorus chorus detection algorithms."""

# Denoising size in seconds
SMOOTHING_SIZE_SEC: float = 2.5

# Number of samples to consider in one chunk.
# Smaller values take more time, but are more accurate
N_FFT: int = 2**14

# For line detection
LINE_THRESHOLD: float = 0.15
MIN_LINES: int = 8
NUM_ITERATIONS: int = 8

# We allow an error proportional to the length of the clip
OVERLAP_PERCENT_MARGIN: float = 0.2
