import logging

# Package Metadata
__version__ = "1.0.0"
__author__ = "AI Proctor Team"

# Configure package-level logging
logger = logging.getLogger(__name__)

# Guard against missing key runtime dependencies with friendly error messages
try:
    from .head_pose import HeadPoseEstimator
    from .proctor_ai import (
        InterviewProctor,
        analyze_frame,
        get_proctor_instance,
    )
except ImportError as e:
    logger.error("Failed to import core proctor modules. Check dependencies.")
    raise ImportError(
        f"Error loading AI Proctor utilities: {e}\n"
        "Ensure OpenCV, MediaPipe, Ultralytics, and NumPy are installed."
    ) from e

# Explicit export definition (alphabetically organized)
__all__ = [
    "HeadPoseEstimator",
    "InterviewProctor",
    "analyze_frame",
    "get_proctor_instance",
    "__version__",
]
