from collections.abc import Iterator, Sequence

import numpy as np


class FrameSequence:
    """Index a sequence of frame paths without retaining decoded images."""

    def __init__(self, paths: Sequence[str]) -> None:
        """Record the source paths in their original order."""
        self._paths = tuple(paths)

    def __len__(self) -> int:
        """Return the number of source frames."""
        return len(self._paths)

    def __getitem__(self, index: int) -> np.ndarray:
        """Decode the requested frame at its original size."""
        import cv2

        path = self._paths[index]
        frame = cv2.imread(path)
        if frame is None:
            raise ValueError(f"Unable to read source frame: {path}")
        return frame

    def __iter__(self) -> Iterator[np.ndarray]:
        """Decode each source frame in order as it is requested."""
        for index in range(len(self)):
            yield self[index]
