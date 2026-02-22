from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator


class CosmosInferenceDataloader(ABC):
    """Abstract base for dataloaders consumed by Cosmos inference runners.

    Subclasses must implement ``__iter__`` which returns a generator
    yielding ``(video_path, label, new_images)`` tuples.
    """

    @abstractmethod
    def __iter__(self) -> Generator[tuple[Path, bool, list], None, None]:
        ...
