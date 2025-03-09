from abc import ABC, abstractmethod

class IKaggleRepository(ABC):
    @abstractmethod
    def download_dataset(self, dataset_name: str, path: str):
        pass

    @abstractmethod
    def get_dataset_metadata(self, dataset_name: str) -> dict:
        pass