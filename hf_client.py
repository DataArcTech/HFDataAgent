import requests
from huggingface_hub import HfApi, hf_hub_download
from typing import Optional, List, Dict, Any

class HFClient:
    """
    A simple wrapper client for interacting with Hugging Face datasets and metadata.
    """

    BASE_URL = "https://datasets-server.huggingface.co"

    def __init__(self):
        self.api = HfApi()

    # -------------------------------
    # Dataset Search & Metadata
    # -------------------------------
    def search_datasets(self, query: str, limit: int = 5) -> List[Any]:
        """
        Search for datasets on Hugging Face Hub by query keyword.
        Results are filtered to JSON datasets and sorted by downloads.
        """
        datasets = list(
            self.api.list_datasets(
                search=query,
                filter="format:json",
                sort="downloads"
            )
        )[:limit]
        return datasets

    def get_readme(self, repo_id: str, repo_type: str = "dataset") -> str:
        """
        Download and return the README.md content of a dataset repository.
        """
        readme_path = hf_hub_download(
            repo_id=repo_id,
            repo_type=repo_type,
            filename="README.md"
        )
        with open(readme_path, encoding="utf-8") as f:
            return f.read()

    # -------------------------------
    # Dataset Server Endpoints
    # -------------------------------
    def get_splits(self, dataset: str, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve the available splits (train/test/validation) of a dataset.
        """
        url = f"{self.BASE_URL}/splits"
        params = {"dataset": dataset}
        if config:
            params["config"] = config

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_first_rows(
        self,
        dataset: str,
        split: str = "train",
        config: str = "default"
    ) -> Dict[str, Any]:
        """
        Get the first few rows of a dataset split.
        """
        url = f"{self.BASE_URL}/first-rows"
        params = {"dataset": dataset, "split": split, "config": config}

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def get_info(self, dataset: str, config: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve dataset information and metadata.
        """
        url = f"{self.BASE_URL}/info"
        params = {"dataset": dataset}
        if config:
            params["config"] = config

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

