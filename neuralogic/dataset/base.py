from typing import Any


class BaseDataset:
    """
    Base class for logic datasets.
    """
    def dump(
        self,
        queries_fp: Any,
        examples_fp: Any,
        sep: str = "\n",
    ) -> None:
        """
        Dumps the dataset queries and examples into the provided file-like objects.

        Parameters
        ----------
        queries_fp : Any
            The file-like object to dump queries into.
        examples_fp : Any
            The file-like object to dump examples into.
        sep : str, optional
            The separator to use between samples. Default: "\\n".
        """
        raise NotImplementedError


class ConvertibleDataset(BaseDataset):
    """
    Base class for datasets that can be converted into a standard dataset format.
    """
    def to_dataset(self) -> Any:
        """
        Converts the dataset to a standard dataset format.

        Returns
        -------
        Any
            The converted dataset.
        """
        raise NotImplementedError
