import os
from typing import Optional
from shutil import copyfileobj

from neuralogic.dataset.base import BaseDataset


class FileDataset(BaseDataset):
    r"""
    ``FileDataset`` represents samples stored in files in the `NeuraLogic <https://github.com/GustikS/NeuraLogic>`_
    (logic) format.

    Parameters
    ----------

    examples_file : Optional[str]
        Path to the examples file. Default: ``None``
    queries_file : Optional[str]
        Path to the queries file. Default: ``None``

    """

    def __init__(
        self,
        examples_file: Optional[str] = None,
        queries_file: Optional[str] = None,
    ):
        self.examples_file = examples_file
        self.queries_file = queries_file

        if self.examples_file is not None:
            self.examples_file = os.path.abspath(self.examples_file)
        if self.queries_file is not None:
            self.queries_file = os.path.abspath(self.queries_file)

    def dump(
        self,
        queries_fp,
        examples_fp,
        sep: str = "\n",
    ):
        if self.examples_file is not None:
            with open(self.examples_file, "r") as fp:
                copyfileobj(fp, examples_fp)

        if self.queries_file is not None:
            with open(self.queries_file, "r") as fp:
                copyfileobj(fp, queries_fp)
