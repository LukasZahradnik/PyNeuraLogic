from typing import Optional
from shutil import copyfileobj

from neuralogic.dataset.base import BaseDataset


class FileDataset(BaseDataset):
    def __init__(
        self,
        examples_file: Optional[str] = None,
        queries_file: Optional[str] = None,
    ):
        self.examples_file = examples_file
        self.queries_file = queries_file

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
