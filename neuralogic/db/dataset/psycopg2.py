from neuralogic.dataset.base import BaseDataset


class Psycopg2Dataset(BaseDataset):
    def __init__(self, cursor, mapping):
        self.cursor = cursor
        self.mapping = mapping

    def dump(
        self,
        queries_fp,
        examples_fp,
        sep: str = "\n",
    ):
        pass