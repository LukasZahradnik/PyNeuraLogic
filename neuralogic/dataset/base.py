class BaseDataset:
    def dump(
        self,
        queries_fp,
        examples_fp,
        sep: str = "\n",
    ):
        raise NotImplementedError


class ConvertibleDataset(BaseDataset):
    def to_dataset(self):
        raise NotImplementedError
