class BaseDataset:
    def dump(
        self,
        queries_fp,
        examples_fp,
        sep: str = "\n",
    ):
        raise NotImplementedError

    def dump_to_file(
        self,
        queries_filename: str,
        examples_filename: str,
        sep: str = "\n",
    ):
        with open(queries_filename, "w") as queries_fp, open(examples_filename, "w") as examples_fp:
            self.dump(queries_fp, examples_fp, sep)


class ConvertableDataset(BaseDataset):
    def to_dataset(self):
        raise NotImplementedError
