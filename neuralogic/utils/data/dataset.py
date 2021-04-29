class Dataset:
    def __init__(self, samples):
        self.__samples = samples
        self.__len = len(samples)

    def __len__(self):
        return self.__len

    def __getitem__(self, item):
        return self.__samples[item]

    @property
    def samples(self):
        return self.__samples
