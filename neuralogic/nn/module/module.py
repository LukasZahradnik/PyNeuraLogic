from abc import ABC, abstractmethod


class Module(ABC):
    @abstractmethod
    def __call__(self):
        ...
