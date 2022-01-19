import dataclasses


@dataclasses.dataclass
class Task:
    collection: str = "TUDataset"
    output_size: int = 1
    activation: str = "sigmoid"
    loss: str = "crossentropy"
    task: str = "classification"
