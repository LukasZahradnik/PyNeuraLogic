from . import get_neuralogic
from py4j.java_gateway import set_field


class Settings:
    def __init__(self):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        self.settings = self.namespace.Settings()
        set_field(self.settings, "isoValueCompression", False)
        # set_field(self.settings, "chainPruning", False)

    @property
    def seed(self) -> int:
        return self.settings.seed

    @seed.setter
    def seed(self, seed: int):
        self.settings.seed = seed

    @property
    def default_fact_value(self) -> float:
        return self.settings.defaultFactValue

    @default_fact_value.setter
    def default_fact_value(self, value: float):
        self.settings.defaultFactValue = value

    def to_json(self) -> str:
        return self.settings.exportToJson()
