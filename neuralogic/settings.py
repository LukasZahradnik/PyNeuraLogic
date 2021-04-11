from neuralogic import get_neuralogic, get_gateway
from py4j.java_gateway import set_field, get_field


class Settings:
    def __init__(self):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.setup
        self.settings = self.namespace.Settings()

        set_field(self.settings, "debugExporting", False)
        set_field(self.settings, "isoValueCompression", False)
        set_field(self.settings, "exportBlocks", get_gateway().new_array(get_gateway().jvm.java.lang.String, 0))
        # set_field(self.settings, "chainPruning", False)

    @property
    def seed(self) -> int:
        return get_field(self.settings, "seed")

    @seed.setter
    def seed(self, seed: int):
        set_field(self.settings, "seed", seed)

    @property
    def debug_exporting(self) -> bool:
        return get_field(self.settings, "debugExporting")

    @debug_exporting.setter
    def debug_exporting(self, debug_export: bool):
        set_field(self.settings, "debugExporting", debug_export)

    @property
    def default_fact_value(self) -> float:
        return get_field(self.settings, "defaultFactValue")

    @default_fact_value.setter
    def default_fact_value(self, value: float):
        set_field(self.settings, "defaultFactValue", value)

    def to_json(self) -> str:
        return self.settings.exportToJson()
