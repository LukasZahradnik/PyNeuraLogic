from typing import Dict, Optional, Union
from py4j.java_gateway import get_field

from neuralogic.core.settings import Settings
from neuralogic.core import Template, Backend, BuiltDataset
from neuralogic.utils.data import Dataset


class AbstractNeuraLogic:
    def __init__(self, template):
        self.need_sync = True
        self.template = template

        self.hooks_set = False
        self.hooks = {}

    def __call__(self, sample):
        raise NotImplementedError

    def set_hooks(self, hooks):
        self.hooks_set = len(hooks) != 0
        self.hooks = hooks

    def run_hook(self, hook: str, value):
        for callback in self.hooks[hook]:
            callback(value)

    def sync_template(self, state_dict: Optional[Dict] = None, weights=None):
        state_dict = self.state_dict() if state_dict is None else state_dict
        weights = self.template.getAllWeights() if weights is None else weights
        weight_dict = state_dict["weights"]

        for weight in weights:
            if not get_field(weight, "isLearnable"):
                continue
            weight_value = get_field(weight, "value")

            index = get_field(weight, "index")
            value = weight_dict[index]

            if isinstance(value, (float, int)):
                weight_value.set(0, float(value))
                continue

            if isinstance(value[0], (float, int)):
                for i, val in enumerate(value):
                    weight_value.set(i, float(val))
                continue

            rows = len(value)

            for i, values in enumerate(value):
                for j, val in enumerate(values):
                    weight_value.set(i * rows + j, float(val))

    def state_dict(self) -> Dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict):
        raise NotImplementedError


class AbstractEvaluator:
    def __init__(self, backend: Backend, template: Template, settings: Settings):
        self.settings = settings
        self.template = template
        self.backend = backend
        self.dataset: Optional[BuiltDataset] = None

        self.neuralogic_model = template.build(backend, self.settings)

        if backend != Backend.PYG:
            self.neuralogic_model.set_hooks(template.hooks)

    def set_dataset(self, dataset: Union[Dataset, BuiltDataset]):
        self.dataset = self.build_dataset(dataset)

    def build_dataset(self, dataset: Union[Dataset, BuiltDataset]):
        if isinstance(dataset, Dataset):
            dataset = self.template.build_dataset(dataset, self.backend, self.settings)
        return dataset

    @property
    def model(self) -> AbstractNeuraLogic:
        return self.neuralogic_model

    def train(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        pass

    def test(self, dataset: Optional[Union[Dataset, BuiltDataset]] = None, *, generator: bool = True):
        pass

    def state_dict(self) -> Dict:
        pass

    def load_state_dict(self, state_dict: Dict):
        pass

    def reset_parameters(self):
        self.neuralogic_model.reset_parameters()
