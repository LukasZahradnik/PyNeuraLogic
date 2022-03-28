from typing import Dict, Optional, Union

from neuralogic.core.settings import Settings
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core import Template, Backend, BuiltDataset, SettingsProxy, Dataset

from neuralogic.utils.visualize import draw_model


class AbstractNeuraLogic:
    def __init__(self, backend: Backend, dataset_builder: DatasetBuilder, settings: SettingsProxy):
        self.need_sync = True

        self.template = dataset_builder.parsed_template
        self.dataset_builder = dataset_builder

        self.backend = backend
        self.settings = settings

        self.hooks_set = False
        self.hooks = {}

    def __call__(self, sample):
        raise NotImplementedError

    def build_dataset(self, dataset: Union[Dataset, BuiltDataset], file_mode: bool = False):
        return self.dataset_builder.build_dataset(dataset, self.backend, self.settings, file_mode)

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
            if not weight.isLearnable:
                continue
            weight_value = weight.value

            index = weight.index
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

    def draw(
        self,
        filename: Optional[str] = None,
        draw_ipython=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_model(self, filename, draw_ipython, img_type, value_detail, graphviz_path, *args, **kwargs)


class AbstractEvaluator:
    def __init__(self, backend: Backend, template: Template, settings: Settings):
        self.settings = settings.create_proxy()
        self.backend = backend
        self.dataset: Optional[BuiltDataset] = None

        self.neuralogic_model = template.build(backend, settings)

        if backend != Backend.PYG:
            self.neuralogic_model.set_hooks(template.hooks)

    def set_dataset(self, dataset: Union[Dataset, BuiltDataset]):
        self.dataset = self.build_dataset(dataset)

    def build_dataset(self, dataset: Union[Dataset, BuiltDataset], file_mode: bool = False):
        if isinstance(dataset, Dataset):
            return self.neuralogic_model.build_dataset(dataset, file_mode)
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

    def draw(
        self,
        filename: Optional[str] = None,
        draw_ipython=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return self.neuralogic_model.draw(
            filename, draw_ipython, img_type, value_detail, graphviz_path, *args, **kwargs
        )
