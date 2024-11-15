from typing import Dict, Optional, Union, Callable, List


from neuralogic.core.settings import Settings
from neuralogic.core.builder import DatasetBuilder
from neuralogic.core import Template, BuiltDataset, SettingsProxy, GroundedDataset
from neuralogic.dataset.base import BaseDataset

from neuralogic.utils.visualize import draw_model

import compute_graph_vectorize.engines.torch as torch_engine
from compute_graph_vectorize.engines.torch.settings import TorchModuleSettings
from compute_graph_vectorize.sources.builders import from_neuralogic
from compute_graph_vectorize.sources.neuralogic_settings import NeuralogicSettings
from compute_graph_vectorize.vectorize.pipeline.pipeline import create_vectorized_network_compiler
from compute_graph_vectorize.vectorize.settings import VectorizeSettings


class AbstractNeuraLogic:
    def __init__(self, dataset_builder: DatasetBuilder, template: Template, settings: SettingsProxy):
        self.need_sync = True

        self.source_template = [rule for rule in template.template]
        self.template = dataset_builder.parsed_template
        self.dataset_builder = dataset_builder

        self.settings = settings

        self.hooks_set = False
        self.hooks: Dict[str, List[Callable]] = {}

    def __call__(self, sample):
        raise NotImplementedError

    def ground(
        self,
        dataset: BaseDataset,
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
    ) -> GroundedDataset:
        return self.dataset_builder.ground_dataset(
            dataset,
            self.settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
        )

    def vectorize(self, dataset: BaseDataset):
        settings = NeuralogicSettings(chain_pruning=False, iso_value_compression=False)
        built_dataset = self.dataset_builder.build_dataset(dataset, settings.create_proxy())
        network = from_neuralogic(built_dataset.samples, settings)

        vectorizer_settings = VectorizeSettings()
        compiler = create_vectorized_network_compiler(
            vectorizer_settings,
            forward_pass_runner=torch_engine.torch_simple_forward_pass_runner,
        )

        vectorized_network = compiler(network)
        t_settings = TorchModuleSettings()
        torch_model = torch_engine.build_torch_model(vectorized_network, t_settings, debug=False)

        return torch_model

    def build_dataset(
        self,
        dataset: Union[BaseDataset, GroundedDataset],
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ) -> BuiltDataset:
        return self.dataset_builder.build_dataset(
            dataset,
            self.settings,
            batch_size=batch_size,
            learnable_facts=learnable_facts,
            progress=progress,
        )

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

            cols = len(value[0])

            for i, values in enumerate(value):
                for j, val in enumerate(values):
                    weight_value.set(i * cols + j, float(val))

    def parameters(self) -> Dict:
        return self.state_dict()

    def state_dict(self) -> Dict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict):
        raise NotImplementedError

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return draw_model(self, filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)


class AbstractEvaluator:
    def __init__(self, template: Template, settings: Settings):
        self.settings = settings.create_proxy()

        self.neuralogic_model = template.build(settings)
        self.neuralogic_model.set_hooks(template.hooks)

    def build_dataset(
        self,
        dataset: Union[BaseDataset, BuiltDataset],
        *,
        batch_size: int = 1,
        learnable_facts: bool = False,
        progress: bool = False,
    ):
        if isinstance(dataset, BaseDataset):
            return self.neuralogic_model.build_dataset(
                dataset,
                batch_size=batch_size,
                learnable_facts=learnable_facts,
                progress=progress,
            )
        return dataset

    @property
    def model(self) -> AbstractNeuraLogic:
        return self.neuralogic_model

    def train(self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True):
        pass

    def test(self, dataset: Optional[Union[BaseDataset, BuiltDataset]] = None, *, generator: bool = True):
        pass

    def parameters(self) -> Dict:
        return self.state_dict()

    def state_dict(self) -> Dict:
        pass

    def load_state_dict(self, state_dict: Dict):
        pass

    def reset_parameters(self):
        self.neuralogic_model.reset_parameters()

    def draw(
        self,
        filename: Optional[str] = None,
        show=True,
        img_type="png",
        value_detail: int = 0,
        graphviz_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        return self.neuralogic_model.draw(filename, show, img_type, value_detail, graphviz_path, *args, **kwargs)
