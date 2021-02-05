from . import get_neuralogic
from py4j.java_gateway import get_field
from .settings import Settings
from .sources import Sources


class Pipeline:
    def __init__(self, settings: Settings, sources: Sources):
        self.namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.building
        self.pipeline = self.namespace.LearningSchemeBuilder.getPipeline(settings.settings, sources.sources)

    def execute(self, sources: Sources):
        result = self.pipeline.execute(sources.sources)

        name = get_field(result, "r")
        data = get_field(result, "s")

        return name, data
