from typing import Optional

from neuralogic import get_neuralogic
from neuralogic.core.settings import Settings

from py4j.java_gateway import set_field


def get_drawing_settings(img_type="png"):
    settings = Settings()

    set_field(settings.settings, "drawing", False)
    set_field(settings.settings, "storeNotShow", True)
    set_field(settings.settings, "imgType", img_type)

    return settings


def get_template_drawer(settings: Settings):
    namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.debugging.drawing

    return namespace.TemplateDrawer(settings.settings)


def get_sample_drawer(settings: Settings):
    namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.debugging.drawing

    return namespace.NeuralNetDrawer(settings.settings)

#todo gusta: + groundingDrawer, pipelineDrawer...

def draw(drawer, obj, filename: Optional[str] = None, draw_ipython=True, *args, **kwargs):
    if filename is not None:
        drawer.drawIntoFile(obj, filename)

        return

    data = drawer.drawIntoBytes(obj)

    if draw_ipython:
        from IPython.display import Image

        return Image(data, *args, **kwargs)
    return data


def to_dot_source(drawer, obj) -> str:
    return drawer.getGraphSource(obj)


def draw_model(model, filename: Optional[str] = None, draw_ipython=True, img_type="png", *args, **kwargs):
    if model.need_sync:
        model.sync_template()

    template = model.template
    template_drawer = get_template_drawer(get_drawing_settings(img_type=img_type))

    return draw(template_drawer, template, filename, draw_ipython, *args, **kwargs)


def draw_sample(sample, filename: Optional[str] = None, draw_ipython=True, img_type="png", *args, **kwargs):
    sample_drawer = get_sample_drawer(get_drawing_settings(img_type=img_type))

    return draw(sample_drawer, sample, filename, draw_ipython, *args, **kwargs)


def model_to_dot_source(model) -> str:
    if model.need_sync:
        model.sync_template()

    template = model.template
    template_drawer = get_template_drawer(get_drawing_settings())

    return to_dot_source(template_drawer, template)


def sample_to_dot_source(sample) -> str:
    sample_drawer = get_sample_drawer(get_drawing_settings())

    return to_dot_source(sample_drawer, sample)
