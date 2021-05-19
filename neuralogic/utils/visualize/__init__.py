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


def draw_model(model, filename: Optional[str] = None, draw_ipython=True, img_type="png", *args, **kwargs):
    if model.need_sync:
        model.sync_template()

    template = model.template
    template_drawer = get_template_drawer(get_drawing_settings(img_type=img_type))

    if filename is not None:
        template_drawer.drawIntoFile(template, filename)

        return

    data = template_drawer.drawIntoBytes(template)

    if draw_ipython:
        from IPython.display import Image

        return Image(data, *args, **kwargs)
    return data


def model_to_dot_source(model) -> str:
    if model.need_sync:
        model.sync_template()

    template = model.template
    template_drawer = get_template_drawer(get_drawing_settings())

    return template_drawer.getGraphSource(template)
