from typing import Optional

from neuralogic import get_neuralogic
from neuralogic.core import Problem
from neuralogic.core.settings import Settings

from py4j.java_gateway import set_field


def get_template_drawer():
    namespace = get_neuralogic().cz.cvut.fel.ida.pipelines.debugging.drawing

    settings = Settings()

    set_field(settings.settings, "drawing", False)
    set_field(settings.settings, "storeNotShow", True)

    return namespace.TemplateDrawer(settings.settings)


def draw_problem(problem: Problem, filename: Optional[str] = None, draw_ipython=True, *args, **kwargs):
    template = problem.get_parsed_template()
    template_drawer = get_template_drawer()

    data = template_drawer.drawForPython(template, filename)

    if filename is None and draw_ipython:
        from IPython.display import Image

        return Image(data, *args, **kwargs)
    return data
