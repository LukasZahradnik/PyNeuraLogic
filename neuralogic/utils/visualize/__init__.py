from __future__ import annotations

import io
import os
from typing import Any

import jpype

from neuralogic.setup import get_default_graphviz_path
from neuralogic.utils.visualize.drawing_settings import (
    DrawingSettings,
)
from neuralogic.utils.visualize.drawing_settings import (
    EdgeAttributeKey as EdgeAttributeKey,
)
from neuralogic.utils.visualize.drawing_settings import (
    EdgeAttributes as EdgeAttributes,
)
from neuralogic.utils.visualize.drawing_settings import (
    EdgeStyleKey as EdgeStyleKey,
)
from neuralogic.utils.visualize.drawing_settings import (
    EdgeStyles as EdgeStyles,
)
from neuralogic.utils.visualize.drawing_settings import (
    GraphAttributeKey as GraphAttributeKey,
)
from neuralogic.utils.visualize.drawing_settings import (
    GraphAttributes as GraphAttributes,
)
from neuralogic.utils.visualize.drawing_settings import (
    NodeAttributeKey as NodeAttributeKey,
)
from neuralogic.utils.visualize.drawing_settings import (
    NodeAttributes as NodeAttributes,
)
from neuralogic.utils.visualize.drawing_settings import (
    NodeStyleKey as NodeStyleKey,
)
from neuralogic.utils.visualize.drawing_settings import (
    NodeStyles as NodeStyles,
)


def get_graphviz_path(path: str | None = None) -> str | None:
    """
    Get the path to the Graphviz executable
    """
    if path is not None:
        return path
    return get_default_graphviz_path()


def get_drawing_settings(
    img_type: str = "png", value_detail: int = 0, graphviz_path: str | None = None
) -> DrawingSettings:
    """Returns standalone drawing settings configured with the specified image type.

    Parameters
    ----------
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values (0, 1 or 2). Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.

    Returns
    -------
    DrawingSettings
        Standalone drawing settings. Style and value detail can be further
        configured on the returned object before it is passed to a drawer.
    """
    settings = DrawingSettings(value_detail=value_detail)
    settings._set_img_type(img_type)
    settings._set_graphviz_path(graphviz_path)
    return settings


def _resolve_drawing_settings(
    settings: DrawingSettings | None,
    img_type: str,
    value_detail: int,
    graphviz_path: str | None,
) -> DrawingSettings:
    if settings is None:
        return get_drawing_settings(img_type, value_detail, graphviz_path)

    settings._set_img_type(img_type)
    settings._set_graphviz_path(graphviz_path)
    return settings


def get_model_drawer(settings: DrawingSettings) -> Any:
    """Returns the model drawer.

    Parameters
    ----------
    settings : DrawingSettings
        The drawing settings.

    Returns
    -------
    Any
        The model drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.TemplateDrawer")(settings.settings)


def get_sample_drawer(settings: DrawingSettings) -> Any:
    """Returns the sample drawer.

    Parameters
    ----------
    settings : DrawingSettings
        The drawing settings.

    Returns
    -------
    Any
        The sample drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.NeuralNetDrawer")(settings.settings)


def get_grounding_drawer(settings: DrawingSettings) -> Any:
    """Returns the grounding drawer.

    Parameters
    ----------
    settings : DrawingSettings
        The drawing settings.

    Returns
    -------
    Any
        The grounding drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.GroundingDrawer")(settings.settings)


def get_pipeline_drawer(settings: DrawingSettings) -> Any:
    """Returns the pipeline drawer.

    Parameters
    ----------
    settings : DrawingSettings
        The drawing settings.

    Returns
    -------
    Any
        The pipeline drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debuging.drawing.PipelineDrawer")(settings.settings)


def draw(
    drawer: Any,
    obj: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draws the object using the provided drawer.

    Parameters
    ----------
    drawer : Any
        The drawer to use.
    obj : Any
        The object to draw.
    filename : str, optional
        The filename to draw into. Default: None.
    show : bool
        Whether to show the image. Default: True.
    img_type : str
        The image type. Default: "png".
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The drawing data, image object, or None if drawn into a file.
    """
    if filename is not None:
        try:
            drawer.drawIntoFile(obj, os.path.abspath(filename))
        except jpype.java.lang.NullPointerException as e:
            raise RuntimeError(
                "Drawing raised NullPointerException. Try to install GraphViz (https://graphviz.org/download/) on "
                "your Path or specify the path via the `graphviz_path` parameter"
            ) from e

        return None

    data = drawer.drawIntoBytes(obj)

    if data is None:
        raise RuntimeError(
            "Drawing failed. Try to install GraphViz (https://graphviz.org/download/) on your Path or specify the "
            "path via the `graphviz_path` parameter"
        )

    data = bytes(data)

    if show:
        if is_jupyter():
            from IPython.display import SVG, Image

            if img_type.lower() == "svg":
                return SVG(data, *args, **kwargs)
            return Image(data, *args, **kwargs)
        else:
            import matplotlib.image as mpimg
            import matplotlib.pyplot as plt

            img = mpimg.imread(io.BytesIO(data), format=img_type)
            fig = plt.figure()

            if hasattr(fig.canvas, "set_window_title"):
                fig.canvas.set_window_title(kwargs.get("title", ""))

            ax = fig.add_axes((0, 0, 1, 1))
            ax.axis("off")
            ax.imshow(img)

            plt.show()
            return
    return data


def to_dot_source(drawer: Any, obj: Any) -> str:
    return str(drawer.getGraphSource(obj))


def to_graphml_source(drawer: Any, obj: Any) -> str | None:
    """Renders the object into its GraphML source representation.

    Parameters
    ----------
    drawer : Any
        The drawer to use.
    obj : Any
        The object to render.

    Returns
    -------
    str | None
        The GraphML source, or None if the backend graphviz-based conversion failed.
    """
    graphml = drawer.getGraphMLSource(obj)

    if graphml is None:
        return None

    return str(graphml)


def graphml_to_networkx(graphml: str | None) -> Any:
    """Loads a GraphML string into a networkx graph.

    Parameters
    ----------
    graphml : str | None
        The GraphML source.

    Returns
    -------
    networkx.DiGraph
        The graph read from the GraphML source.
    """
    if graphml is None:
        raise RuntimeError(
            "GraphML conversion failed. Try to install GraphViz (https://graphviz.org/download/) and ensure it is "
            "available on your PATH"
        )

    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError(
            "networkx is required for GraphML to networkx conversion. Install it with `pip install networkx`."
        ) from e

    return nx.read_graphml(io.StringIO(graphml))


def draw_model(
    model: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
    settings: DrawingSettings | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draws model either as an image of type img_type either into:
        * a file - if filename is specified),
        * an IPython Image or Image popup - if show is True
        * or bytes otherwise

    Parameters
    ----------
    model : NeuralModule
        The model to draw.
    filename : str, optional
        The filename to draw into. Default: None.
    show : bool
        Whether to show the image. Default: True.
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values. Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.
    settings : DrawingSettings, optional
        Styling settings. When provided, they control value detail and graph
        styling, while ``filename``, ``img_type`` and ``graphviz_path`` from this
        call still control the output.
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The model drawing.
    """
    if model._need_sync:
        model._sync_model()

    model = model._parsed_model
    drawing_settings = _resolve_drawing_settings(settings, img_type, value_detail, graphviz_path)
    template_drawer = get_model_drawer(drawing_settings)

    return draw(template_drawer, model, filename, show, img_type, *args, **kwargs)


def draw_grounding(
    grounding: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
    settings: DrawingSettings | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draws sample's grounding either as an image of type img_type either into:
        * a file - if filename is specified),
        * an IPython Image or Image popup - if show is True
        * or bytes otherwise

    Parameters
    ----------
    grounding : Any
        The grounding to draw.
    filename : str, optional
        The filename to draw into. Default: None.
    show : bool
        Whether to show the image. Default: True.
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values. Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.
    settings : DrawingSettings, optional
        Styling settings. When provided, they control value detail and graph
        styling, while ``filename``, ``img_type`` and ``graphviz_path`` from this
        call still control the output.
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The grounding drawing.
    """
    drawing_settings = _resolve_drawing_settings(settings, img_type, value_detail, graphviz_path)
    grounding_drawer = get_grounding_drawer(drawing_settings)

    return draw(grounding_drawer, grounding, filename, show, img_type, *args, **kwargs)


def draw_sample(
    sample: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
    settings: DrawingSettings | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draws sample either as an image of type img_type either into:
        * a file - if filename is specified),
        * an IPython Image or Image popup - if show is True
        * or bytes otherwise

    Parameters
    ----------
    sample : Any
        The sample to draw.
    filename : str, optional
        The filename to draw into. Default: None.
    show : bool
        Whether to show the image. Default: True.
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values. Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.
    settings : DrawingSettings, optional
        Styling settings. When provided, they control value detail and graph
        styling, while ``filename``, ``img_type`` and ``graphviz_path`` from this
        call still control the output.
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The sample drawing.
    """
    draw_object = sample._java_sample

    drawing_settings = _resolve_drawing_settings(settings, img_type, value_detail, graphviz_path)
    sample_drawer = get_sample_drawer(drawing_settings)

    return draw(sample_drawer, draw_object, filename, show, img_type, *args, **kwargs)


def draw_pipeline(
    pipeline: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
    settings: DrawingSettings | None = None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Draws a backend pipeline either as an image of type img_type either into:
        * a file - if filename is specified),
        * an IPython Image or Image popup - if show is True
        * or bytes otherwise

    Parameters
    ----------
    pipeline : Any
        The backend pipeline (``cz.cvut.fel.ida.pipelines.Pipeline``) to draw.
    filename : str, optional
        The filename to draw into. Default: None.
    show : bool
        Whether to show the image. Default: True.
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values. Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.
    settings : DrawingSettings, optional
        Styling settings. When provided, they control value detail and graph
        styling, while ``filename``, ``img_type`` and ``graphviz_path`` from this
        call still control the output.
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The pipeline drawing.
    """
    drawing_settings = _resolve_drawing_settings(settings, img_type, value_detail, graphviz_path)
    pipeline_drawer = get_pipeline_drawer(drawing_settings)

    return draw(pipeline_drawer, pipeline, filename, show, img_type, *args, **kwargs)


def model_to_dot_source(model: Any, settings: DrawingSettings | None = None) -> str:
    """Renders the model into its dot source representation.

    Parameters
    ----------
    model : NeuralModule
        The model to render.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, they are used instead of the defaults.

    Returns
    -------
    str
        The dot source representation.
    """
    if model._need_sync:
        model._sync_model()

    model = model._parsed_model
    template_drawer = get_model_drawer(settings or get_drawing_settings())

    return to_dot_source(template_drawer, model)


def sample_to_dot_source(
    sample: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> str:
    """Renders the sample into its dot source representation.

    Parameters
    ----------
    sample : Any
        The sample to render.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    str
        The dot source representation.
    """
    sample_drawer = get_sample_drawer(settings or get_drawing_settings(value_detail=value_detail))

    return to_dot_source(sample_drawer, sample._java_sample)


def model_to_graphml_source(
    model: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> str | None:
    """Renders the model into its GraphML source representation.

    Parameters
    ----------
    model : NeuralModule
        The model to render.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    str | None
        The GraphML source, or None if the backend graphviz-based conversion failed.
    """
    if model._need_sync:
        model._sync_model()

    template_drawer = get_model_drawer(settings or get_drawing_settings(value_detail=value_detail))

    return to_graphml_source(template_drawer, model._parsed_model)


def sample_to_graphml_source(
    sample: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> str | None:
    """Renders the sample into its GraphML source representation.

    Parameters
    ----------
    sample : Any
        The sample to render.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    str | None
        The GraphML source, or None if the backend graphviz-based conversion failed.
    """
    sample_drawer = get_sample_drawer(settings or get_drawing_settings(value_detail=value_detail))

    return to_graphml_source(sample_drawer, sample._java_sample)


def save_model_graphml(
    model: Any, path: str, value_detail: int = 0, settings: DrawingSettings | None = None
) -> None:
    """Saves the model graph as a GraphML file consumable by networkx.

    Parameters
    ----------
    model : NeuralModule
        The model to save.
    path : str
        The destination file path. Parent directories are created if needed.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.
    """
    if model._need_sync:
        model._sync_model()

    path = os.path.abspath(path)
    template_drawer = get_model_drawer(settings or get_drawing_settings(value_detail=value_detail))
    template_drawer.saveGraphML(model._parsed_model, path)

    if not os.path.exists(path):
        raise RuntimeError(
            "GraphML saving failed. Try to install GraphViz (https://graphviz.org/download/) and ensure it is "
            "available on your PATH"
        )


def save_sample_graphml(
    sample: Any, path: str, value_detail: int = 0, settings: DrawingSettings | None = None
) -> None:
    """Saves the sample graph as a GraphML file consumable by networkx.

    Parameters
    ----------
    sample : Any
        The sample to save.
    path : str
        The destination file path. Parent directories are created if needed.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.
    """
    path = os.path.abspath(path)
    sample_drawer = get_sample_drawer(settings or get_drawing_settings(value_detail=value_detail))
    sample_drawer.saveGraphML(sample._java_sample, path)

    if not os.path.exists(path):
        raise RuntimeError(
            "GraphML saving failed. Try to install GraphViz (https://graphviz.org/download/) and ensure it is "
            "available on your PATH"
        )


def pipeline_to_dot_source(
    pipeline: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> str:
    """Renders the pipeline into its dot source representation.

    Parameters
    ----------
    pipeline : Any
        The backend pipeline to render.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    str
        The dot source representation.
    """
    pipeline_drawer = get_pipeline_drawer(settings or get_drawing_settings(value_detail=value_detail))

    return to_dot_source(pipeline_drawer, pipeline)


def pipeline_to_graphml_source(
    pipeline: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> str | None:
    """Renders the pipeline into its GraphML source representation.

    Parameters
    ----------
    pipeline : Any
        The backend pipeline to render.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    str | None
        The GraphML source, or None if the backend graphviz-based conversion failed.
    """
    pipeline_drawer = get_pipeline_drawer(settings or get_drawing_settings(value_detail=value_detail))

    return to_graphml_source(pipeline_drawer, pipeline)


def save_pipeline_graphml(
    pipeline: Any, path: str, value_detail: int = 0, settings: DrawingSettings | None = None
) -> None:
    """Saves the pipeline graph as a GraphML file consumable by networkx.

    Parameters
    ----------
    pipeline : Any
        The backend pipeline to save.
    path : str
        The destination file path. Parent directories are created if needed.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.
    """
    path = os.path.abspath(path)
    pipeline_drawer = get_pipeline_drawer(settings or get_drawing_settings(value_detail=value_detail))
    pipeline_drawer.saveGraphML(pipeline, path)

    if not os.path.exists(path):
        raise RuntimeError(
            "GraphML saving failed. Try to install GraphViz (https://graphviz.org/download/) and ensure it is "
            "available on your PATH"
        )


def pipeline_to_networkx(
    pipeline: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> Any:
    """Converts the pipeline into a networkx graph.

    Parameters
    ----------
    pipeline : Any
        The backend pipeline to convert.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    networkx.DiGraph
        The pipeline as a directed graph. Node labels are stored in the "label" attribute.
    """
    return graphml_to_networkx(pipeline_to_graphml_source(pipeline, value_detail, settings))


def model_to_networkx(
    model: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> Any:
    """Converts the model into a networkx graph.

    Parameters
    ----------
    model : NeuralModule
        The model to convert.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    networkx.DiGraph
        The model as a directed graph. Node labels are stored in the "label" attribute.
    """
    return graphml_to_networkx(model_to_graphml_source(model, value_detail, settings))


def sample_to_networkx(
    sample: Any, value_detail: int = 0, settings: DrawingSettings | None = None
) -> Any:
    """Converts the sample into a networkx graph.

    Parameters
    ----------
    sample : Any
        The sample to convert.
    value_detail : int
        The level of detail for values. Default: 0.
    settings : DrawingSettings, optional
        Standalone drawing settings. When provided, ``value_detail`` is ignored.

    Returns
    -------
    networkx.DiGraph
        The sample as a directed graph. Node labels are stored in the "label" attribute.
    """
    return graphml_to_networkx(sample_to_graphml_source(sample, value_detail, settings))


def is_jupyter() -> bool:
    try:
        __IPYTHON__  # noqa: F821
        return True
    except NameError:
        return False
