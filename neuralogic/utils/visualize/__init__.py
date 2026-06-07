from __future__ import annotations

import io
import os
import tempfile
from typing import TYPE_CHECKING, Any

import jpype

from neuralogic.setup import get_default_graphviz_path

if TYPE_CHECKING:
    from neuralogic.core.settings import Settings, SettingsProxy


def get_graphviz_path(path: str | None = None) -> str | None:
    """
    Get the path to the Graphviz executable
    """
    if path is not None:
        return path
    return get_default_graphviz_path()


def get_drawing_settings(
    img_type: str = "png", value_detail: int = 0, graphviz_path: str | None = None
) -> SettingsProxy:
    """Returns the default settings instance for drawing with a specified image type.

    Parameters
    ----------
    img_type : str
        The image type. Default: "png".
    value_detail : int
        The level of detail for values. Default: 0.
    graphviz_path : str, optional
        The path to the Graphviz executable. Default: None.

    Returns
    -------
    SettingsProxy
        The settings proxy for drawing.
    """
    from neuralogic.core.settings import Settings

    settings = Settings().create_proxy()

    graphviz = get_graphviz_path(graphviz_path)
    if graphviz is not None:
        settings.settings.graphvizPath = graphviz

    settings.settings.drawing = False
    settings.settings.storeNotShow = True
    settings.settings.imgType = img_type.lower()
    settings.settings.outDir = tempfile.gettempdir()

    if value_detail not in [0, 1, 2]:
        raise ValueError(f"Invalid value_detail - {value_detail}. Expected 0, 1, or 2.")

    settings_class = settings.settings_class
    details = [
        settings_class.shortNumberFormat,
        settings_class.detailedNumberFormat,
        settings_class.superDetailedNumberFormat,
    ]

    settings.settings.defaultNumberFormat = details[value_detail]

    return settings


def get_model_drawer(settings: SettingsProxy) -> Any:
    """Returns the model drawer.

    Parameters
    ----------
    settings : SettingsProxy
        The settings proxy.

    Returns
    -------
    Any
        The model drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.TemplateDrawer")(settings.settings)


def get_sample_drawer(settings: SettingsProxy) -> Any:
    """Returns the sample drawer.

    Parameters
    ----------
    settings : SettingsProxy
        The settings proxy.

    Returns
    -------
    Any
        The sample drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.NeuralNetDrawer")(settings.settings)


def get_grounding_drawer(settings: SettingsProxy) -> Any:
    """Returns the grounding drawer.

    Parameters
    ----------
    settings : SettingsProxy
        The settings proxy.

    Returns
    -------
    Any
        The grounding drawer.
    """
    return jpype.JClass("cz.cvut.fel.ida.pipelines.debugging.drawing.GroundingDrawer")(settings.settings)


# todo gusta: + groundingDrawer, pipelineDrawer...


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


def draw_model(
    model: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
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
    template_drawer = get_model_drawer(get_drawing_settings(img_type, value_detail, graphviz_path))

    return draw(template_drawer, model, filename, show, img_type, *args, **kwargs)


def draw_grounding(
    grounding: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
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
    args : Any
        Additional arguments for the drawer.
    kwargs : Any
        Additional keyword arguments for the drawer.

    Returns
    -------
    Union[Any, bytes, None]
        The grounding drawing.
    """
    grounding_drawer = get_grounding_drawer(get_drawing_settings(img_type, value_detail, graphviz_path))

    return draw(grounding_drawer, grounding, filename, show, img_type, *args, **kwargs)


def draw_sample(
    sample: Any,
    filename: str | None = None,
    show: bool = True,
    img_type: str = "png",
    value_detail: int = 0,
    graphviz_path: str | None = None,
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

    sample_drawer = get_sample_drawer(get_drawing_settings(img_type, value_detail, graphviz_path))

    return draw(sample_drawer, draw_object, filename, show, img_type, *args, **kwargs)


def model_to_dot_source(model: Any) -> str:
    """Renders the model into its dot source representation.

    Parameters
    ----------
    model : NeuralModule
        The model to render.

    Returns
    -------
    str
        The dot source representation.
    """
    if model._need_sync:
        model._sync_model()

    model = model._model
    template_drawer = get_model_drawer(get_drawing_settings())

    return to_dot_source(template_drawer, model)


def sample_to_dot_source(sample: Any, value_detail: int = 0) -> str:
    """Renders the sample into its dot source representation.

    Parameters
    ----------
    sample : Any
        The sample to render.
    value_detail : int
        The level of detail for values. Default: 0.

    Returns
    -------
    str
        The dot source representation.
    """
    sample_drawer = get_sample_drawer(get_drawing_settings(value_detail=value_detail))

    return to_dot_source(sample_drawer, sample._java_sample)


def is_jupyter() -> bool:
    try:
        __IPYTHON__  # noqa: F821
        return True
    except NameError:
        return False
