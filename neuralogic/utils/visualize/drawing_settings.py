# ruff: noqa: N815

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Literal, TypedDict, Unpack, overload

import jpype

from neuralogic.setup import get_default_graphviz_path, initialize, is_initialized


@dataclass
class GraphAttributes:
    """Graphviz graph attributes accepted by :class:`DrawingSettings`."""

    bgcolor: str | None = None
    colorscheme: str | None = None
    compound: str | bool | None = None
    concentrate: str | bool | None = None
    dpi: str | int | None = None
    fontname: str | None = None
    fontsize: str | int | None = None
    label: str | None = None
    labeljust: str | None = None
    labelloc: str | None = None
    margin: str | float | None = None
    nodesep: str | float | None = None
    overlap: str | bool | None = None
    pad: str | float | None = None
    rankdir: str | None = None
    ranksep: str | float | None = None
    sep: str | float | None = None
    splines: str | None = None
    style: str | None = None


@dataclass
class NodeAttributes:
    """Graphviz node attributes accepted by :class:`DrawingSettings`."""

    color: str | None = None
    distortion: str | float | None = None
    fillcolor: str | None = None
    fixedsize: str | bool | None = None
    fontcolor: str | None = None
    fontname: str | None = None
    fontsize: str | int | None = None
    height: str | float | None = None
    image: str | None = None
    imagescale: str | bool | None = None
    label: str | None = None
    orientation: str | float | None = None
    penwidth: str | float | None = None
    peripheries: str | int | None = None
    shape: str | None = None
    sides: str | int | None = None
    skew: str | float | None = None
    style: str | None = None
    width: str | float | None = None


@dataclass
class EdgeAttributes:
    """Graphviz edge attributes accepted by :class:`DrawingSettings`."""

    arrowhead: str | None = None
    arrowtail: str | None = None
    color: str | None = None
    constraint: str | bool | None = None
    dir: str | None = None
    fontcolor: str | None = None
    fontname: str | None = None
    fontsize: str | int | None = None
    label: str | None = None
    labelangle: str | float | None = None
    labeldistance: str | float | None = None
    len: str | float | None = None
    minlen: str | int | None = None
    penwidth: str | float | None = None
    style: str | None = None
    weight: str | int | float | None = None


@dataclass
class NodeStyles:
    """Node style keys accepted by :class:`DrawingSettings`."""

    literal: NodeAttributes | dict[str, Any] | None = None
    fact: NodeAttributes | dict[str, Any] | None = None
    rule: NodeAttributes | dict[str, Any] | None = None
    groundLiteral: NodeAttributes | dict[str, Any] | None = None
    groundHeadRule: NodeAttributes | dict[str, Any] | None = None
    groundRule: NodeAttributes | dict[str, Any] | None = None
    target: NodeAttributes | dict[str, Any] | None = None
    atomNeuron: NodeAttributes | dict[str, Any] | None = None
    weightedAtomNeuron: NodeAttributes | dict[str, Any] | None = None
    negationNeuron: NodeAttributes | dict[str, Any] | None = None
    aggregationNeuron: NodeAttributes | dict[str, Any] | None = None
    ruleNeuron: NodeAttributes | dict[str, Any] | None = None
    weightedRuleNeuron: NodeAttributes | dict[str, Any] | None = None
    factNeuron: NodeAttributes | dict[str, Any] | None = None
    targetNeuron: NodeAttributes | dict[str, Any] | None = None
    pipe: NodeAttributes | dict[str, Any] | None = None
    identityPipe: NodeAttributes | dict[str, Any] | None = None
    branch: NodeAttributes | dict[str, Any] | None = None
    merge: NodeAttributes | dict[str, Any] | None = None
    start: NodeAttributes | dict[str, Any] | None = None
    terminal: NodeAttributes | dict[str, Any] | None = None


@dataclass
class EdgeStyles:
    """Edge style keys accepted by :class:`DrawingSettings`."""

    ruleToBody: EdgeAttributes | dict[str, Any] | None = None
    negatedRuleToBody: EdgeAttributes | dict[str, Any] | None = None
    ruleToHead: EdgeAttributes | dict[str, Any] | None = None
    bodyToFact: EdgeAttributes | dict[str, Any] | None = None
    headToGroundRule: EdgeAttributes | dict[str, Any] | None = None
    groundHeadRuleToGroundRule: EdgeAttributes | dict[str, Any] | None = None
    groundRuleToLiteral: EdgeAttributes | dict[str, Any] | None = None
    compactRuleEdge: EdgeAttributes | dict[str, Any] | None = None
    unweightedInput: EdgeAttributes | dict[str, Any] | None = None
    weightedInput: EdgeAttributes | dict[str, Any] | None = None
    fixedWeightInput: EdgeAttributes | dict[str, Any] | None = None
    pipelineEdge: EdgeAttributes | dict[str, Any] | None = None
    pipelineTypeEdge: EdgeAttributes | dict[str, Any] | None = None


class _GraphAttributesDict(TypedDict, total=False):
    bgcolor: str
    colorscheme: str
    compound: str | bool
    concentrate: str | bool
    dpi: str | int
    fontname: str
    fontsize: str | int
    label: str
    labeljust: str
    labelloc: str
    margin: str | float
    nodesep: str | float
    overlap: str | bool
    pad: str | float
    rankdir: str
    ranksep: str | float
    sep: str | float
    splines: str
    style: str


class _NodeAttributesDict(TypedDict, total=False):
    color: str
    distortion: str | float
    fillcolor: str
    fixedsize: str | bool
    fontcolor: str
    fontname: str
    fontsize: str | int
    height: str | float
    image: str
    imagescale: str | bool
    label: str
    orientation: str | float
    penwidth: str | float
    peripheries: str | int
    shape: str
    sides: str | int
    skew: str | float
    style: str
    width: str | float


class _EdgeAttributesDict(TypedDict, total=False):
    arrowhead: str
    arrowtail: str
    color: str
    constraint: str | bool
    dir: str
    fontcolor: str
    fontname: str
    fontsize: str | int
    label: str
    labelangle: str | float
    labeldistance: str | float
    len: str | float
    minlen: str | int
    penwidth: str | float
    style: str
    weight: str | int | float


NodeStyleKey = Literal[
    "literal",
    "fact",
    "rule",
    "groundLiteral",
    "groundHeadRule",
    "groundRule",
    "target",
    "atomNeuron",
    "weightedAtomNeuron",
    "negationNeuron",
    "aggregationNeuron",
    "ruleNeuron",
    "weightedRuleNeuron",
    "factNeuron",
    "targetNeuron",
    "pipe",
    "identityPipe",
    "branch",
    "merge",
    "start",
    "terminal",
]

EdgeStyleKey = Literal[
    "ruleToBody",
    "negatedRuleToBody",
    "ruleToHead",
    "bodyToFact",
    "headToGroundRule",
    "groundHeadRuleToGroundRule",
    "groundRuleToLiteral",
    "compactRuleEdge",
    "unweightedInput",
    "weightedInput",
    "fixedWeightInput",
    "pipelineEdge",
    "pipelineTypeEdge",
]

GraphAttributeKey = Literal[
    "bgcolor",
    "colorscheme",
    "compound",
    "concentrate",
    "dpi",
    "fontname",
    "fontsize",
    "label",
    "labeljust",
    "labelloc",
    "margin",
    "nodesep",
    "overlap",
    "pad",
    "rankdir",
    "ranksep",
    "sep",
    "splines",
    "style",
]

NodeAttributeKey = Literal[
    "color",
    "distortion",
    "fillcolor",
    "fixedsize",
    "fontcolor",
    "fontname",
    "fontsize",
    "height",
    "image",
    "imagescale",
    "label",
    "orientation",
    "penwidth",
    "peripheries",
    "shape",
    "sides",
    "skew",
    "style",
    "width",
]

EdgeAttributeKey = Literal[
    "arrowhead",
    "arrowtail",
    "color",
    "constraint",
    "dir",
    "fontcolor",
    "fontname",
    "fontsize",
    "label",
    "labelangle",
    "labeldistance",
    "len",
    "minlen",
    "penwidth",
    "style",
    "weight",
]


_NODE_STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "fact": ("fact", "groundLiteral", "factNeuron"),
    "rule": ("rule", "groundHeadRule", "groundRule", "ruleNeuron"),
    "literal": ("literal", "groundLiteral"),
    "target": ("target", "targetNeuron"),
}

_EDGE_STYLE_ALIASES: dict[str, tuple[str, ...]] = {
    "ruleToBody": ("ruleToBody", "headToGroundRule"),
    "ruleToHead": ("ruleToHead", "groundHeadRuleToGroundRule"),
    "bodyToFact": ("bodyToFact", "groundRuleToLiteral"),
}


def _as_plain_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        value = asdict(value)

    return {key: val for key, val in value.items() if val is not None}


class DrawingSettings:
    """Standalone settings for graph drawing, independent of :class:`SettingsProxy`.

    This class owns its own Java ``Settings`` instance and configures only the
    drawing-related part of the backend. It exposes the Java ``DrawingStyle`` through
    a small JPype-friendly Python API, so styles and value detail can be configured
    without touching the training settings.

    Examples
    --------
    >>> settings = DrawingSettings(
    ...     value_detail=1,
    ...     display_values=False,
    ...     graph=GraphAttributes(bgcolor="white"),
    ...     node=NodeStyles(fact=NodeAttributes(shape="box", fillcolor="#fff2cc")),
    ...     edge=EdgeStyles(ruleToHead=EdgeAttributes(color="red")),
    ... )
    """

    _VALUE_DETAIL_DECIMALS = {0: 2, 1: 10, 2: 16}

    def __init__(
        self,
        value_detail: int = 0,
        *,
        value_decimals: int | None = None,
        display_values: bool | None = None,
        display_gradients: bool | None = None,
        display_dimensions: bool | None = None,
        display_functions: bool | None = None,
        display_weights: bool | None = None,
        graph: GraphAttributes | dict[str, Any] | None = None,
        node: NodeStyles | dict[str, dict[str, Any]] | None = None,
        edge: EdgeStyles | dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not is_initialized():
            initialize()

        self.settings_class = jpype.JClass("cz.cvut.fel.ida.setup.Settings")
        self.settings = self.settings_class()

        self.settings.drawing = False  # drawers create GraphViz lazily via drawInto*
        self.settings.storeNotShow = True
        self.settings.outDir = tempfile.gettempdir()
        self.settings.imageFile = os.path.join(tempfile.gettempdir(), "graph")

        # Output file name and image type are owned by the draw() call, not by
        # these styling settings. Defaults are set here so drawers can be created
        # directly, and draw() overrides them from its own arguments.
        self.settings.imgType = "png"

        graphviz = get_default_graphviz_path()
        if graphviz is not None:
            self.settings.graphvizPath = graphviz

        self._style = self.settings.drawingStyle
        self.value_detail = value_detail

        if value_decimals is not None:
            self.value_decimals = value_decimals
        if display_values is not None:
            self.display_values = display_values
        if display_gradients is not None:
            self.display_gradients = display_gradients
        if display_dimensions is not None:
            self.display_dimensions = display_dimensions
        if display_functions is not None:
            self.display_functions = display_functions
        if display_weights is not None:
            self.display_weights = display_weights

        if graph is not None:
            for attribute, value in _as_plain_dict(graph).items():
                self.set_graph_attr(attribute, value)
        if node is not None:
            for style_key, attributes in _as_plain_dict(node).items():
                for attribute, value in _as_plain_dict(attributes).items():
                    self.set_node_attr(style_key, attribute, value)
        if edge is not None:
            for style_key, attributes in _as_plain_dict(edge).items():
                for attribute, value in _as_plain_dict(attributes).items():
                    self.set_edge_attr(style_key, attribute, value)

    def _set_img_type(self, img_type: str) -> None:
        self.settings.imgType = img_type.lower()

    def _set_graphviz_path(self, graphviz_path: str | None) -> None:
        if graphviz_path is not None:
            self.settings.graphvizPath = graphviz_path

    @property
    def value_detail(self) -> int:
        return self._value_detail

    @value_detail.setter
    def value_detail(self, value_detail: int) -> None:
        if value_detail not in self._VALUE_DETAIL_DECIMALS:
            raise ValueError(f"Invalid value_detail - {value_detail}. Expected 0, 1, or 2.")
        self._value_detail = value_detail
        self._style.valueDecimals = self._VALUE_DETAIL_DECIMALS[value_detail]

    @property
    def value_decimals(self) -> int:
        """Number of decimal places for numeric values in drawings."""
        return int(self._style.valueDecimals)

    @value_decimals.setter
    def value_decimals(self, value: int) -> None:
        self._style.valueDecimals = int(value)

    @property
    def display_values(self) -> bool:
        """Master switch for numeric values in drawings."""
        return bool(self._style.displayValues)

    @display_values.setter
    def display_values(self, value: bool) -> None:
        self._style.displayValues = bool(value)

    @property
    def display_gradients(self) -> bool:
        return bool(self._style.displayGradients)

    @display_gradients.setter
    def display_gradients(self, value: bool) -> None:
        self._style.displayGradients = bool(value)

    @property
    def display_dimensions(self) -> bool:
        return bool(self._style.displayDimensions)

    @display_dimensions.setter
    def display_dimensions(self, value: bool) -> None:
        self._style.displayDimensions = bool(value)

    @property
    def display_functions(self) -> bool:
        return bool(self._style.displayFunctions)

    @display_functions.setter
    def display_functions(self, value: bool) -> None:
        self._style.displayFunctions = bool(value)

    @property
    def display_weights(self) -> bool:
        return bool(self._style.displayWeights)

    @display_weights.setter
    def display_weights(self, value: bool) -> None:
        self._style.displayWeights = bool(value)

    @property
    def style(self) -> Any:
        """The raw Java ``DrawingStyle`` object, for full escape-hatch access."""
        return self._style

    @overload
    def set_graph_attr(self, attribute: GraphAttributeKey, value: Any) -> None: ...

    @overload
    def set_graph_attr(self, attribute: str, value: Any) -> None: ...

    def set_graph_attr(self, attribute: str, value: Any) -> None:
        self._style.setGraphAttribute(str(attribute), str(value))

    @overload
    def set_node_attr(self, style_key: NodeStyleKey, attribute: NodeAttributeKey, value: Any) -> None: ...

    @overload
    def set_node_attr(self, style_key: str, attribute: str, value: Any) -> None: ...

    def set_node_attr(self, style_key: str, attribute: str, value: Any) -> None:
        for key in _NODE_STYLE_ALIASES.get(style_key, (style_key,)):
            self._style.setNodeAttribute(str(key), str(attribute), str(value))

    @overload
    def set_edge_attr(self, style_key: EdgeStyleKey, attribute: EdgeAttributeKey, value: Any) -> None: ...

    @overload
    def set_edge_attr(self, style_key: str, attribute: str, value: Any) -> None: ...

    def set_edge_attr(self, style_key: str, attribute: str, value: Any) -> None:
        for key in _EDGE_STYLE_ALIASES.get(style_key, (style_key,)):
            self._style.setEdgeAttribute(str(key), str(attribute), str(value))

    def graph(self, **attributes: Unpack[_GraphAttributesDict]) -> None:
        for attribute, value in attributes.items():
            self.set_graph_attr(attribute, value)

    def node(self, style_key: NodeStyleKey, **attributes: Unpack[_NodeAttributesDict]) -> None:
        for attribute, value in attributes.items():
            self.set_node_attr(style_key, attribute, value)

    def edge(self, style_key: EdgeStyleKey, **attributes: Unpack[_EdgeAttributesDict]) -> None:
        for attribute, value in attributes.items():
            self.set_edge_attr(style_key, attribute, value)
