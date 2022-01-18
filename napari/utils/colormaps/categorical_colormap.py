from typing import Any, Dict, Union

import numpy as np
from pydantic import validator

from ...utils.events import EventedModel
from ...utils.events.custom_types import Array
from ..translations import trans
from .categorical_colormap_utils import ColorCycle, compare_colormap_dicts
from .standardize_color import transform_color


class CategoricalColormap(EventedModel):
    """Defines a map from categorical values to colors.

    This is typically used to layer feature values to displayed colors.
    In most cases, only one of the ``colormap`` and ``fallback_color`` parameters
    is specified on initialization.
    If the mapping is known, then the ``colormap`` should be specified as a
    dictionary that directly maps each categorical value to an RGBA color.
    If the mapping is unknown or there is no need to explicitly define it,
    then the ``fallback_color`` cycle should be specified as a sequence of
    desired output colors.

    The mapping is performed by calling the ``map`` method with an array of
    input values. If an input value is in ``colormap``, then it is mapped to
    its corresponding color entry. If it is absent, it is mapped to the next
    color in the ``fallback_color`` cycle, and the new mapping entry is added
    to ``colormap``.

    Attributes
    ----------
    colormap : Dict[Any, np.ndarray]
        The mapping between categorical property values and color.
    fallback_color : ColorCycle
        The color to be used in the case that a value is mapped that is not
        in colormap. This can be given as any ColorType and it will be converted
        to a ColorCycle. An array of the values contained in the
        ColorCycle.cycle is stored in ColorCycle.values.
        The default value is a cycle of all white.

    Examples
    --------

    Define a colormap from a dictionary from string labels to RGB color lists.

    >>> colormap = CategoricalColormap(
    ...         colormap={
    ...             'astrocyte': [1, 0, 0],
    ...             'oligodendrocyte': [0, 1, 0],
    ...             'ependymal': [0, 0, 1],
    ...         },
    ... )
    >>> colormap.map('astrocyte')
    array([[1., 0., 0., 1.]], dtype=float32)
    >>> colormap.map('radial')
    array([[1., 1., 1., 1.]], dtype=float32)
    >>> colormap.colormap
    {'astrocyte': array([1., 0., 0., 1.], dtype=float32),
    'oligodendrocyte': array([0., 1., 0., 1.], dtype=float32),
    'ependymal': array([0., 0., 1., 1.], dtype=float32),
    'radial': array([1., 1., 1., 1.], dtype=float32)}

    Define a colormap from a color cycle specified as RGB color lists.

    >>> colormap = CategoricalColormap(
    ...         fallback_color=[
    ...             [1, 0, 0],
    ...             [0, 1, 0],
    ...             [0, 0, 1],
    ...         ],
    ... )
    >>> colormap.map('astrocyte')
    array([[1., 0., 0., 1.]], dtype=float32)
    >>> colormap.colormap
    {'astrocyte': array([1., 0., 0., 1.], dtype=float32)}
    >>> colormap.map('radial')
    array([[0., 1., 0., 1.]], dtype=float32)
    >>> colormap.colormap
    {'astrocyte': array([1., 0., 0., 1.], dtype=float32),
    'radial': array([0., 1., 0., 1.], dtype=float32)}
    """

    colormap: Dict[Any, Array[np.float32, (4,)]] = {}
    fallback_color: ColorCycle = 'white'

    @validator('colormap', pre=True)
    def _standardize_colormap(cls, v):
        transformed_colormap = {k: transform_color(v)[0] for k, v in v.items()}
        return transformed_colormap

    def map(self, color_properties: Union[list, np.ndarray]) -> np.ndarray:
        """Maps input values to RGBA colors.

        Parameters
        ----------
        color_properties : Union[list, np.ndarray]
            The property values to be converted to colors.

        Returns
        -------
        colors : np.ndarray
            An Nx4 color array where N is the number of property values provided.
        """
        if isinstance(color_properties, (list, np.ndarray)):
            color_properties = np.asarray(color_properties)
        else:
            color_properties = np.asarray([color_properties])

        # add properties if they are not in the colormap
        color_cycle_keys = [*self.colormap]
        props_in_map = np.in1d(color_properties, color_cycle_keys)
        if not np.all(props_in_map):
            new_prop_values = color_properties[np.logical_not(props_in_map)]
            indices_to_add = np.unique(new_prop_values, return_index=True)[1]
            props_to_add = [
                new_prop_values[index] for index in sorted(indices_to_add)
            ]
            for prop in props_to_add:
                new_color = next(self.fallback_color.cycle)
                self.colormap[prop] = np.squeeze(transform_color(new_color))
        # map the colors
        colors = np.array([self.colormap[x] for x in color_properties])
        return colors

    @classmethod
    def from_array(cls, fallback_color):
        return cls(fallback_color=fallback_color)

    @classmethod
    def from_dict(cls, params: dict):
        if ('colormap' in params) or ('fallback_color' in params):
            if 'colormap' in params:
                colormap = {
                    k: transform_color(v)[0]
                    for k, v in params['colormap'].items()
                }
            else:
                colormap = {}
            if 'fallback_color' in params:
                fallback_color = params['fallback_color']
            else:
                fallback_color = 'white'
        else:
            colormap = {k: transform_color(v)[0] for k, v in params.items()}
            fallback_color = 'white'

        return cls(colormap=colormap, fallback_color=fallback_color)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate_type

    @classmethod
    def validate_type(cls, val):
        if isinstance(val, cls):
            return val
        if isinstance(val, list) or isinstance(val, np.ndarray):
            return cls.from_array(val)
        elif isinstance(val, dict):
            return cls.from_dict(val)
        else:
            raise TypeError(
                trans._(
                    'colormap should be an array or dict',
                    deferred=True,
                )
            )

    def __eq__(self, other):
        if isinstance(other, CategoricalColormap):
            if not compare_colormap_dicts(self.colormap, other.colormap):
                return False
            if not np.allclose(
                self.fallback_color.values, other.fallback_color.values
            ):
                return False
            return True
        else:
            return False
