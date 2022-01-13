from enum import auto
from typing import Any, Optional

import numpy as np
from pydantic import validator

from ..events import EventedModel
from ..events.custom_types import Array
from ..misc import StringEnum
from ..translations import trans
from .colorbars import make_colorbar
from .standardize_color import transform_color


class ColormapInterpolationMode(StringEnum):
    """Interpolation mode for colormaps.

    Attributes
    ----------
    LINEAR
        Colors are defined by linear interpolation between colors of
        neighboring control points.
    ZERO
        Colors are defined by the value of the color in the bin between
        neighboring control points.
    """

    LINEAR = auto()
    ZERO = auto()


class Colormap(EventedModel):
    """Defines a map from quantitative values to colors.

    This is typically used to map image pixel values to displayed colors,
    but in general can be used with any quantitative input values.
    The input values should typically be floating point values in the closed
    interval [0, 1], as any values outside this interval will be clamped.

    The correspondence between input values and colors is mostly defined by
    the ``colors`` and ``controls`` array attributes.
    Each control point in the ``controls`` array defines an input value in the
    closed interval [0, 1] and corresponds to an output RGBA color defined
    at the same index in the ``colors`` array.

    The mapping is performed by calling the ``map`` method with an array of
    input values. Typically, an input value will not exactly equal one
    of the control point values, so cannot be exactly mapped to a color.
    Instead, the output color is interpolated from the colors that correspond
    to the control points that are neighbors of the input value.
    A value of 0 or less is mapped to the first color, and a value of 1 or more
    is mapped to the last color.

    Attributes
    ----------
    colors : array, shape (N, 4)
        The colors to which this maps, where each row is an RGBA color.
    controls : array, shape (N,) or (N + 1,)
        The quantitative control points that define how input values are
        mapped to colors.
        The first value must be 0, the last value must be 1, all others
        should be in the open interval (0, 1) and sorted in increasing order.
        The required length of this depends on the length of colors and the
        desired interpolation mode.
    interpolation : {'linear', 'zero'}
        The mode used to interpolate colors.
        If 'linear', there is a one-to-one correspondence between control
        points and colors, so that ``len(controls) == len(colors)``.
        If 'zero', the control points represent the edges of histogram bins
        with one color per bin, so that ``len(controls) == len(colors) + 1``.
    name : str
        A name that uniquely identifies this colormap instance among others.
    _display_name : str
        The display name that may be a translation of the name.

    Examples
    --------

    Define a colormap that uniformly maps values in [0, 1] to red, green, and blue
    colors specified as RGB lists.

    >>> colormap = Colormap(
    ...     colors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    ...     controls=[0, 0.5, 1],
    ...     name='rgb',
    ... )
    >>> colormap.map(-1)
    array([[1., 0., 0., 1.]])
    >>> colormap.map(0)
    array([[1., 0., 0., 1.]])
    >>> colormap.map(0.25)
    array([[0.5, 0.5, 0., 1.]])
    >>> colormap.map(0.5)
    array([[0., 1., 0., 1.]])
    >>> colormap.map(1)
    array([[0., 0., 1., 1.]])
    >>> colormap.map(1.5)
    array([[0., 0., 1., 1.]])
    """

    # fields
    colors: Array[float, (-1, 4)]
    name: str = 'custom'
    _display_name: Optional[str] = None
    interpolation: ColormapInterpolationMode = ColormapInterpolationMode.LINEAR
    controls: Array[float, (-1,)] = None

    def __init__(self, colors, display_name: Optional[str] = None, **data):
        if display_name is None:
            display_name = data.get('name', 'custom')

        super().__init__(colors=colors, **data)
        self._display_name = display_name

    def __iter__(self):
        yield from (self.colors, self.controls, self.interpolation)

    def map(self, values: Any) -> np.ndarray:
        """Maps input values to RGBA colors.

        Parameters
        ----------
        values
            The input values to map. May be a single scalar value or
            a 1D array-like. Typically floating point values in [0, 1].

        Returns
        -------
        np.ndarray
            An (N, 4) array of colors where each row is an RGBA color.
        """
        values = np.atleast_1d(values)
        if self.interpolation == ColormapInterpolationMode.LINEAR:
            # One color per control point
            cols = [
                np.interp(values, self.controls, self.colors[:, i])
                for i in range(4)
            ]
            cols = np.stack(cols, axis=1)
        elif self.interpolation == ColormapInterpolationMode.ZERO:
            # One color per bin
            indices = np.clip(
                np.searchsorted(self.controls, values) - 1, 0, len(self.colors)
            )
            cols = self.colors[indices.astype(np.int32)]
        else:
            raise ValueError(
                trans._(
                    'Unrecognized Colormap Interpolation Mode',
                    deferred=True,
                )
            )

        return cols

    @property
    def colorbar(self):
        """The colorbar used to visualize this with the default shape and horizontal orientation.

        See Also
        --------
        make_colorbar
        """
        return make_colorbar(self)

    @validator('colors', pre=True)
    def _ensure_color_array(cls, v):
        return transform_color(v)

    # Set always=True to handle controls == None.
    @validator('controls', pre=True, always=True)
    def _check_controls(cls, v, values):
        # If no control points provided generate defaults
        if v is None or len(v) == 0:
            n_controls = len(values['colors']) + int(
                values['interpolation'] == ColormapInterpolationMode.ZERO
            )
            return np.linspace(0, 1, n_controls)

        # Check control end points are correct
        if v[0] != 0 or v[-1] != 1:
            raise ValueError(
                trans._(
                    'Control points must start with 0.0 and end with 1.0. Got {start_control_point} and {end_control_point}',
                    deferred=True,
                    start_control_point=v[0],
                    end_control_point=v[-1],
                )
            )

        # Check control points are sorted correctly
        if not np.array_equal(v, sorted(v)):
            raise ValueError(
                trans._(
                    'Control points need to be sorted in ascending order',
                    deferred=True,
                )
            )

        # Check number of control points is correct
        n_controls_target = len(values['colors']) + int(
            values['interpolation'] == ColormapInterpolationMode.ZERO
        )
        n_controls = len(v)
        if n_controls != n_controls_target:
            raise ValueError(
                trans._(
                    'Wrong number of control points provided. Expected {n_controls_target}, got {n_controls}',
                    deferred=True,
                    n_controls_target=n_controls_target,
                    n_controls=n_controls,
                )
            )

        return v
