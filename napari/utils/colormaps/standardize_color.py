"""This module contains functions that 'standardize' the color handling
of napari layers by supplying functions that are able to convert most
color representation the user had in mind into a single representation -
a numpy Nx4 array of float32 values between 0 and 1 - that is used across
the codebase. The color is always in an RGBA format. To handle colors in
HSV, for example, we should point users to skimage, matplotlib and others.

The main function of the module is "transform_color", which might call
a cascade of other, private, function in the module to do the hard work
of converting the input. This function will either be called directly, or
used by the function "transform_color_with_defaults", which is a helper
function for the layer objects located in
``layers.utils.color_transformations.py``.

In general, when handling colors we try to catch invalid color
representations, warn the users of their misbehaving and return a default
white color array, since it seems unreasonable to crash the entire napari
session due to mis-represented colors.
"""

import types
import warnings
from typing import Any, Sequence

import numpy as np
import pydantic.color
from vispy.color import _color_dict, get_color_dict, get_color_names

from ..translations import trans


def transform_color(colors: Any) -> np.ndarray:
    """Transforms provided color(s) to an Nx4 array of RGBA np.float32
    values.

    N is the number of given colors. The function is designed to parse all
    valid color representations a user might have and convert them properly.
    That being said, combinations of different color representation in the
    same list of colors is prohibited, and will error. This means that a list
    of ['red', np.array([1, 0, 0])] cannot be parsed and has to be manually
    pre-processed by the user before sent to this function. In addition, the
    provided colors - if numeric - should already be in an RGB(A) format. To
    convert an existing numeric color array to RGBA format use skimage before
    calling this function.

    Parameters
    ----------
    colors : string and array-like.
        The color(s) to interpret and convert

    Returns
    -------
    colors : np.ndarray
        An instance of np.ndarray with a data type of float32, 4 columns in
        RGBA order and N rows, with N being the number of colors. The array
        will always be 2D even if a single color is passed.

    Raises
    ------
    ValueError, AttributeError, KeyError
        invalid inputs
    """
    if colors is None:
        return np.zeros((1, 4), dtype=np.float32)

    # Always make this into a numpy array, so we can reason about
    # dtype and shape easily.
    if isinstance(colors, types.GeneratorType):
        colors = list(colors)
    colors = np.asarray(colors)

    # If the array contains numpy strings, then we will stupidly
    # convert back to regular strings, which is not ideal.
    if np.issubdtype(colors.dtype, str):
        if colors.ndim == 0:
            return _multi_rgba_from_pydantic_colors([str(colors)])
        elif colors.ndim == 1:
            return _multi_rgba_from_pydantic_colors([str(c) for c in colors])
        else:
            warnings.warn(
                trans._(
                    "String color arrays should be one-dimensional. Converting input to a white color array.",
                    deferred=True,
                )
            )
            return np.ones((max(1, len(colors)), 4), dtype=np.float32)

    # We only support strings and numbers.
    if not np.issubdtype(colors.dtype, np.number):
        raise TypeError('Bad dtype')

    # Check the shape, then convert and normalize.
    colors = np.atleast_2d(colors)
    if colors.ndim > 2:
        raise ValueError('Too many dims')
    if colors.shape[1] not in (3, 4):
        raise ValueError('Bad shape')
    return _convert_array_to_correct_format(colors)


def _multi_rgba_from_pydantic_colors(colors) -> np.ndarray:
    if len(colors) == 0:
        return np.zeros((1, 4), dtype=np.float32)
    return np.array(
        [_single_rgba_from_pydantic_color(color) for color in colors]
    )


def _single_rgba_from_pydantic_color(color) -> np.ndarray:
    if isinstance(color, str) and color in _color_dict._color_dict:
        color = _color_dict._color_dict[color]
    color = pydantic.color.Color(color)
    rgba_tuple = color.as_rgb_tuple(alpha=True)
    rgba_array = np.array(rgba_tuple[:3]) / 255
    return np.append(rgba_array, rgba_tuple[-1]).astype(dtype=np.float32)


def _convert_array_to_correct_format(colors: np.ndarray) -> np.ndarray:
    """Asserts shape, dtype and normalization of given color array.

    This function deals with arrays which are already 'well-behaved',
    i.e. have (almost) the correct number of columns and are able to represent
    colors correctly. It then it makes sure that the array indeed has exactly
    four columns and that its values are normalized between 0 and 1, with a
    data type of float32.

    Parameters
    ----------
    colors : np.ndarray
        Input color array, perhaps un-normalized and without the alpha channel.

    Returns
    -------
    colors : np.ndarray
        Nx4, float32 color array with values in the range [0, 1]
    """
    if colors.shape[1] == 3:
        colors = np.column_stack(
            [colors, np.ones(len(colors), dtype=np.float32)]
        )

    if colors.min() < 0:
        raise ValueError(
            trans._(
                "Colors input had negative values.",
                deferred=True,
            )
        )

    if colors.max() > 1:
        warnings.warn(
            trans._(
                "Colors with values larger than one detected. napari will normalize these colors for you. If you'd like to convert these yourself, please use the proper method from skimage.color.",
                deferred=True,
            )
        )
        colors = _normalize_color_array(colors)
    return np.atleast_2d(np.asarray(colors, dtype=np.float32))


def _normalize_color_array(colors: np.ndarray) -> np.ndarray:
    """Normalize all array values to the range [0, 1].

    The added complexity here stems from the fact that if a row in the given
    array contains four identical value a simple normalization might raise a
    division by zero exception.

    Parameters
    ----------
    colors : np.ndarray
        A numpy array with values possibly outside the range of [0, 1]

    Returns
    -------
    colors : np.ndarray
        Copy of input array with normalized values
    """
    colors = colors.astype(np.float32, copy=True)
    out_of_bounds_idx = np.unique(np.where((colors > 1) | (colors < 0))[0])
    out_of_bounds = colors[out_of_bounds_idx]
    norm = np.linalg.norm(out_of_bounds, np.inf, axis=1)
    out_of_bounds = out_of_bounds / norm[:, np.newaxis]
    colors[out_of_bounds_idx] = out_of_bounds
    return colors.astype(np.float32)


def _create_hex_to_name_dict():
    """Create a dictionary mapping hexadecimal RGB colors into their
    'official' name.

    Returns
    -------
    hex_to_rgb : dict
        Mapping from hexadecimal RGB ('#ff0000') to name ('red').
    """
    colordict = get_color_dict()
    hex_to_name = {f"{v.lower()}ff": k for k, v in colordict.items()}
    hex_to_name["#00000000"] = "transparent"
    return hex_to_name


def get_color_namelist():
    """A wrapper around vispy's get_color_names designed to add a
    "transparent" (alpha = 0) color to it.

    Once https://github.com/vispy/vispy/pull/1794 is merged this
    function is no longer necessary.

    Returns
    -------
    color_dict : list
        A list of all valid vispy color names plus "transparent".
    """
    names = get_color_names()
    names.append('transparent')
    return names


hex_to_name = _create_hex_to_name_dict()


def _check_color_dim(val):
    """Ensures input is Nx4.

    Parameters
    ----------
    val : np.ndarray
        A color array of possibly less than 4 columns

    Returns
    -------
    val : np.ndarray
        A four columns version of the input array. If the original array
        was a missing the fourth channel, it's added as 1.0 values.
    """
    val = np.atleast_2d(val)
    if val.shape[1] not in (3, 4):
        raise RuntimeError(
            trans._(
                'Value must have second dimension of size 3 or 4',
                deferred=True,
            )
        )

    if val.shape[1] == 3:
        val = np.column_stack([val, np.float32(1.0)])
    return val


def rgb_to_hex(rgbs: Sequence) -> np.ndarray:
    """Convert RGB to hex quadruplet.

    Taken from vispy with slight modifications.

    Parameters
    ----------
    rgbs : Sequence
        A list-like container of colors in RGBA format with values
        between [0, 1]

    Returns
    -------
    arr : np.ndarray
        An array of the hex representation of the input colors

    """
    rgbs = _check_color_dim(rgbs)
    return np.array(
        [
            f'#{"%02x" * 4}' % tuple((255 * rgb).astype(np.uint8))
            for rgb in rgbs
        ],
        '|U9',
    )
