from typing import Any
from uuid import uuid4

import pytest

from napari.utils.migrations import (
    DeprecatingDict,
    add_deprecated_property,
    deprecated_class_name,
    rename_argument,
)


def test_simple():
    @rename_argument('a', 'b', '1', '0.5')
    def sample_fun(b):
        return b

    assert sample_fun(1) == 1
    assert sample_fun(b=1) == 1
    with pytest.deprecated_call():
        assert sample_fun(a=1) == 1
    with pytest.raises(ValueError, match='already defined'):
        sample_fun(b=1, a=1)


def test_constructor():
    class Sample:
        @rename_argument('a', 'b', '1', '0.5')
        def __init__(self, b) -> None:
            self.b = b

    assert Sample(1).b == 1
    assert Sample(b=1).b == 1
    with pytest.deprecated_call():
        assert Sample(a=1).b == 1


def test_deprecated_property() -> None:
    class Dummy:
        def __init__(self) -> None:
            self._value = 0

        @property
        def new_property(self) -> int:
            return self._value

        @new_property.setter
        def new_property(self, value: int) -> int:
            self._value = value

    instance = Dummy()

    add_deprecated_property(
        Dummy, 'old_property', 'new_property', '0.1.0', '0.0.0'
    )

    assert instance.new_property == 0

    instance.new_property = 1

    msg = 'Dummy.old_property is deprecated since 0.0.0 and will be removed in 0.1.0. Please use new_property'

    with pytest.warns(FutureWarning, match=msg):
        assert instance.old_property == 1

    with pytest.warns(FutureWarning, match=msg):
        instance.old_property = 2

    assert instance.new_property == 2


def test_deprecated_class_name():
    """Test the deprecated class name function."""

    class macOS:
        pass

    MacOSX = deprecated_class_name(
        macOS, 'MacOSX', version='10.12', since_version='10.11'
    )

    with pytest.warns(FutureWarning, match='deprecated.*macOS'):
        _os = MacOSX()

    with pytest.warns(FutureWarning, match='deprecated.*macOS'):

        class MacOSXServer(MacOSX):
            pass


def deprecating_dict_with_derived() -> tuple[DeprecatingDict, str]:
    d = DeprecatingDict({'a': 1, 'b': 2})

    def getter(x: dict[str, Any]) -> Any:
        return (x['a'], x['b'])

    def setter(x: dict[str, Any], y: Any) -> None:
        x['a'] = y[0]
        x['b'] = y[1]

    def deleter(x: dict[str, Any]) -> None:
        del x['a']
        del x['b']

    message = str(uuid4())
    d.set_deprecated_as_derived(
        'c',
        getter=getter,
        setter=setter,
        deleter=deleter,
        message=message,
    )
    return d, message


def test_deprecating_dict_with_derived_then_in_deprecated_keys():
    d, _ = deprecating_dict_with_derived()
    assert 'c' in d.deprecated_keys


def test_deprecating_dict_with_derived_then_get_deprecated():
    d, message = deprecating_dict_with_derived()
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == (1, 2)


def test_deprecating_dict_with_derived_then_set_nondeprecated():
    d, message = deprecating_dict_with_derived()
    d['a'] = 3
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == (3, 2)


def test_deprecating_dict_with_derived_then_set_deprecated():
    d, message = deprecating_dict_with_derived()
    with pytest.warns(FutureWarning, match=message):
        d['c'] = (5, 7)
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == (5, 7)
    assert d['a'] == 5
    assert d['b'] == 7


def test_deprecating_dict_with_derived_then_del_deprecated():
    d, message = deprecating_dict_with_derived()
    with pytest.warns(FutureWarning, match=message):
        assert 'c' in d
    with pytest.warns(FutureWarning, match=message):
        del d['c']
    assert 'c' not in d


def deprecating_dict_with_renamed() -> tuple[DeprecatingDict, str]:
    d = DeprecatingDict({'a': 1, 'b': 2})
    d.set_deprecated_from_rename(
        'c', 'a', version='v2.0', since_version='v1.6'
    )
    return d, 'is deprecated since'


def test_deprecating_dict_with_renamed_then_in_deprecated_keys():
    d, _ = deprecating_dict_with_renamed()
    assert 'c' in d.deprecated_keys


def test_deprecating_dict_with_renamed_then_get_deprecated():
    d, message = deprecating_dict_with_renamed()
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == 1


def test_deprecating_dict_with_renamed_then_set_nondeprecated():
    d, message = deprecating_dict_with_renamed()
    d['a'] = 3
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == 3


def test_deprecating_dict_with_renamed_then_set_deprecated():
    d, message = deprecating_dict_with_renamed()
    with pytest.warns(FutureWarning, match=message):
        d['c'] = 3
    with pytest.warns(FutureWarning, match=message):
        assert d['c'] == 3
    assert d['a'] == 3


def test_deprecating_dict_with_renamed_then_del_deprecated():
    d, message = deprecating_dict_with_renamed()
    with pytest.warns(FutureWarning, match=message):
        assert 'c' in d
    with pytest.warns(FutureWarning, match=message):
        del d['c']
    assert 'c' not in d
