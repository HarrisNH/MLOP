# content of test_exceptiongroup.py
import pytest


def f():
    raise ExceptionGroup(
        "Group message",
        [RuntimeError()],
    )


def test_exception_in_group():
    with pytest.raises(ExceptionGroup) as excinfo:
        f()
    assert excinfo.group_contains(RuntimeError)
    assert not excinfo.group_contains(TypeError)
