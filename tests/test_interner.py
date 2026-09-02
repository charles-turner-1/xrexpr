import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from xrexpr.interner import Interner


@pytest.fixture(autouse=True)
def _clear_interner():
    """Reset the interner singleton before each test — it now persists process-wide."""
    Interner()._clear()
    yield


class TestInterner:
    def test_interning(self):
        interner = Interner[str]()
        handle1 = interner("foo")
        handle2 = interner("bar")
        handle3 = interner("foo")

        assert handle1 == handle3
        assert handle1 != handle2
        assert interner[handle1] == "foo"
        assert interner[handle2] == "bar"
        assert len(interner) == 2

    def test_interner_is_singleton(self):
        # The interner is a process-wide singleton: constructing it again returns the same
        # populated instance rather than a fresh table.
        i1 = Interner[str]()
        i2 = Interner[str]()

        assert i1 is i2
        handle = i1("foo")
        assert i2("foo") == handle
        assert len(i2) == 1

    def test_interner_dedup(self):
        interner = Interner[str]()
        handle1 = interner("foo")
        handle2 = interner("foo")
        assert handle1 == handle2
        assert len(interner) == 1

    def test_interner_contains(self):
        interner = Interner[str]()
        interner("foo")
        assert "foo" in interner
        assert "bar" not in interner

    def test_interner_iter(self):

        interner = Interner[str]()
        interner("foo")
        interner("bar")
        item_list = list(interner)
        item_tup = tuple(interner)

        assert item_list == ["foo", "bar"]
        assert item_tup == ("foo", "bar")

    def test_interner_round_trip(self):
        interner = Interner[str]()
        items = ["foo", "bar", "baz"]
        handles = [interner(item) for item in items]
        round_trip_items = [interner[handle] for handle in handles]
        assert round_trip_items == items

    def test_internet_rejects_unhashable(self):
        interner = Interner()
        with pytest.raises(TypeError):
            interner(["foo", "bar"])


# Now lets do a bunch of property tests for the interener.


@given(st.lists(st.text(), min_size=1, max_size=100))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_interner_property(items):
    # Hypothesis re-runs the body per example; the singleton persists, so clear per example.
    interner = Interner[str]()
    interner._clear()
    interner = Interner[str]()
    handles = [interner(item) for item in items]
    round_trip_items = [interner[handle] for handle in handles]
    assert round_trip_items == items


@given(st.lists(st.text(), min_size=1, max_size=100))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_interner_dedups(items):
    # Hypothesis re-runs the body per example; the singleton persists, so clear per example.
    Interner[str]()._clear()
    interner = Interner[str]()
    handles = [interner(item) for item in items]

    assert len(interner) == len(set(items))
    # equal inputs -> equal handles, distinct inputs -> distinct handles
    assert all(
        (items[i] == items[j]) == (handles[i] == handles[j])
        for i in range(len(items))
        for j in range(len(items))
    )
