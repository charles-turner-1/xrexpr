{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :special-members: __getitem__, __getattr__

.. Two departures from the stock autosummary class template, both deliberate.
..
.. ``:members:`` — the stock template emits a bare ``autoclass`` plus a *summary table*
.. of member names, so the page lists ``collect`` and ``explain`` without their
.. signatures or their docstrings. For a reference whose whole premise is that it is
.. generated from the source, that is the one thing it must not do.
..
.. ``:special-members:`` names two dunders rather than enabling all of them, because on
.. ``LazyProxy`` these two *are* the API: ``__getattr__`` is how every chained method is
.. recorded, and ``__getitem__`` is the call whose meaning depends on the base type.
.. Classes without them are unaffected.
..
.. Deliberately absent: ``:inherited-members:``. ``Explanation`` subclasses ``str`` and
.. ``frozendict`` is a ``Mapping``, so inheriting members would bury three lines of
.. package documentation under a hundred lines of stdlib.
