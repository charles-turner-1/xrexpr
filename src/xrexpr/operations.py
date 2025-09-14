"""
This is just going to hold a set of all the operations in xarray that we might
want to optimise. Honestly seems like a bit of a janky idea but maybe it'll
gradually lead to a better understanding of how to handle the problem
"""

AGGREGATIONS = {
    "reduce",
    "count",
    "all",
    "any",
    "max",
    "min",
    "mean",
    "prod",
    "sum",
    "std",
    "var",
    "median",
    "cumsum",
    "cumprod",
}

SELECTIONS = {
    "sel",
    "isel",
}
