from typing import Final, final

@final
class AllDims:
    def __new__(cls) -> AllDims: ...

ALL_DIMS: Final[AllDims]
