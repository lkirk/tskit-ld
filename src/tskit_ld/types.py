from typing import Optional, Sequence, SupportsIndex

import numpy as np
import numpy.typing as npt
import polars.type_aliases

NPShapeLike = Optional[Sequence | SupportsIndex]  # (as defined in the _typing module)
NPInt64Array = npt.NDArray[np.int64]
NPUInt32Array = npt.NDArray[np.uint32]
NPInt32Array = npt.NDArray[np.int32]

# Type alias for polars expressions
IntoExpr = polars.type_aliases.IntoExpr
