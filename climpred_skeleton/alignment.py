from typing import Union

import xarray as xr

from .comparison import Comparison
from .core import Verification


class LeadAlignment(Verification):
    """Base class for Alignment step of pipeline.

    This is the second step for Hindcast ensembles, and is not
    required by Perfect Model.

    It doesn't inherit from ``Comparison`` because we don't want to retain
    the `.broadcast()` and `.factory()` functions.

    At initialization, it retrieves the appropriate comparison subclass and overwrites
    the `initialized` and `observation` attribute with their dbroadcasted versions.
    """

    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
        comparison: str,
        alignment: str,
    ):
        super().__init__(initialized, observation)
        self._alignment = alignment
        comparison_obj = Comparison(
            self._initialized, self._observation
        ).get_comparison(comparison)
        self._comparison_method = comparison_obj._comparison_method
        self._initialized, self._observation = comparison_obj.broadcast()

    def align(self):
        pass
