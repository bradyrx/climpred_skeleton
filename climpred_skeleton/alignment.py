from typing import Union

import xarray as xr

from .comparison import Comparison
from .time import TimeManager


class LeadAlignment(TimeManager):
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
        self._all_verifs = self._observation['time'].data
        self._all_inits = self._initialized['init'].data

        # Run comparison on the object
        # (this might go to Scoring, since PM doesn't go through here)
        comparison_obj = Comparison(
            self._initialized, self._observation
        ).get_comparison(comparison)
        self._comparison_method = comparison_obj._comparison_method
        self._initialized, self._observation = comparison_obj.broadcast()

    def _construct_init_lead_matrix(self):
        """Constructs the init-lead matrix to figure out which inits and verif dates
        to use at a given lead for a given alignment strategy."""
        n, freq = self._get_all_lead_cftime_shift_args()
        init_lead_matrix = xr.concat(
            [
                xr.DataArray(
                    self._shift_hindcast_inits(n, freq),
                    dims=['init'],
                    coords=[self._all_inits],
                )
                for n in n
            ],
            'lead',
        )
        return init_lead_matrix

    def _shift_hindcast_inits(self, n: int, freq: str):
        """Shifts hindcast inits `n` timesteps forward at `freq`."""
        time_index = self._initialized['init'].to_index()
        return time_index.shift(n, freq)


class SameInitializations(LeadAlignment):
    """Class for `same_inits` keyword."""

    def __init__(self, initialized, observation, comparison, alignment):
        super().__init__(initialized, observation, comparison, alignment)

    def align(self):
        pass
