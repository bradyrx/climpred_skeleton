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
        self._all_verifs = self._observation['time'].data
        self._all_inits = self._initialized['init'].data

        # Run comparison on the object
        # (this might go to Scoring, since PM doesn't go through here)
        comparison_obj = Comparison(
            self._initialized, self._observation
        ).get_comparison(comparison)
        self._comparison_method = comparison_obj._comparison_method
        self._initialized, self._observation = comparison_obj.broadcast()

    def align(self):
        pass

    def _construct_init_lead_matrix(self):
        n, freq = self._get_multiple_lead_cftime_shift_args()
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

    def _get_lead_cftime_shift_args(self, lead):
        lead = int(lead)

        d = {
            # Currently assumes yearly aligns with year start.
            'years': (lead, 'YS'),
            'seasons': (lead * 3, 'MS'),
            # Currently assumes monthly aligns with month start.
            'months': (lead, 'MS'),
            'weeks': (lead * 7, 'D'),
            'pentads': (lead * 5, 'D'),
            'days': (lead, 'D'),
        }

        n, freq = d[self._units]
        return n, freq

    def _get_multiple_lead_cftime_shift_args(self):
        n_freq_tuples = [self._get_lead_cftime_shift_args(l) for l in self._leads]
        n, freq = list(zip(*n_freq_tuples))
        return n, freq[0]

    def _shift_hindcast_inits(self, n, freq):
        time_index = self._initialized['init'].to_index()
        return time_index.shift(n, freq)


class SameInitializations(LeadAlignment):
    """Class for `same_inits` keyword."""

    def __init__(self, initialized, observation, comparison, alignment):
        super().__init__(initialized, observation, comparison, alignment)

    def align(self):
        pass
