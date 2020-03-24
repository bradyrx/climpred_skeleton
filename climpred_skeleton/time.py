import warnings
from typing import Union

import cftime
import pandas as pd
import xarray as xr

from climpred_skeleton.core import Verification


class TimeManager(Verification):
    def __init__(self, initialized, observation):
        super().__init__(initialized, observation)
        # Convert `init` and `time` indices to CFTimeIndex.
        self._convert_to_cftime_index()
        self._units = self._initialized['lead'].attrs['units']

    def _convert_to_cftime_index(self, calendar: str = 'DatetimeProlepticGregorian'):
        """Converts time indices for the prediction and observations to a
        CFTimeIndex."""

        def _return_converted_time_index(
            time_index: Union[
                xr.CFTimeIndex, pd.DatetimeIndex, pd.Float64Index, pd.Int64Index
            ]
        ):
            assume_annual = False

            if not isinstance(time_index, xr.CFTimeIndex):
                if isinstance(time_index, pd.DatetimeIndex):
                    time_strings = [str(t) for t in time_index]
                    split_dates = [d.split(' ')[0].split('-') for d in time_strings]

                # If Float64Index or Int64Index, assume annual and convert accordingly.
                elif isinstance(time_index, pd.Float64Index) | isinstance(
                    time_index, pd.Int64Index
                ):
                    warnings.warn(
                        'Assuming annual resolution due to numeric inits. '
                        'Change init to a datetime if it is another resolution.'
                    )
                    dates = [str(int(t)) + '-01-01' for t in time_index]
                    split_dates = [d.split('-') for d in dates]
                    assume_annual = True

                else:
                    raise ValueError(
                        f'The incoming time index must be pd.Float64Index, '
                        'pd.Int64Index, xr.CFTimeIndex or '
                        'pd.DatetimeIndex.'
                    )

                cftime_dates = [
                    getattr(cftime, calendar)(int(y), int(m), int(d))
                    for (y, m, d) in split_dates
                ]
                time_index = xr.CFTimeIndex(cftime_dates)
            return time_index, assume_annual

        self._initialized['init'], assume_annual = _return_converted_time_index(
            self._initialized['init'].to_index()
        )
        if assume_annual:
            self._initialized['lead'].attrs['units'] = 'years'
        self._observation['time'], _ = _return_converted_time_index(
            self._observation['time'].to_index()
        )

    def _get_all_lead_cftime_shift_args(self):
        """Returns a tuple of all lead shifts and the frequency string."""
        n_freq_tuples = [self._get_lead_cftime_shift_args(l) for l in self._leads]
        n, freq = list(zip(*n_freq_tuples))
        return n, freq[0]

    def _get_lead_cftime_shift_args(self, lead: int):
        """Returns a tuple of the number of units to shift and the frequency string."""
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
