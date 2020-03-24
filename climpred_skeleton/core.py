import warnings
from typing import Union

import cftime
import pandas as pd
import xarray as xr


class Verification:
    """Absolute base class. Begins the pipeline of verifying a forecast.

    Holds a prediction that is being verified against observations. Stores
    helpful base attributes (`nmember`, etc.) and runs checks on input
    data.
    """

    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
    ):
        # E.g., check that inputs are xarray objects, convert to dataset.

        # Make sure that we don't overwrite the original arrays.
        self._initialized = initialized.copy()
        self._observation = observation.copy()

        # Convert `init` and `time` indices to CFTimeIndex.
        self._convert_to_cftime_index()
        if 'member' in self._initialized.dims:
            self._nmember = self._initialized['member'].size
            self._members = self._initialized['member'].data
        else:
            self._nmember, self._members = None, None
        self._all_verifs = self._observation['time'].data
        self._all_inits = self._initialized['init'].data

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

    def _drop_members(self, members: list = None):
        if members is None:
            members = self._members[0]
        return self._initialized.drop_sel(member=members)
