from typing import Union

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

        self._leads = self._initialized['lead'].data
        self._units = self._initialized['lead'].attrs['units']
        if 'member' in self._initialized.dims:
            self._nmember = self._initialized['member'].size
            self._members = self._initialized['member'].data
        else:
            self._nmember, self._members = None, None

    @property
    def initialized(self):
        return self._initialized

    @property
    def observation(self):
        return self._observation

    def _drop_members(self, members: list = None):
        if members is None:
            members = self._members[0]
        return self._initialized.drop_sel(member=members)
