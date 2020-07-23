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

        self._leads = self.initialized['lead'].data
        self._units = self.initialized['lead'].attrs['units']
        if 'member' in self.initialized.dims:
            self._nmember = self.initialized['member'].size
            self._members = self.initialized['member'].data
        else:
            self._nmember, self._members = None, None

    @property
    def initialized(self):
        return self._initialized

    @initialized.setter
    def initialized(self, value):
        self._initialized = value

    @property
    def leads(self):
        return self._leads

    @property
    def members(self):
        return self._members

    @property
    def nmember(self):
        return self._nmember

    @property
    def observation(self):
        return self._observation

    @observation.setter
    def observation(self, value):
        self._observation = value

    @property
    def units(self):
        return self._units

    def _drop_members(self, members: list = None):
        if members is None:
            members = self.members[0]
        return self.initialized.drop_sel(member=members)
