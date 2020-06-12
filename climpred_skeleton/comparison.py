from typing import Union

import pkg_resources
import xarray as xr
import yaml

from .time import TimeManager


class Comparison(TimeManager):
    """First step in pipeline. Broadcast your initialized and observations
    in a certain manner.

    Base Comparison class hints at which attributes are expected for each
    sub comparison class. It also stores the factory for internal use, to
    return the appropriate subfunction.
    """

    _comparison_method: str
    _hindcast_comparison: bool
    _probabilistic_comparison: bool

    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
    ):
        super().__init__(initialized, observation)

    def broadcast(self):
        pass

    def get_comparison(self, method) -> 'Comparison':
        try:
            return comparison_dict[method](self._initialized, self._observation)
        except KeyError:
            raise ValueError(
                f'{method} not valid keyword from {list(comparison_dict.keys())}'
            )


class EnsembleToObservation(Comparison):
    """Example subclass for Comparison.

    Stores useful information (e.g. whether this is a probabilistic comparison)
    and houses main `broadcast` function, returning the modified initialized
    and observation objects.
    """

    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
    ):
        super().__init__(initialized, observation)
        self._comparison_method = 'e2o'
        self._hindcast_comparison = True
        self._probabilistic_comparison = False

    def broadcast(self) -> Union[xr.Dataset, xr.Dataset]:
        if 'member' in self._initialized.dims:
            initialized = self._initialized.mean('member')
        else:
            initialized = self._initialized
        return initialized, self._observation


class MemberToObservation(Comparison):
    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
    ):
        super().__init__(initialized, observation)
        self._comparison_method = 'm2o'
        self._hindcast_comparison = True
        self._probabilistic_comparison = True

    def broadcast(self) -> Union[xr.Dataset, xr.Dataset]:
        observation = self._observation.expand_dims({'member': self._nmember})
        observation['member'] = self._members
        return self._initialized, observation


class MemberToControl(Comparison):
    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
    ):
        super().__init__(initialized, observation)
        self._comparison_method = 'm2c'
        self._hindcast_comparison = False
        self._probabilistic_comparison = True

    def broadcast(self, control_member: list = None) -> Union[xr.Dataset, xr.Dataset]:
        if control_member is None:
            control_member = self._members[0]
        observation = self._initialized.isel(member=control_member).squeeze()
        # drop the member being considered as the control.
        initialized = self._drop_members(members_to_remove=control_member)
        return initialized, observation


comparison_yaml_file = pkg_resources.resource_filename(
    'climpred_skeleton', 'comparison.yaml'
)
with open(comparison_yaml_file) as f:
    """Pull aliases from comparison YAML into dictionary to guide factory."""
    metadata = yaml.safe_load(f)
    comparison_dict = {}
    for k, v in metadata.items():
        for keyword in v['keywords']:
            comparison_dict.update({keyword: eval(k)})
