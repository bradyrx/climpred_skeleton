from typing import Union

import pkg_resources
import xarray as xr
import yaml

from .time import TimeManager


class LeadAlignment(TimeManager):
    """Base class for Alignment step of pipeline.

    This is the second step for Hindcast ensembles, and is not
    required by Perfect Model.
    """

    def __init__(
        self,
        initialized: Union[xr.Dataset, xr.DataArray],
        observation: Union[xr.Dataset, xr.DataArray],
        alignment: str,
        reference: Union[None, str, list, tuple],
    ):
        super().__init__(initialized, observation)
        self._alignment = alignment

        if isinstance(reference, str):
            reference = [reference]
        elif reference is None:
            reference = []
        self._reference = reference
        if 'persistence' in reference:
            self._persistence = True
        else:
            self._persistence = False

    def get_alignment(self) -> 'LeadAlignment':
        try:
            return alignment_dict[self._alignment](
                self._initialized, self._observation, self._alignment, self._reference
            )
        except KeyError:
            raise ValueError(
                f'{self._alignment} not valid keyword from '
                f'{list(alignment_dict.keys())}'
            )

    def _return_inits_and_verifs(self):
        pass

    def _construct_init_lead_matrix(self):
        """Constructs the init-lead matrix to figure out which inits and verif dates
        to use at a given lead for a given alignment strategy."""
        n, freq = self._get_all_lead_cftime_shift_args()
        init_lead_matrix = xr.concat(
            [
                xr.DataArray(
                    self._shift_hindcast_inits(int(n), freq),
                    dims=['init'],
                    coords=[self._all_inits],
                )
                for n in n
            ],
            'lead',
        )
        return init_lead_matrix

    def _shift_hindcast_inits(self, n: int, freq: str):
        """Helper function to shift the initialized inits by a specific n and freq. Used
        in constructing the init/lead matrix."""
        time_index = self._all_inits.to_index()
        return time_index.shift(n, freq)


class SameInitializations(LeadAlignment):
    """Class for `same_inits` keyword."""

    def __init__(self, initialized, observation, alignment, reference=None):
        """Reference will alter how initializations are selected."""
        super().__init__(initialized, observation, alignment, reference)

    def _return_inits_and_verifs(self):
        (
            n,
            freq,
        ) = self._get_all_lead_cftime_shift_args()  # n and freq could be attributes.
        init_lead_matrix = self._construct_init_lead_matrix()

        # If a persistence forecast is desired, need a union between the
        # initializations and verifs so the same are used.
        if self._persistence:
            union_with_verifs = self._all_inits.isin(self._all_verifs)
            init_lead_matrix = init_lead_matrix.where(union_with_verifs, drop=True)

        verifies_at_all_leads = init_lead_matrix.isin(self._all_verifs).all('lead')
        valid_inits = init_lead_matrix['init']
        inits = valid_inits.where(verifies_at_all_leads, drop=True)
        inits = {lead: inits for lead in self._leads}
        verif_dates = {
            lead: self._shift_cftime_index(inits[lead], 'init', int(n), freq)
            for (lead, n) in zip(self._leads, n)
        }
        return inits, verif_dates


alignment_yaml_file = pkg_resources.resource_filename(
    'climpred_skeleton', 'alignment.yaml'
)
with open(alignment_yaml_file) as f:
    """Pull aliases from alignment YAML into dictionary to guide factory."""
    metadata = yaml.safe_load(f)
    alignment_dict = {}
    for k, v in metadata.items():
        for keyword in v['keywords']:
            alignment_dict.update({keyword: eval(k)})
