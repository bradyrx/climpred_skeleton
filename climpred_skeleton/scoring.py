import xarray as xr
from xskillscore import pearson_r

from .alignment import LeadAlignment
from .comparison import Comparison
from .time import TimeManager


class Scoring(TimeManager):
    def __init__(self, initialized, observation, comparison):
        super().__init__(initialized, observation)

        # Run comparison on the object. This uses the factory pattern from
        # Comparison to pull the appropriate comparison. We don't use inheritance
        # here so that we don't maintain the factory method and `broadcast` and
        # just handle the comparison upon instantiation.
        comparison_obj = Comparison(self.initialized, self.observation).get_comparison(
            comparison
        )
        # Overwrites initialized and observation. Maybe a bad practice.
        # Should be using a setter, which I tried in `core.py` but it says
        # I can't set the attribute here if I do that.
        self._initialized, self._observation = comparison_obj.broadcast()

    def score(self):
        pass


class HindcastScoring(Scoring):
    def __init__(self, initialized, observation, comparison, alignment, reference=None):
        """
        Reference is set here so that the alignment system can select
        inits/verifs appropriately. It may differ based on the reference.
        If `persistence` and the verif dates are shorter than the dates of
        the initialized forecast, they get trimmed even if they verify in the
        same window so that the initialized forecast and verifications use the
        same inits.
        """
        super().__init__(initialized, observation, comparison)

        if isinstance(reference, str):
            reference = [reference]
        elif reference is None:
            reference = []
        self._reference = reference

        # Same use of factory pattern. Not sure there's a way to do this in the classes
        # themselves without a recursion nightmare.
        alignment_obj = LeadAlignment(
            self.initialized, self.observation, alignment, reference
        ).get_alignment()
        inits, verif_dates = alignment_obj._return_inits_and_verifs()

        # These are at most a union with all_inits and all_verifs, but are likely
        # a subset.
        self._scoring_inits = inits
        self._scoring_verifs = verif_dates

    @property
    def reference(self):
        return self._reference

    @property
    def scoring_inits(self):
        return self._scoring_inits

    @property
    def scoring_verifs(self):
        return self._scoring_verifs

    def _apply_metric_at_given_lead(self, lead, fct_type=None):
        """Returns score for a given forecast type at a given lead."""
        FORECAST_TYPE = {
            'init': self._initialized_forecast,
            'persistence': self._persistence_forecast,
        }

        a, b = FORECAST_TYPE[fct_type](lead)
        a['time'] = b['time']
        # Just an example metric here.
        return pearson_r(a, b, 'time')

    def _initialized_forecast(self, lead):
        # Use `.where()` instead of `.sel()` to account for resampled inits when
        # bootstrapping.
        a = (
            self.initialized.sel(lead=lead)
            .where(self.all_inits.isin(self.scoring_inits[lead]), drop=True)
            .drop_vars('lead')
            .rename({'init': 'time'})
        )
        b = self.observation.sel(time=self.scoring_verifs[lead])
        return a, b

    def _persistence_forecast(self, lead):
        # Use `.where()` instead of `.sel()` to account for resampled inits when
        # bootstrapping.
        a = self.observation.where(
            self.all_verifs.isin(self.scoring_inits[lead]), drop=True
        )
        b = self.observation.sel(time=self.scoring_verifs[lead])
        return a, b

    def score(self):
        # NOTE: Alignment applied at lower level and accounts for reference, so score
        # computation will automatically apply this.
        if self.reference is None:
            fct_list = ['init']
        else:
            fct_list = ['init'] + self.reference

        result = xr.Dataset()
        for fct_type in fct_list:
            metric_over_leads = [
                self._apply_metric_at_given_lead(lead, fct_type) for lead in self.leads
            ]
            current_forecast = xr.concat(
                metric_over_leads, dim='lead', coords='minimal', compat='override'
            )
            result = xr.concat(
                [result, current_forecast],
                dim='fct_type',
                coords='minimal',
                compat='override',
            )
        result['lead'] = self.leads
        result = result.assign_coords(fct_type=fct_list)
        return result


class PerfectModelScoring(Scoring):
    """We might just need to inherit from TimeManager here since
    PM doesn't need any of the alignment attributes and methods."""

    def __init__(self, initialized, observation, comparison):
        super().__init__(initialized, observation, comparison)

    def score(self):
        return pearson_r(self.initialized, self.observation, dim=['init', 'member'])
