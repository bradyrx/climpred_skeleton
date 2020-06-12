from .alignment import LeadAlignment
from .comparison import Comparison
from .time import TimeManager


# Might want multiple inheritance here, since PM doesn't need to do alignment.
class Scoring(LeadAlignment):
    def __init__(self, initialized, observation, comparison, alignment, metric):
        super().__init__(initialized, observation, alignment)
        self._metric = metric

        # Run comparison on the object. This uses the factory pattern from
        # Comparison to pull the appropriate comparison. We don't use inheritance
        # here so that we don't maintain the factory method and `broadcast` and
        # just handle the comparison upon instantiation.
        comparison_obj = Comparison(
            self._initialized, self._observation
        ).get_comparison(comparison)
        # Overwrites initialized and observation. Maybe a bad practice.
        self._initialized, self._observation = comparison_obj.broadcast()

    def score(self):
        pass


class HindcastScoring(Scoring):
    def __init__(self, initialized, observation, comparison, alignment, metric):
        super().__init__(initialized, observation, comparison, alignment, metric)

        # Same use of factory pattern. Not sure there's a way to do this in the classes
        # themselves without a recursion nightmare.
        alignment_obj = LeadAlignment(
            self._initialized, self._observation, alignment
        ).get_alignment(alignment)
        inits, verif_dates = alignment_obj._return_inits_and_verifs()
        self._scoring_inits = inits
        self._scoring_verifs = verif_dates


class PerfectModelScoring(TimeManager):
    """We might just need to inherit from TimeManager here since
    PM doesn't need any of the alignment attributes and methods."""

    pass
