from .alignment import LeadAlignment
from .comparison import Comparison


# Might want multiple inheritance here, since PM doesn't need to do alignment.
class Scoring(LeadAlignment):
    def __init__(self, initialized, observation, comparison, alignment, metric):
        super().__init__(initialized, observation, comparison, alignment)
        self._metric = metric

        # Run comparison on the object. This uses the factory pattern from
        # Comparison to pull the appropriate comparison. We don't use inheritance
        # here so that we don't maintain the factory method and `broadcast` and
        # just handle the comparison upon instantiation.
        comparison_obj = Comparison(
            self._initialized, self._observation
        ).get_comparison(comparison)
        self._initialized, self._observation = comparison_obj.broadcast()

    def score(self):
        pass


class HindcastScoring(Scoring):
    pass
