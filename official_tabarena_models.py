"""Named checkpoint variants for TabArena's existing TabICLv2 model adapter."""

from tabarena.models.tabicl.model import TabICLv2Model


class OriginalTabICLv2(TabICLv2Model):
    ag_name = "FG-TabICLv2-Original"
    _supported_problem_types = ["binary", "multiclass"]


class ControlTabICLv2(TabICLv2Model):
    ag_name = "FG-TabICLv2-FT0"
    _supported_problem_types = ["binary", "multiclass"]


class PenalizedTabICLv2(TabICLv2Model):
    ag_name = "FG-TabICLv2-FT05"
    _supported_problem_types = ["binary", "multiclass"]


# Inherit ag_key = "TA-TABICLv2" so TabArena applies its TabICLv2 dataset constraints.
MODEL_CLASSES = (OriginalTabICLv2, ControlTabICLv2, PenalizedTabICLv2)
