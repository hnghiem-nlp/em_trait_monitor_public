"""The replication entry point must launch only the requested conditions."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'extensions/harmful_fraction'))
from pipeline import selected_fractions


def test_two_condition_repeat_does_not_schedule_six_runs():
    assert selected_fractions('50,100') == (50,100)
    assert selected_fractions('100, 50') == (50,100)
    assert selected_fractions('') == (25,0,5,10,50,75,100)


@pytest.mark.parametrize('value', ['50,50', '2,3', '50,', '105', 'no'])
def test_invalid_selection_is_rejected(value):
    with pytest.raises(ValueError):
        selected_fractions(value)


def test_seventy_five_selection():
    assert selected_fractions("75") == (75,)
