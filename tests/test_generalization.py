import numpy as np
import pytest

from ygorl.train.generalization import card_ids_per_row, exclude_card_groups, reserved_action_rows, source_group


@pytest.mark.parametrize("field,column", [("cards", 0), ("actions", 2), ("actions", 3), ("events", 2), ("events", 3)])
def test_exclusion_covers_all_identity_channels_and_whole_hands(field, column):
    obs = {
        "cards": np.zeros((3, 2, 23), int),
        "actions": np.zeros((3, 2, 10), int),
        "events": np.zeros((3, 2, 20), int),
    }
    obs[field][0, 1, column] = 42
    meta = [{"deck": "a", "hand_index": i, "variant": str(k)} for k, i in enumerate([0, 0, 1])]
    keep, excluded = exclude_card_groups(obs, meta, {42})
    assert keep.tolist() == [False, False, True]
    assert excluded == {("opening", "a", 0)}
    assert 42 not in card_ids_per_row({k: v[keep] for k, v in obs.items()})


def test_missing_provenance_is_not_silently_row_split():
    with pytest.raises(ValueError, match="source"):
        source_group({"step": 1})


def test_novel_action_slice_uses_legal_candidates_and_card_references():
    obs = {
        "cards": np.zeros((4, 2, 23), int),
        "actions": np.zeros((4, 2, 10), int),
        "action_mask": np.ones((4, 2), bool),
    }
    obs["cards"][:, 0, 0] = 42
    obs["actions"][0, 0, 1] = 1
    obs["actions"][1, 0, 3] = 42
    obs["actions"][2, 0, 2] = 42
    obs["action_mask"][2, 0] = False
    assert reserved_action_rows(obs, {42}).tolist() == [True, True, False, False]
