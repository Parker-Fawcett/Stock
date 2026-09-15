from deflated_sharpe import FORMAL_PROPOSAL_IDS


def test_historical_dsr_proposal_set_is_frozen_to_p1_through_p12():
    assert FORMAL_PROPOSAL_IDS == tuple(f"P{number}" for number in range(1, 13))
