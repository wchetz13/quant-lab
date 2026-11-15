import numpy as np
import pytest
from scipy import stats

import var_backtest as vb


@pytest.fixture(scope="module")
def returns():
    return vb.simulate_garch(4000, seed=3)


def test_simulated_returns_have_fat_tails_and_clustering(returns):
    assert stats.kurtosis(returns) > 1.0
    assert np.corrcoef(returns[1:] ** 2, returns[:-1] ** 2)[0, 1] > 0.1


def test_garch_parameter_validation():
    with pytest.raises(ValueError):
        vb.simulate_garch(10, alpha=0.6, beta=0.5)


def test_gaussian_var_matches_closed_form_on_iid_normal():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 2000)
    var, es = vb.gaussian(r, 500, 0.01)
    assert np.mean(var) == pytest.approx(2.3263 * 0.01, rel=0.05)
    assert np.mean(es) == pytest.approx(2.6652 * 0.01, rel=0.05)


def test_es_exceeds_var_for_every_forecaster(returns):
    for fn in vb.FORECASTERS.values():
        var, es = fn(returns, 500, 0.01)
        assert np.all(es >= var - 1e-12)
        assert np.all(var > 0)


def test_forecasts_are_aligned_and_use_only_past_data():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 800)
    for fn in vb.FORECASTERS.values():
        var, _ = fn(r, 500, 0.05)
        assert len(var) == 300
        r2 = r.copy()
        r2[-1] = 10.0  # a shock on the last day cannot change any forecast for that day
        var2, _ = fn(r2, 500, 0.05)
        np.testing.assert_allclose(var, var2)


def test_ewma_recursion_and_no_lookahead():
    r = np.array([0.01, -0.02, 0.03, 0.0, 0.01])
    s = vb.ewma_volatility(r, lam=0.9, init_window=2)
    v0 = np.mean(r[:2] ** 2)
    v1 = 0.9 * v0 + 0.1 * r[0] ** 2
    assert s[0] == pytest.approx(np.sqrt(v0))
    assert s[1] == pytest.approx(np.sqrt(v1))


def test_kupiec_known_value():
    # 10 violations in 250 days at 1%: LR = -2 ln[(.99^240 .01^10) / ((.96)^240 (.04)^10)]
    hits = np.zeros(250, dtype=bool)
    hits[:10] = True
    res = vb.kupiec_pof(hits, 0.01)
    expected = -2 * (240 * np.log(0.99) + 10 * np.log(0.01) - 240 * np.log(0.96) - 10 * np.log(0.04))
    assert res.statistic == pytest.approx(expected)
    assert res.p_value == pytest.approx(stats.chi2.sf(expected, 1))
    assert res.reject()


def test_kupiec_accepts_exact_rate_and_handles_zero_hits():
    hits = np.zeros(1000, dtype=bool)
    hits[::100] = True
    assert vb.kupiec_pof(hits, 0.01).statistic == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(vb.kupiec_pof(np.zeros(250, dtype=bool), 0.01).p_value)


def test_independence_detects_clustering():
    spread = np.zeros(1000, dtype=bool)
    spread[::50] = True  # 20 hits, evenly spaced
    clustered = np.zeros(1000, dtype=bool)
    clustered[500:520] = True  # same count, back to back
    assert not vb.christoffersen_independence(spread).reject()
    assert vb.christoffersen_independence(clustered).reject()
    # Kupiec sees both as identical: it only counts hits
    assert vb.kupiec_pof(spread, 0.02).statistic == pytest.approx(vb.kupiec_pof(clustered, 0.02).statistic)


def test_conditional_coverage_is_sum_of_parts():
    rng = np.random.default_rng(5)
    hits = rng.random(1500) < 0.03
    cc = vb.conditional_coverage(hits, 0.01)
    parts = vb.kupiec_pof(hits, 0.01).statistic + vb.christoffersen_independence(hits).statistic
    assert cc.statistic == pytest.approx(parts)
    assert cc.p_value == pytest.approx(stats.chi2.sf(parts, 2))


@pytest.mark.parametrize("n, zone", [(0, "green"), (4, "green"), (5, "yellow"), (9, "yellow"), (10, "red")])
def test_basel_zones_match_published_table(n, zone):
    assert vb.basel_zone(n, 0.01, 250) == zone


def test_kupiec_size_is_close_to_nominal_when_model_is_correct():
    rng = np.random.default_rng(7)
    rejections = np.mean([vb.kupiec_pof(rng.random(1000) < 0.05, 0.05).reject() for _ in range(400)])
    assert 0.02 < rejections < 0.09


def test_ideal_forecaster_passes_and_gaussian_fails_on_fat_tails(returns):
    res = vb.backtest_all(returns, 500, 0.01)
    # the true conditional VaR would pass; EWMA-based models should beat plain rolling Gaussian
    assert res["Gaussian"]["cc_p"] < 0.05
    assert res["Filtered HS"]["kupiec_p"] > 0.05
    assert res["Gaussian"]["independence_p"] < res["Filtered HS"]["independence_p"]


def test_acerbi_szekely_zero_when_es_is_correct():
    rng = np.random.default_rng(11)
    r = rng.normal(size=400_000)
    alpha = 0.025
    var = np.full_like(r, -stats.norm.ppf(alpha))
    es = np.full_like(r, stats.norm.pdf(stats.norm.ppf(alpha)) / alpha)
    assert abs(vb.acerbi_szekely_z2(r, var, es, alpha)) < 0.05
    assert vb.acerbi_szekely_z2(r, var, es * 0.7, alpha) > 0.2


def test_input_validation():
    with pytest.raises(ValueError):
        vb.historical(np.zeros(100), window=500)
    with pytest.raises(ValueError):
        vb.historical(np.zeros(1000), alpha=0.7)
    with pytest.raises(ValueError):
        vb.violations(np.zeros(3), np.zeros(4))
