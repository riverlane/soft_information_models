import numpy as np
import pytest

from soft_information_models.superconducting_pdf import SuperconductingPDF


@pytest.mark.parametrize("snr", [0.5, 1, 2])
def test_zero_state_meas_pdf_is_normalised(snr):
    step = 0.01
    domain = np.arange(-10, 10, step)
    pdf = SuperconductingPDF(snr, 0.1)
    pdf_eval = pdf._0_state_pdf(domain)
    assert np.sum(pdf_eval) * step == pytest.approx(1, rel=1e-5)


@pytest.mark.parametrize("snr", [0.5, 1, 10])
@pytest.mark.parametrize("beta", [0.01, 0.1, 1, 2])
def test_one_state_meas_pdf_is_normalised(snr, beta):
    step = 0.001
    domain = np.arange(-10, 10, step)
    pdf = SuperconductingPDF(snr, beta)
    pdf_eval = pdf._1_state_pdf(domain)
    assert np.sum(pdf_eval) * step == pytest.approx(1, rel=1e-5)


class TestSuperconductingPDF:

    def test_readout_error_is_zero_with_no_overlap(self):
        pdf = SuperconductingPDF(100, 0)
        assert pdf.readout_error_probability == pytest.approx(0)

    def test_readout_error_is_half_with_max_overlap(self):
        pdf = SuperconductingPDF(1e-6, 0)
        assert pdf.readout_error_probability == pytest.approx(0.5, rel=1e-3)

    def test_readout_error_is_half_with_amp_damping(self):
        pdf = SuperconductingPDF(10, 100)
        assert pdf.readout_error_probability == pytest.approx(0.5, rel=2e-2)

    def test_sampling_zeros_gives_expected_mean(self):
        x = np.zeros(100_000)
        pdf = SuperconductingPDF(10, 0, seed=0)
        assert np.mean(pdf.sample(x)) == pytest.approx(1, rel=1e-3)

    @pytest.mark.parametrize("snr", [0.5, 1, 10])
    def test_sampling_zeros_shows_effect_of_snr_on_stdev(self, snr):
        x = np.zeros(100_000)
        pdf = SuperconductingPDF(snr, 0, seed=0)
        expected_std = np.sqrt(2 / snr)
        assert np.std(pdf.sample(x)) == pytest.approx(expected_std, rel=0.01)

    @pytest.mark.parametrize(
            "beta, expected_mean",
            [
                (0, -1.0),
                (1, -0.2655),
                (10, 0.8),
            ]
    )
    def test_sampling_ones_shows_effect_of_amplitude_damping(self, beta, expected_mean):
        x = np.ones(100_000)
        pdf = SuperconductingPDF(10, beta, seed=0)
        assert np.mean(pdf.sample(x)) == pytest.approx(expected_mean, rel=1e-3)

    def test_predict_p_gives_correct_result_without_amp_damp(self):
        pdf = SuperconductingPDF(100, 0, seed=0)
        p_x = pdf.predict_p(pdf.sample(np.zeros(10_000)))
        p_y = pdf.predict_p(pdf.sample(np.ones(10_000)))
        assert np.mean(p_x[:, 0]) == pytest.approx(1)
        assert np.mean(p_x[:, 1]) == pytest.approx(0)
        assert np.mean(p_y[:, 1]) == pytest.approx(1)
        assert np.mean(p_y[:, 0]) == pytest.approx(0)

    def test_sample_p_matches_predict_p(self):
        pdf = SuperconductingPDF(10, 0, seed=0)
        x = np.hstack((np.zeros(1000), np.ones(1000)))
        samples = pdf.sample(x)
        probability_eval = pdf.predict_p(samples)
        probability_samples = pdf.sample_p(x)
        assert np.mean(probability_eval - probability_samples) == pytest.approx(0)

    @pytest.mark.parametrize(
            "seed, expected_meas",
            [
                (0, (1.495945, -1.250948)),
                (1, (1.041902, 2.369411)),
            ]
    )
    def test_sampling_seeded_gives_fixed_results(self, seed, expected_meas):
        pdf = SuperconductingPDF(1, 1, seed=seed)
        samples = pdf.sample(np.array([0, 1]))
        np.testing.assert_array_almost_equal(samples, np.array(expected_meas))

    @pytest.mark.parametrize("p_clf", [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3])
    def test_pdf_from_p_clf_has_expected_error_prob(self, p_clf):
        pdf = SuperconductingPDF.from_error_probability(p_clf)
        assert pdf.readout_error_probability == pytest.approx(p_clf, rel=1e-3)
