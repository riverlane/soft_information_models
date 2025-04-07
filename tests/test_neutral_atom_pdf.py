import numpy as np
import pytest

from soft_information_models.neutral_atom_pdf import NeutralAtomPDF


class TestNeutralAtomPDF:

    @pytest.fixture(scope="class")
    def default_pdf(self) -> NeutralAtomPDF:
        return NeutralAtomPDF(seed=0)

    @pytest.mark.parametrize("t_meas", [1e-6, 5e-6, 1e-5, 1e-4])
    def test_meas_pdfs_are_normalised(self, t_meas):
        pdf = NeutralAtomPDF(t_measurement=t_meas)
        pdf_0_eval = pdf._0_state_pdf(pdf._domain)
        assert np.sum(pdf_0_eval) == pytest.approx(1, rel=1e-3)
        pdf_1_eval = pdf._1_state_pdf(pdf._domain)
        assert np.sum(pdf_1_eval) == pytest.approx(1, rel=1e-3)

    @pytest.mark.parametrize("t_meas, expected_err", [
        (1e-5,  0.001227),
        (5e-5,  0.001447),
        (10e-5, 0.001517),
    ])
    def test_readout_error_increases_for_long_meas_time(self, t_meas, expected_err):
        pdf = NeutralAtomPDF(t_measurement=t_meas)
        assert pdf.readout_error_probability == pytest.approx(expected_err, rel=1e-3)

    def test_sampling_ones_gives_expected_mean(self):
        pdf = NeutralAtomPDF(1e-3, 160e-6, 1e7, 1, 1, 1, seed=0)
        samples = pdf.sample(np.ones(100_000))
        rate = (pdf.eta * pdf.r_0 + pdf.r_bg) * pdf.t_m
        assert np.mean(samples) == pytest.approx(rate, rel=1e-2)

    def test_sampling_zeros_gives_expected_mean(self):
        pdf = NeutralAtomPDF(1e-3, 160e-6, 1e7, 1, 1, 1, seed=0)
        samples = pdf.sample(np.zeros(100_000))
        rate = pdf.r_bg * pdf.t_m
        assert np.mean(samples) == pytest.approx(rate, abs=1e-3)

    def test_sample_p_matches_predict_p(self, default_pdf):
        x = np.hstack((np.zeros(1000, dtype=int), np.ones(1000, dtype=int)))
        samples = default_pdf.sample(x)
        probability_eval = default_pdf.predict_p(samples)
        probability_samples = default_pdf.sample_p(x)
        assert np.mean(probability_eval - probability_samples) == pytest.approx(0)

    @pytest.mark.parametrize(
            "seed, expected_meas",
            [
                (0, (0, 92)),
                (1, (0, 116)),
            ]
    )
    def test_sampling_seeded_gives_fixed_results(self, seed, expected_meas):
        pdf = NeutralAtomPDF(seed=seed)
        samples = pdf.sample(np.array([0, 1]))
        np.testing.assert_array_equal(samples, np.array(expected_meas))

    @pytest.mark.parametrize("p_soft_target", [
        0.002,
        0.005,
        0.01,
        0.02,
        0.1,
    ])
    def test_from_p_clf_matches_actual(self, p_soft_target):
        pdf = NeutralAtomPDF.from_error_probability(p_soft_target)
        assert pytest.approx(p_soft_target, rel=0.1) == pdf.readout_error_probability
