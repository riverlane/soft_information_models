import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar  # type: ignore

from soft_information_models.abstract_pdf import AbstractPDF


def _p_clf_gaussian(snr: float) -> float:
    """
    Helper function to calculate classification error probability assuming no
    amplitude damping noise is present.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio of the measurement.
    """
    return 0.5 * math.erfc(np.sqrt(snr) / 2)


class SuperconductingPDF(AbstractPDF):
    """
    Probability density function class that describes a superconducting qubit
    measurement in the presence of finite SNR and amplitude damping noise. The means
    of the 0- and 1- measurement responses in the absence of amplitude damping are
    +1 [a.u.] and -1 [a.u.] respectively.

    Parameters
    ----------
    snr : float
        Signal-to-noise ratio of the 0- state measurement, related to the standard
        deviation of the Gaussian via (SNR = 2 / std ^ 2) and to the characteristic
        device timescales via (SNR = 2 * t_m / t_f) for measurement time t_m and
        fluctuation time t_f. The factor of two comes from |mu_0 - mu_1| = 2.
    beta : float
        Amplitude damping parameter, given by beta = t_m / T1 for a measurement
        time t_m and qubit T1 time.
    num_sampling_intervals : int, optional
        Number of sample points when drawing randomly generated soft measurements.
        By default, 1000.
    seed : Optional[int], optional
        Random seed used in sampling. By default, None.
    """

    def __init__(
        self,
        snr: float,
        beta: float,
        num_sampling_intervals: int = 10_000,
        seed: Optional[int] = None,
    ):
        self.snr = snr
        self.beta = beta
        std = np.sqrt(2 / snr)
        self._domain_lims = (-1 - 4 * std, 1 + 4 * std)
        domain = np.linspace(*self._domain_lims, num_sampling_intervals)
        super().__init__(domain=domain, dtype=np.float64, seed=seed)

    def _0_state_pdf(self, z_soft: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Gaussian probability density function for a 0-state measurement, with
        a mean at measurement response +1 [a.u.].

        Parameters
        ----------
        z_soft : NDArray[np.float64]
            Points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            Values of the PDF as evaluated with given values z_soft.
        """
        amplitude = np.sqrt(self.snr / (4 * np.pi))
        return amplitude * np.exp(- 0.25 * self.snr * (z_soft - 1) ** 2)

    def _1_state_pdf(self, z_soft: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Amplitude damping probability density function for a 1-state measurement, where
        the response decays towards the 0-state when the amplitude damping damping
        parameter beta is increased. The means of the 0 and 1 measurement responses are
        +1 [a.u.] and -1 [a.u.] respectively.

        Parameters
        ----------
        z_soft : NDArray[np.float64]
            Points at which to evaluate the PDF.

        Returns
        -------
        NDArray[np.float64]
            Values of the PDF as evaluated with given values z_soft.
        """
        exp_part_1 = np.sqrt(self.snr / (4 * np.pi)) * np.exp(
            - 0.25 * self.snr * (z_soft + 1) ** 2 - self.beta
        )
        exp_part_2 = 0.25 * self.beta * np.exp(
            (self.beta ** 2) / (4 * self.snr) + 0.5 * self.beta * (z_soft - 1)
        )

        def _erf_component(z_i: float) -> float:

            arg_1 = np.sqrt(self.beta ** 2 / (4 * self.snr)) + \
                (z_i - 1) * np.sqrt(self.snr / 4)
            arg_2 = np.sqrt(self.beta ** 2 / (4 * self.snr)) + \
                (z_i + 1) * np.sqrt(self.snr / 4)

            return math.erfc(arg_2) - math.erfc(arg_1)

        erf_part = np.fromiter(map(_erf_component, z_soft), dtype=float)
        return exp_part_1 - exp_part_2 * erf_part

    @classmethod
    def from_error_probability(
        cls,
        clf_p: float,
        num_sampling_intervals: int = 1000,
        seed: Optional[int] = None,
    ):
        """
        Construct an instance of `SuperconductingPDF` class with beta=0
        and the SNR calculated based on the classification error probability.

        Parameters
        ----------
        clf_p : float
            Classification error probability, 0 < p_clf < 0.5.
        num_sampling_intervals : int, optional
            Number of sample points when drawing randomly generated soft measurements.
            By default, 1000.
        seed : Optional[int], optional
            Random seed used in sampling. By default, None.

        Returns
        -------
        SuperconductingPDF
            PDF class with classification error probability equal to p_clf.
        """

        def _obj_func(snr: float) -> float:
            return clf_p - _p_clf_gaussian(snr)

        result = root_scalar(_obj_func, bracket=[0.01, 100])
        return cls(result.root, 0, num_sampling_intervals, seed)
