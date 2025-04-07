from enum import Enum
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar  # type: ignore
from scipy.special import gammaln  # type: ignore

from soft_information_models.abstract_pdf import AbstractPDF


class MeasParameters(Enum):
    """Hard-coded default parameters for neutral atom measurement."""

    ETA = 0.1           # detection efficiency
    T_MEAS = 100e-6     # measurement time
    R_BRIGHT = 1e7      # bright state scattering rate
    R_BG = 1000         # background scattering rate
    R_B_TO_D = 960      # bright-to-dark transition rate
    R_D_TO_B = 2        # dark-to-bright transition rate


# pylint: disable=too-many-arguments
def _bright_pdf(
    n_counts: int,
    eta: float,
    t_m: float,
    r_0: float,
    r_bg: float,
    r_bd: float,
) -> float:
    """
    Poissonian probability density function of the bright (0) state measurement
    response, including error terms caused by 0 -> 1 transitions due to off-resonant
    pumping and background scattering rates. Including the effect of background scatter
    and detector dark counts, the effective scattering rate when atom is in bright (0)
    state is given by eta * r_0 + r_bg.

    Parameters
    ----------
    n_counts : int
        Number of photons detected in the measurement.
    eta : float
        Detection efficiency.
    t_m : float
        Measurement duration (s).
    r_0 : float
        Scattering rate of the bright state.
    r_bg : float
        Background scattering rate. Dominates the scattering rate when atom is in
        the dark state.
    r_bd : float
        Transition rate from the bright state to the dark state.

    Returns
    -------
    float
        The probability density of the bright state, given the number of counts.
    """
    log_a = (
        -(eta * r_0 + r_bg + r_bd) * t_m + n_counts * np.log(eta * r_0 + r_bg)
        + n_counts * np.log(t_m) - gammaln(n_counts + 1)
    )
    log_b = (
        np.log(r_bd) - r_bg * t_m - np.log(eta * r_0 + r_bg)
        + n_counts * np.log(eta * r_0) - n_counts * np.log(eta * r_0 + r_bd)
    )
    sum_1 = sum(np.exp(
            k * np.log(eta * r_0 + r_bg) + k * np.log(r_bg * t_m)
            - gammaln(k + 1) - k * np.log(eta * r_0)
        )
        for k in range(n_counts + 1)
    )
    sum_2 = np.exp(-(eta * r_0 + r_bd) * t_m) * sum(
        np.exp(
            k * np.log(eta * r_0 + r_bd) + k * np.log(eta * r_0 + r_bg) + k * np.log(t_m)
            - gammaln(k + 1) - k * np.log(eta * r_0)
        )
        for k in range(n_counts + 1)
    )
    return np.exp(log_a) + np.exp(log_b) * (sum_1 - sum_2)


# pylint: disable=too-many-arguments
def _dark_pdf(
    n_counts: int,
    eta: float,
    t_m: float,
    r_0: float,
    r_bg: float,
    r_db: float,
) -> float:
    """
    Poissonian probability density function of the dark (1) state measurement
    response, including error terms caused by 1 -> 0 transitions due to off-resonant
    pumping and background scattering rates. The total scattering rate when atom is
    in the dark state is dominated by the background scatter r_bg.

    Parameters
    ----------
    n_counts : int
        Number of photons detected in the measurement.
    eta : float
        Detection efficiency.
    t_m : float
        Measurement duration (s).
    r_0 : float
        Scattering rate of the bright state.
    r_bg : float
        Background scattering rate. Dominates the scattering rate when atom is in
        the dark state.
    r_db : float
        Transition rate from the dark state to the bright state.

    Returns
    -------
    float
        The probability density of the dark state, given the number of counts.
    """
    log_a = (
        - t_m * (r_bg + r_db) + n_counts * np.log(r_bg * t_m) - gammaln(n_counts + 1)
    )
    log_b = (
        -r_bg * t_m * np.log(r_db) - np.log(eta * r_0 - r_db)
        + n_counts * np.log(eta * r_0) - n_counts * np.log(eta * r_0 - r_db)
    )
    sum_1 = np.exp(-r_db * t_m) * sum(
        np.exp(
            k * np.log(eta * r_0 - r_db) + k * np.log(r_bg * t_m)
            - gammaln(k + 1) - k * np.log(eta * r_0)
        )
        for k in range(n_counts + 1)
    )
    sum_2 = np.exp(-eta * r_0 * t_m) * sum(
        np.exp(
           k * np.log(eta * r_0 - r_db) + k * np.log(eta * r_0 + r_bg)
           + k * np.log(t_m) - gammaln(k + 1) - k * np.log(eta * r_0)
        )
        for k in range(n_counts + 1)
    )
    return np.exp(log_a) + np.exp(log_b) * (sum_1 - sum_2)


class NeutralAtomPDF(AbstractPDF):
    """
    Probability density function class that describes the measurement response of
    non-destructive neutral atom readout. The soft measurement signal z is a photon
    count, based on which the 0- and 1-states can be discriminated.

    The functions describing the measurement PDFs are from the PhD thesis of M. E. Shea:
    'Fast, nondestructive, quantum-state readout of a single, trapped, neutral atom'
    Duke University, 2018, pp. 148-152, https://hdl.handle.net/10161/17492. Further
    details of the derivation can be found in the PhD thesis of S. G. Crain,
    Duke University, 2016, pp. 92-98, https://hdl.handle.net/10161/12270.

    Parameters
    ----------
    detection_efficiency : float, optional
        Detection efficiency. By default, 0.6%.
    t_measurement : float, optional
        Measurement duration (s). By default, 160 us.
    rate_bright_scatter : float, optional
        Scattering rate of the bright state. By default, 960.
    rate_background_scatter : float, optional
        Background scattering rate. Dominates the scattering rate when atom is in
        the dark state. By default, 1e3.
    rate_bright_to_dark : float, optional
        Transition rate from the bright state to the dark state.
    rate_dark_to_bright : float, optional
        Transition rate from the dark state to the bright state.
    seed : Optional[int], optional
        Seed for random sampling. By default, None.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        detection_efficiency: float = MeasParameters.ETA.value,
        t_measurement: float = MeasParameters.T_MEAS.value,
        rate_bright_scatter: float = MeasParameters.R_BRIGHT.value,
        rate_background_scatter: float = MeasParameters.R_BG.value,
        rate_bright_to_dark: float = MeasParameters.R_B_TO_D.value,
        rate_dark_to_bright: float = MeasParameters.R_D_TO_B.value,
        seed: Optional[int] = None,
    ):
        self.eta = detection_efficiency
        self.t_m = t_measurement
        self.r_0 = rate_bright_scatter
        self.r_bg = rate_background_scatter
        self.r_bd = rate_bright_to_dark
        self.r_db = rate_dark_to_bright
        # The domain is defined by the average emission rate - below we add a tail
        # of length 5 * stdev to make sure we capture the full distribution
        mean_rate = (self.eta * self.r_0 + self.r_bg) * self.t_m
        domain = np.arange(int(np.ceil(mean_rate + 5 * np.sqrt(mean_rate))))
        super().__init__(domain=domain, dtype=np.uint16, seed=seed)

    def _0_state_pdf(self, z_soft: NDArray[np.float_]) -> NDArray[np.float_]:
        return np.fromiter(map(
            lambda z: _dark_pdf(z, self.eta, self.t_m, self.r_0, self.r_bg, self.r_db),
            z_soft
        ), dtype=float)

    def _1_state_pdf(self, z_soft: NDArray[np.float_]) -> NDArray[np.float_]:
        return np.fromiter(map(
            lambda z: _bright_pdf(z, self.eta, self.t_m, self.r_0, self.r_bg, self.r_bd),
            z_soft
        ), dtype=float)

    @classmethod
    def from_error_probability(cls, p_soft: float):
        """Given an error probability between 0.2% and 10%, construct a `NeutralAtomPDF`
        class that has the given soft measurement error proabbility.

        Parameters
        ----------
        p_soft: float
            Target soft measurement error probability.
        """
        if (p_soft < 0.002) or (p_soft > 0.1):
            raise ValueError(
                f"Cannot construct NeutralAtomPDF class with p_soft={p_soft}."
                " Only values 0.002 < p_soft < 0.1 are supported."
            )

        def _obj_func(t_m: float) -> float:
            mock_pdf = NeutralAtomPDF(t_measurement=t_m)
            return p_soft - mock_pdf.readout_error_probability

        result = root_scalar(
            _obj_func,
            bracket=[1e-6, 2e-5],
            method="bisect",
            xtol=1e-12,
        )

        return cls(MeasParameters.ETA.value, result.root)
