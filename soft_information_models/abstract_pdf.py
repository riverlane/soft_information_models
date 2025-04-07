from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray


class AbstractPDF(ABC):
    """
    Abstract base class for measurement probability density functions. Implements
    logic for fast sampling and probability evaluation.

    Parameters
    ----------
    domain : NDArray
        Range of the PDF, given in a 1D Numpy array.
    dtype : np.dtype
        Default data type of the soft measurement value.
    seed : Optional[int], optional
        Seed for random generated sampling.
    """

    def __init__(
        self,
        domain: NDArray,
        dtype: DTypeLike,
        seed: Optional[int] = None,
    ):
        self._domain = domain
        self.dtype = dtype
        self._random_number_generator = np.random.default_rng(seed=seed)

    @abstractmethod
    def _0_state_pdf(
        self,
        z_soft: NDArray
    ) -> NDArray[np.float64]:
        """Probability density function for the 0-state measurement response."""

    @abstractmethod
    def _1_state_pdf(
        self,
        z_soft: NDArray
    ) -> NDArray[np.float64]:
        """Probability density function for the 1-state measurement response."""

    @cached_property
    def _0_state_values(self) -> NDArray[np.float64]:
        """PDF of the 0-state pre-computed across the domain."""
        return self._0_state_pdf(self._domain)

    @cached_property
    def _1_state_values(self) -> NDArray[np.float64]:
        """PDF of the 1-state pre-computed across the domain."""
        return self._1_state_pdf(self._domain)

    @property
    def readout_error_probability(self) -> float:
        """
        The readout error probability is given by the overlap p(i|j), i!=j
        for states i, j in {0, 1}.
        """
        p_0 = self._0_state_values
        p_1 = self._1_state_values
        p_1_given_0 = np.sum(p_0[p_1 > p_0]) / np.sum(p_0)
        p_0_given_1 = np.sum(p_1[p_0 > p_1]) / np.sum(p_1)
        return (p_1_given_0 + p_0_given_1) / 2

    @cached_property
    def _sorted_cdf_0_state(self) -> NDArray[np.float64]:
        """
        Pre-compute the discretised CDF of the 0-state measurement in ascending
        order, for use in inversion sampling.
        """
        pdf_values = self._0_state_pdf(self._domain)
        return np.array(np.cumsum(pdf_values) / sum(pdf_values))

    @cached_property
    def _sorted_cdf_1_state(self) -> NDArray[np.float64]:
        """
        Pre-compute the discretised CDF of the 1-state measurement in ascending
        order, for use in inversion sampling.
        """
        pdf_values = self._1_state_pdf(self._domain)
        return np.array(np.cumsum(pdf_values) / sum(pdf_values))

    def sample(self, z_ideal: NDArray[np.int_]) -> NDArray[Any]:
        """
        Given a binary vector of ideal measurements, sample soft measurement
        outcomes from the 0-state PDF for any 0 measurements and the 1-state PDF
        for any 1 measurements.

        Parameters
        ----------
        z_ideal : NDArray[np.int_]
            Vector of ideal measurement outcomes 0 and 1.

        Returns
        -------
        NDArray[Any]
            Vector of soft measurements.
        """
        soft_meas_array = np.zeros(len(z_ideal), dtype=self.dtype)
        uniform_samples = self._random_number_generator.random(len(z_ideal))
        one_indices = z_ideal == 1
        num_0_samples = len(z_ideal) - np.count_nonzero(z_ideal)
        zero_measurements = self._domain[
            np.searchsorted(self._sorted_cdf_0_state, uniform_samples[:num_0_samples])
        ]
        one_measurements = self._domain[
            np.searchsorted(self._sorted_cdf_1_state, uniform_samples[num_0_samples:])
        ]
        soft_meas_array[~one_indices] = zero_measurements
        soft_meas_array[one_indices] = one_measurements
        return soft_meas_array

    def sample_p(self, z_ideal: NDArray[np.int_]) -> NDArray[np.float64]:
        """
        Given a binary vector of ideal measurements, calculate a vector of soft
        measurement probabilities [[P(0 | z)], [P(1 | z)]].T where z are soft
        measurements sampled from the 0-state PDF for any 0 measurements and the
        1-state PDF for any 1 measurements.

        Parameters
        ----------
        z_ideal : NDArray[np.int_]
            Vector of ideal measurement outcomes 0 and 1.

        Returns
        -------
        NDArray[np.float64]
            Probability matrix of the soft measurement corresponding to a 0- or 1-
            state preparation.
        """
        p_0_eval = np.zeros(len(z_ideal), dtype=float)
        p_1_eval = np.zeros(len(z_ideal), dtype=float)
        uniform_samples = self._random_number_generator.random(len(z_ideal))
        zero_indices = z_ideal == 0
        num_0_samples = len(z_ideal) - np.count_nonzero(z_ideal)
        zero_loc = np.searchsorted(
            self._sorted_cdf_0_state, uniform_samples[:num_0_samples]
        )
        one_loc = np.searchsorted(
            self._sorted_cdf_1_state, uniform_samples[num_0_samples:]
        )
        p_0_eval[zero_indices] = self._0_state_values[zero_loc]
        p_0_eval[~zero_indices] = self._0_state_values[one_loc]
        p_1_eval[zero_indices] = self._1_state_values[zero_loc]
        p_1_eval[~zero_indices] = self._1_state_values[one_loc]
        return (np.vstack((p_0_eval, p_1_eval)) / (p_0_eval + p_1_eval)).T

    def predict_p(
        self,
        z_soft: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Given a length N vector z_soft of soft measurement values, calculate the Nx2
        probability array [p_0, p_1].T, where p_i = P(z | i) / sum_i[P(z | i)] for a
        probability density function P(z | i) of the measurement outcome i = {0, 1}.

        Parameters
        ----------
        z_soft : NDArray[np.float64]
            Vector of soft measurements.

        Returns
        -------
        NDArray[np.float64]
            Probability matrix of the soft measurement corresponding to a 0- or 1-
            state preparation.
        """
        eval_0 = self._0_state_pdf(z_soft)
        eval_1 = self._1_state_pdf(z_soft)
        return (np.vstack((eval_0, eval_1)) / (eval_0 + eval_1)).T
