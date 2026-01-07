import numpy as np
import scipy.linalg
from threading import Lock


class UKFException(Exception):
    pass


class UKF:
    """
    Fully generic Unscented Kalman Filter (UKF)

    Assumptions:
    - Process noise is additive Gaussian with covariance Q
    - Measurement noise is additive Gaussian with covariance R
    - f(x, dt, u) is deterministic
    - h(x) can be any nonlinear measurement function

    State shape: (n, 1)
    """

    def __init__(
        self,
        num_states,
        process_noise,
        initial_state,
        initial_covar,
        alpha=1e-3,
        kappa=0,
        beta=2.0,
    ):
        self.n = int(num_states)
        self.n_sig = 2 * self.n + 1

        self.x = initial_state.reshape(self.n, 1)
        self.P = initial_covar
        self.Q = process_noise

        # UKF scaling parameters
        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta

        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        # Weights
        self.Wm = np.full(self.n_sig, 1.0 / (2 * (self.n + self.lambda_)))
        self.Wc = self.Wm.copy()

        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - self.alpha**2 + self.beta)

        self.lock = Lock()
        self.sigmas = self._compute_sigma_points()

    # ------------------------------------------------------------------
    # Sigma points
    # ------------------------------------------------------------------
    def _compute_sigma_points(self):
        """Generate sigma points using Cholesky factorization"""
        try:
            P = self.P + 1e-9 * np.eye(self.n)
            S = scipy.linalg.cholesky(P, lower=True)
        except np.linalg.LinAlgError:
            raise UKFException("Covariance matrix is not positive definite")

        sigmas = np.zeros((self.n, self.n_sig))
        sigmas[:, 0] = self.x[:, 0]

        for i in range(self.n):
            offset = self.gamma * S[:, i]
            sigmas[:, i + 1] = self.x[:, 0] + offset
            sigmas[:, i + 1 + self.n] = self.x[:, 0] - offset

        return sigmas

    # ------------------------------------------------------------------
    # Prediction step
    # ------------------------------------------------------------------
    def predict(self, f, dt, u=None):
        """
        f : process model function f(x, dt, u) -> (n,)
        """
        if u is None:
            u = []

        with self.lock:
            # Propagate sigma points
            sigmas_pred = np.zeros_like(self.sigmas)
            for i in range(self.n_sig):
                sigmas_pred[:, i] = f(self.sigmas[:, i], dt, u)

            # Predicted mean
            self.x = (sigmas_pred @ self.Wm).reshape(self.n, 1)

            # Predicted covariance
            diff = sigmas_pred - self.x
            self.P = diff @ np.diag(self.Wc) @ diff.T + dt * self.Q

            # Enforce symmetry
            self.P = 0.5 * (self.P + self.P.T)

            self.sigmas = self._compute_sigma_points()

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, z, h, R):
        """
        z : measurement vector (m, 1)
        h : measurement model h(x) -> (m,)
        R : measurement noise covariance (m, m)
        """
        z = np.asarray(z).reshape(-1, 1)
        m = z.shape[0]

        with self.lock:
            # Transform sigma points into measurement space
            Y = np.zeros((m, self.n_sig))
            for i in range(self.n_sig):
                Y[:, i] = h(self.sigmas[:, i])

            # Measurement mean
            y_mean = (Y @ self.Wm).reshape(m, 1)

            # Innovation covariance
            y_diff = Y - y_mean
            P_yy = y_diff @ np.diag(self.Wc) @ y_diff.T + R

            # Cross covariance
            x_diff = self.sigmas - self.x
            P_xy = x_diff @ np.diag(self.Wc) @ y_diff.T

            # Kalman gain (NO explicit inverse)
            K = P_xy @ scipy.linalg.solve(P_yy, np.eye(m), assume_a="pos")

            # Update state and covariance
            self.x = self.x + K @ (z - y_mean)
            self.P = self.P - K @ P_yy @ K.T

            # Enforce symmetry
            self.P = 0.5 * (self.P + self.P.T)

            self.sigmas = self._compute_sigma_points()

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def get_state(self):
        return self.x.copy()

    def get_covariance(self):
        return self.P.copy()

    def reset(self, state, covariance):
        with self.lock:
            self.x = state.reshape(self.n, 1)
            self.P = covariance
            self.sigmas = self._compute_sigma_points()
