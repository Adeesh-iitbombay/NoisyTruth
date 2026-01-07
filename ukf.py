import numpy as np
import scipy.linalg
from threading import Lock


class UKFException(Exception):
    pass


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar,
                 alpha, k, beta, iterate_function):

        self.n_dim = int(num_states)
        self.n_sig = 1 + 2 * self.n_dim

        self.q = process_noise
        self.x = initial_state.reshape(-1, 1)
        self.p = initial_covar

        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.iterate = iterate_function

        self.lambd = self.alpha ** 2 * (self.n_dim + self.k) - self.n_dim

        self.mean_weights = np.zeros(self.n_sig)
        self.covar_weights = np.zeros(self.n_sig)

        self.mean_weights[0] = self.lambd / (self.n_dim + self.lambd)
        self.covar_weights[0] = self.mean_weights[0] + (1 - self.alpha ** 2 + self.beta)

        for i in range(1, self.n_sig):
            w = 1.0 / (2 * (self.n_dim + self.lambd))
            self.mean_weights[i] = w
            self.covar_weights[i] = w

        self.sigmas = self.__get_sigmas()
        self.lock = Lock()

  def __get_sigmas(self):
    # Use Cholesky for stability
    try:
        # L @ L.T = P
        L = np.linalg.cholesky((self.n_dim + self.lambd) * self.p)
    except np.linalg.LinAlgError:
        # Fallback or jitter addition if P is not positive-definite
        raise UKFException("Covariance matrix is not positive-definite")

    sigmas = np.zeros((self.n_dim, self.n_sig))
    sigmas[:, 0] = self.x.flatten()
    sigmas[:, 1:self.n_dim+1] = self.x + L
    sigmas[:, self.n_dim+1:] = self.x - L
    return sigmas

    def predict(self, timestep, inputs=[]):
        self.lock.acquire()

        sigmas_out = np.zeros_like(self.sigmas)

        for i in range(self.n_sig):
            sigmas_out[:, i] = self.iterate(self.sigmas[:, i], timestep, inputs)

        x_out = np.zeros((self.n_dim, 1))
        for i in range(self.n_sig):
            x_out += self.mean_weights[i] * sigmas_out[:, i].reshape(-1, 1)

        p_out = np.zeros((self.n_dim, self.n_dim))
        for i in range(self.n_sig):
            diff = sigmas_out[:, i].reshape(-1, 1) - x_out
            p_out += self.covar_weights[i] * (diff @ diff.T)

        p_out += timestep * self.q

        self.x = x_out
        self.p = p_out
        self.sigmas = self.__get_sigmas()

        self.lock.release()

    def update(self, states, data, r_matrix):
        self.lock.acquire()

        states = list(states)
        z = np.array(data).reshape(-1, 1)
        m = len(states)

        y = self.sigmas[states, :]

        y_mean = np.zeros((m, 1))
        for i in range(self.n_sig):
            y_mean += self.mean_weights[i] * y[:, i].reshape(-1, 1)

        y_diff = np.zeros((m, self.n_sig))
        x_diff = np.zeros((self.n_dim, self.n_sig))

        for i in range(self.n_sig):
            y_diff[:, i] = (y[:, i].reshape(-1, 1) - y_mean).flatten()
            x_diff[:, i] = (self.sigmas[:, i].reshape(-1, 1) - self.x).flatten()

        p_yy = np.zeros((m, m))
        for i in range(self.n_sig):
            dy = y_diff[:, i].reshape(-1, 1)
            p_yy += self.covar_weights[i] * (dy @ dy.T)

        p_yy += r_matrix

        p_xy = np.zeros((self.n_dim, m))
        for i in range(self.n_sig):
            dx = x_diff[:, i].reshape(-1, 1)
            dy = y_diff[:, i].reshape(-1, 1)
            p_xy += self.covar_weights[i] * (dx @ dy.T)

        k_gain = p_xy @ np.linalg.inv(p_yy)

        self.x = self.x + k_gain @ (z - y_mean)
        self.p = self.p - k_gain @ p_yy @ k_gain.T

        self.sigmas = self.__get_sigmas()

        self.lock.release()

    def get_state(self, index=-1):
        if index >= 0:
            return self.x[index, 0]
        return self.x

    def get_covar(self):
        return self.p

    def set_state(self, value, index=-1):
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value.reshape(-1, 1)

    def reset(self, state, covar):
        with self.lock:
            self.x = state.reshape(-1, 1)
            self.p = covar
            self.sigmas = self.__get_sigmas()
