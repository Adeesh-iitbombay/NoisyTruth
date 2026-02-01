import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# CONSTANT VELOCITY PROCESS MODEL
# =====================================================
def constant_velocity_model(state, dt, inputs=None):
    """
    State = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state

    x += vx * dt
    y += vy * dt
    z += vz * dt

    return np.array([x, y, z, vx, vy, vz])


# =====================================================
# MINIMAL UKF IMPLEMENTATION (EDUCATIONAL)
# =====================================================
class UKF:
    def __init__(self, num_states, Q, init_state, init_covar,
                 alpha, kappa, beta, process_model):
        self.n = num_states
        self.Q = Q
        self.x = init_state
        self.P = init_covar
        self.fx = process_model

        self.alpha = alpha
        self.kappa = kappa
        self.beta = beta

        self.lambda_ = alpha**2 * (self.n + kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        self.Wm = np.full(2*self.n+1, 1/(2*(self.n+self.lambda_)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def sigma_points(self):
        sqrtP = np.linalg.cholesky(self.P)
        sigmas = [self.x]
        for i in range(self.n):
            sigmas.append(self.x + self.gamma * sqrtP[:, i])
            sigmas.append(self.x - self.gamma * sqrtP[:, i])
        return np.array(sigmas)

    def predict(self, dt):
        sigmas = self.sigma_points()
        sigmas_f = np.array([self.fx(s, dt) for s in sigmas])

        self.x = np.sum(self.Wm[:, None] * sigmas_f, axis=0)

        self.P = self.Q.copy()
        for i in range(len(sigmas_f)):
            diff = sigmas_f[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)

    def update(self, states, data, r_matrix):
        sigmas = self.sigma_points()
        z_sigmas = sigmas[:, states]

        z_pred = np.sum(self.Wm[:, None] * z_sigmas, axis=0)

        S = r_matrix.copy()
        Pxz = np.zeros((self.n, len(states)))

        for i in range(len(sigmas)):
            dz = z_sigmas[i] - z_pred
            dx = sigmas[i] - self.x
            S += self.Wc[i] * np.outer(dz, dz)
            Pxz += self.Wc[i] * np.outer(dx, dz)

        K = Pxz @ np.linalg.inv(S)
        innovation = np.array(data) - z_pred

        self.x += K @ innovation
        self.P -= K @ S @ K.T

    def get_state(self):
        return self.x.copy()


# =====================================================
# LOCALIZATION RUNNER
# =====================================================
def run_localization(filename):
    num_states = 6
    Q = np.eye(num_states) * 0.05
    init_state = np.zeros(num_states)
    init_covar = np.eye(num_states)

    ukf = UKF(num_states, Q, init_state, init_covar,
              alpha=0.3, kappa=0.0, beta=2.0,
              process_model=constant_velocity_model)

    last_time = None
    results = []

    with open(filename, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.strip().split(',')
            timestamp = float(parts[0])
            sensor = parts[1].strip().upper()

            if last_time is None:
                last_time = timestamp
                continue

            dt = timestamp - last_time
            if dt <= 0:
                continue
            last_time = timestamp

            ukf.predict(dt)

            if sensor == 'DEPTH':
                z = float(parts[2])
                ukf.update([2], [z], np.eye(1) * 0.5)

            elif sensor == 'GPS':
                x = float(parts[2])
                y = float(parts[3])
                ukf.update([0, 1], [x, y], np.eye(2) * 2.0)

            state = ukf.get_state()
            results.append([timestamp, state[0], state[1], state[2]])

    return np.array(results)


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    data = run_localization("sensor_data.txt")

    plt.figure()
    plt.plot(data[:, 0], data[:, 3], label="Filtered Depth")
    plt.xlabel("Time (s)")
    plt.ylabel("Depth (m)")
    plt.legend()
    plt.grid()
    plt.show()
