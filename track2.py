import numpy as np
import scipy.linalg
from ukf import UKF
from scipy.spatial.transform import Rotation


# ==========================================================
# AUV PROCESS MODEL (USED BY UKF)
# ==========================================================
def auv_process_model(state, dt, u):
    """
    State:
    [x, y, z, roll, pitch, yaw, u, v, w]

    Input u:
    [p, q, r, ax, ay, az]  (IMU)
    """

    # Unpack state
    x, y, z, roll, pitch, yaw, u_b, v_b, w_b = state
    p, q, r, ax, ay, az = u

    g = 9.81  # gravity (NED, positive down)

    # --------------------------------------------------
    # 1. Orientation propagation (Euler integration)
    # --------------------------------------------------
    roll  += p * dt
    pitch += q * dt
    yaw   += r * dt

    # --------------------------------------------------
    # 2. Velocity propagation (body frame)
    # --------------------------------------------------
    R = Rotation.from_euler("XYZ", [roll, pitch, yaw]).as_matrix()

    # gravity expressed in body frame
    g_body = R.T @ np.array([0.0, 0.0, g])

    u_b += (ax - g_body[0]) * dt
    v_b += (ay - g_body[1]) * dt
    w_b += (az - g_body[2]) * dt

    # --------------------------------------------------
    # 3. Position propagation (world frame)
    # --------------------------------------------------
    v_world = R @ np.array([u_b, v_b, w_b])

    x += v_world[0] * dt
    y += v_world[1] * dt
    z += v_world[2] * dt

    return np.array([x, y, z, roll, pitch, yaw, u_b, v_b, w_b])


# ==========================================================
# MEASUREMENT MODELS
# ==========================================================
def h_dvl(state):
    """DVL measures body-frame velocity"""
    return state[6:9]


def h_pressure(state):
    """Pressure sensor measures depth (z)"""
    return np.array([state[2]])


# ==========================================================
# LOCALIZATION RUNNER
# ==========================================================
def run_localization(data_stream):
    """
    data_stream: iterable of dicts with keys:
    {
        't': timestamp,
        'imu': [p, q, r, ax, ay, az],
        'dvl': [u, v, w] or None,
        'pressure': depth or None
    }
    """

    # -------------------------------
    # UKF initialization
    # -------------------------------
    n = 9

    Q = np.diag([
        0.01, 0.01, 0.01,     # position
        0.001, 0.001, 0.001,  # orientation
        0.05, 0.05, 0.05     # velocity
    ])

    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.5

    ukf = UKF(
        num_states=n,
        process_noise=Q,
        initial_state=x0,
        initial_covar=P0,
        alpha=0.3,
        kappa=0.0,
        beta=2.0
    )

    R_dvl = np.diag([0.05, 0.05, 0.05])
    R_ps = np.array([[0.1]])

    last_t = None
    trajectory = []

    # -------------------------------
    # Main loop
    # -------------------------------
    for entry in data_stream:
        t = entry["t"]

        if last_t is None:
            last_t = t
            continue

        dt = t - last_t
        if dt <= 0:
            continue
        last_t = t

        # -------- Prediction --------
        ukf.predict(
            f=auv_process_model,
            dt=dt,
            u=entry["imu"]
        )

        # -------- DVL update --------
        if entry["dvl"] is not None:
            ukf.update(
                z=entry["dvl"],
                h=h_dvl,
                R=R_dvl
            )

        # -------- Pressure update --------
        if entry["pressure"] is not None:
            ukf.update(
                z=[entry["pressure"]],
                h=h_pressure,
                R=R_ps
            )

        state = ukf.get_state().flatten()
        trajectory.append([t, state[0], state[1], state[2]])

    return np.array(trajectory)
