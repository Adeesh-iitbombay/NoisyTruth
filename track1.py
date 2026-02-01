#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Empty
from auv_msgs.msg import DVLVel, PsData, AuvState
from auv_localization.ukf import UKF


class Localization(Node):

    def __init__(self):
        super().__init__('localization')

        # ===============================
        # PARAMETERS
        # ===============================
        self.declare_parameter('dt', 0.1)
        self.declare_parameter('use_ukf', True)
        self.declare_parameter('debug', False)

        self.dt = self.get_parameter('dt').value
        self.use_ukf = self.get_parameter('use_ukf').value
        self.debug = self.get_parameter('debug').value

        # ===============================
        # STATE STORAGE
        # ===============================
        self.imu_data = None          # [roll, pitch, yaw, p, q, r, ax, ay, az]
        self.dvl_velocity = None      # [u, v, w]
        self.depth = None

        self.origin_yaw = 0.0
        self.initialized = False

        # ===============================
        # ROS INTERFACES
        # ===============================
        self.sub_imu = self.create_subscription(Imu, '/imu/data', self.cb_imu, 10)
        self.sub_dvl = self.create_subscription(DVLVel, '/dvl/velData', self.cb_dvl, 10)
        self.sub_ps = self.create_subscription(PsData, '/ps/data', self.cb_ps, 10)

        self.pub_state = self.create_publisher(AuvState, '/localization/pose', 10)
        self.reset_srv = self.create_service(Empty, '/localization/reset', self.cb_reset)

        self.timer = self.create_timer(self.dt, self.run)

        # ===============================
        # UKF INITIALIZATION
        # ===============================
        self.init_ukf()

        self.get_logger().info("Localization node started")

    # ======================================================
    # UKF SETUP
    # ======================================================
    def init_ukf(self):
        n = 9

        initial_state = np.zeros(n)
        initial_cov = np.diag([1, 1, 1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

        Q = np.diag([0.01]*3 + [0.001]*3 + [0.05]*3)
        self.R_dvl = np.diag([0.05, 0.05, 0.05])
        self.R_depth = np.array([[0.1]])

        self.ukf = UKF(
            num_states=n,
            process_noise=Q,
            initial_state=initial_state,
            initial_covar=initial_cov,
            alpha=0.3,
            beta=2.0,
            k=0.0,
            iterate_function=self.motion_model
        )

    # ======================================================
    # CALLBACKS
    # ======================================================
    def cb_imu(self, msg: Imu):
        rot = Rotation.from_quat([
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w
        ])

        # Convert ENU → NED
        roll, pitch, yaw = rot.as_euler('XYZ', degrees=False)
        roll, pitch = roll, -pitch
        yaw = -yaw

        self.imu_data = np.array([
            roll, pitch, yaw,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

    def cb_dvl(self, msg: DVLVel):
        self.dvl_velocity = np.array([
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z
        ])

    def cb_ps(self, msg: PsData):
        self.depth = msg.depth

    def cb_reset(self, req, res):
        self.origin_yaw = self.ukf.get_state()[5]
        self.get_logger().info("Localization reset")
        return res

    # ======================================================
    # MOTION MODEL (UKF fx)
    # ======================================================
    def motion_model(self, state, dt, inputs):
        x, y, z, roll, pitch, yaw, u, v, w = state
        p, q, r, ax, ay, az = inputs

        g = 9.81

        # --- Orientation ---
        roll += p * dt
        pitch += q * dt
        yaw += r * dt

        # --- Velocity (gravity compensated) ---
        R = Rotation.from_euler('XYZ', [roll, pitch, yaw]).as_matrix()
        g_body = R.T @ np.array([0, 0, g])

        u += (ax - g_body[0]) * dt
        v += (ay - g_body[1]) * dt
        w += (az - g_body[2]) * dt

        # --- Position ---
        vel_world = R @ np.array([u, v, w])
        x += vel_world[0] * dt
        y += vel_world[1] * dt
        z += vel_world[2] * dt

        return np.array([x, y, z, roll, pitch, yaw, u, v, w])

    # ======================================================
    # MAIN LOOP
    # ======================================================
    def run(self):
        if self.imu_data is None or self.dvl_velocity is None or self.depth is None:
            return

        if self.use_ukf:
            self.ukf.predict(self.dt, self.imu_data[3:9])
            self.ukf.update([6, 7, 8], self.dvl_velocity, self.R_dvl)
            self.ukf.update([2], [self.depth], self.R_depth)

            state = self.ukf.get_state()
        else:
            state = np.zeros(9)

        self.publish_state(state)

    # ======================================================
    # PUBLISH STATE
    # ======================================================
    def publish_state(self, s):
        msg = AuvState()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.position.x = s[0]
        msg.position.y = s[1]
        msg.position.z = s[2]

        msg.orientation.roll = s[3]
        msg.orientation.pitch = s[4]
        msg.orientation.yaw = s[5] - self.origin_yaw

        msg.velocity.x = s[6]
        msg.velocity.y = s[7]
        msg.velocity.z = s[8]

        self.pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Localization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
