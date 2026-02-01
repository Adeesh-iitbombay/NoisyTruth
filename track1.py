#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from scipy.spatial.transform import Rotation

from sensor_msgs.msg import Imu
from std_srvs.srv import Empty
from auv_msgs.msg import DVLVel, PsData, AuvState
from auv_localization.ukf import UKF


class Localization(Node):

    def __init__(self):
        super().__init__('localization')

        # ===============================
        # PARAMETERS
        # ===============================
        self.declare_parameter('debug', False)
        self.debug = self.get_parameter('debug').value

        # ===============================
        # STATE & INPUT STORAGE
        # ===============================
        self.imu_input = None      # [p, q, r, ax, ay, az]  (NED, body)
        self.dvl_meas = None       # [u, v, w]             (body)
        self.depth_meas = None     # z (down)

        self.last_imu_time = None
        self.origin_yaw = 0.0
        self.initialized = False

        # ===============================
        # ROS INTERFACES
        # ===============================
        self.create_subscription(Imu, '/imu/data', self.cb_imu, 20)
        self.create_subscription(DVLVel, '/dvl/velData', self.cb_dvl, 10)
        self.create_subscription(PsData, '/ps/data', self.cb_ps, 10)

        self.pub_state = self.create_publisher(AuvState, '/localization/pose', 10)
        self.create_service(Empty, '/localization/reset', self.cb_reset)

        # ===============================
        # UKF INITIALIZATION
        # ===============================
        self.init_ukf()

        self.get_logger().info("Localization node initialized")

    # ======================================================
    # UKF SETUP
    # ======================================================
    def init_ukf(self):
        n = 9  # [x,y,z, roll,pitch,yaw, u,v,w]

        x0 = np.zeros(n)
        P0 = np.diag([1, 1, 1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5])

        Q = np.diag([
            0.01, 0.01, 0.01,
            0.001, 0.001, 0.001,
            0.05, 0.05, 0.05
        ])

        self.R_dvl = np.diag([0.05, 0.05, 0.05])
        self.R_depth = np.array([[0.1]])

        self.ukf = UKF(
            num_states=n,
            process_noise=Q,
            initial_state=x0,
            initial_covar=P0,
            alpha=0.3,
            beta=2.0,
            k=0.0,
            iterate_function=self.process_model
        )

    # ======================================================
    # CALLBACKS
    # ======================================================
    def cb_imu(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        # Angular velocity ENU → NED
        p = msg.angular_velocity.x
        q = -msg.angular_velocity.y
        r = -msg.angular_velocity.z

        # Linear acceleration ENU → NED
        ax = msg.linear_acceleration.x
        ay = -msg.linear_acceleration.y
        az = -msg.linear_acceleration.z

        self.imu_input = np.array([p, q, r, ax, ay, az])

        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0 or not self.initialized:
            return

        # ---------- UKF PREDICT ----------
        self.ukf.predict(dt, self.imu_input)

        # ---------- MEASUREMENT UPDATES ----------
        if self.dvl_meas is not None:
            self.ukf.update([6, 7, 8], self.dvl_meas, self.R_dvl)

        if self.depth_meas is not None:
            self.ukf.update([2], [self.depth_meas], self.R_depth)

        self.publish_state()

    def cb_dvl(self, msg: DVLVel):
        self.dvl_meas = np.array([
            msg.velocity.x,
            msg.velocity.y,
            msg.velocity.z
        ])

    def cb_ps(self, msg: PsData):
        self.depth_meas = msg.depth

        if not self.initialized:
            state = self.ukf.get_state()
            state[2] = self.depth_meas
            self.ukf.reset(state, self.ukf.get_covariance())
            self.initialized = True
            self.get_logger().info("Localization initialized")

    def cb_reset(self, req, res):
        self.origin_yaw = self.ukf.get_state()[5]
        self.get_logger().info("Localization yaw reset")
        return res

    # ======================================================
    # PROCESS MODEL
    # ======================================================
    def process_model(self, state, dt, u):
        x, y, z, roll, pitch, yaw, u_b, v_b, w_b = state
        p, q, r, ax, ay, az = u

        g = 9.81

        # Orientation
        roll  += p * dt
        pitch += q * dt
        yaw   += r * dt

        R = Rotation.from_euler('XYZ', [roll, pitch, yaw]).as_matrix()

        g_body = R.T @ np.array([0, 0, g])

        u_b += (ax - g_body[0]) * dt
        v_b += (ay - g_body[1]) * dt
        w_b += (az - g_body[2]) * dt

        vel_world = R @ np.array([u_b, v_b, w_b])

        x += vel_world[0] * dt
        y += vel_world[1] * dt
        z += vel_world[2] * dt

        return np.array([x, y, z, roll, pitch, yaw, u_b, v_b, w_b])

    # ======================================================
    # PUBLISH
    # ======================================================
    def publish_state(self):
        s = self.ukf.get_state()

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
