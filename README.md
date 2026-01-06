# NoisyTruth
The repository deals with the implementation of Unscented Kalman Filter from scratch. 

## What is this UKF useful for?

The Unscented Kalman Filter is used for **state estimation in nonlinear systems**.  
Unlike the Extended Kalman Filter (EKF), it does **not require Jacobians or complex calculations in linear algebra** and handles nonlinearities more accurately by chosing sigma points.

Typical applications include:
- Robotics and autonomous vehicles
- Sensor fusion
- Tracking and navigation
- Control systems

---

## Features of this Implementation

- Pure NumPy + SciPy (no heavy dependencies)
- Thread-safe (uses locks)
- Clear separation of **predict** and **update**
- Supports **partial state measurements**
- Easy to plug in your own system dynamics

---


## How the UKF Works (High-Level)

1. **Sigma points** are generated around the current state.
2. These points are **propagated through the system model**.
3. The predicted mean and covariance are calculated.
4. Measurements are used to **correct the prediction** using a Kalman gain.

All of this happens without forcefully linearizing the system.

---
