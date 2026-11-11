"""Actuation and tire measurements for the course car."""

import math

import mujoco
import numpy as np


WHEELBASE = .384
HALF_TRACK = .169
TIRE_RADIUS = .086
TIRE_HALF_WIDTH = .032
MAX_SPEED = 5.
MAX_STEER = .44
MAX_WHEEL_ANGLE = .54
STEER_RATE = 1.85
DRIVE_ACCEL = 3.
BRAKE_ACCEL = 5.
NO_LOAD_SPEED = MAX_SPEED / TIRE_RADIUS
WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
SIDES = np.array([HALF_TRACK, -HALF_TRACK])
CORNERS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])


class Vehicle:
    def __init__(self, model, data, diagnostics):
        self.model = model
        self.data = data
        self.diagnostics = diagnostics
        self.steer_qpos = np.array([model.joint(f"steer_{side}").qposadr.item()
                                   for side in ("left", "right")])
        self.rear_dof = np.array([model.joint(name).dofadr.item() for name in WHEELS[2:]])
        self.tires = [model.geom(f"{name}_tire").id for name in WHEELS]
        self.tire_bodies = model.geom_bodyid[self.tires]
        self.tire_body = dict(zip(self.tires, self.tire_bodies.tolist()))
        self.floor = model.geom("floor").id
        self.chassis = model.geom("chassis").id
        self.chassis_vertices = np.array([[x, y, z] for x in (-.27, .27)
                                          for y in (-.15, .15) for z in (-.045, .045)])
        self.contact_force = np.empty(6)
        self.wheel_velocity = np.empty(6)
        self.reset()

    def reset(self):
        self.curvature = 0.
        self.speed_target = 0.
        self.steer_target = np.zeros(2)

    def apply(self, steering, speed):
        dt = self.model.opt.timestep
        self.speed_target += float(np.clip(speed - self.speed_target,
                                           -BRAKE_ACCEL * dt, DRIVE_ACCEL * dt))

        # Bound each wheel's rate while preserving a common Ackermann turn center.
        reach = STEER_RATE * dt
        low = np.tan(np.maximum(-MAX_WHEEL_ANGLE, self.steer_target - reach))
        high = np.tan(np.minimum(MAX_WHEEL_ANGLE, self.steer_target + reach))
        min_curvature = (low / (WHEELBASE + SIDES * low)).max()
        max_curvature = (high / (WHEELBASE + SIDES * high)).min()
        desired_curvature = math.tan(steering) / WHEELBASE
        self.curvature = float(np.clip(desired_curvature, min_curvature, max_curvature))
        self.steer_target[:] = np.arctan2(WHEELBASE * self.curvature, 1. - SIDES * self.curvature)
        self.data.ctrl[:2] = self.steer_target

        ground_speed = math.hypot(*self.data.qvel[:2])
        omega = self.data.qvel[self.rear_dof]
        wheel_target = self.speed_target * (1. - SIDES * self.curvature) / TIRE_RADIUS
        rolloff = np.clip((NO_LOAD_SPEED - omega) / (.25 * NO_LOAD_SPEED), 0., 1.)
        governor = float(np.clip((MAX_SPEED - ground_speed) / .15, 0., 1.))
        drive_limit = .23 * rolloff * governor
        brake_limit = .38 * np.clip(omega / 1., 0., 1.)
        requested = .09 * (wheel_target - omega)
        overspeed_brake = 3.8 * max(0., ground_speed - MAX_SPEED)
        self.data.ctrl[2:] = np.clip(requested - overspeed_brake, -brake_limit, drive_limit)

    def footprint(self):
        data = self.data
        chassis_rotation = data.geom_xmat[self.chassis].reshape(3, 3)
        chassis = data.geom_xpos[self.chassis, :2] + self.chassis_vertices @ chassis_rotation[:2].T
        axle = data.xmat[self.tire_bodies].reshape(-1, 3, 3)[:, :, 1]
        axle_len = np.linalg.norm(axle[:, :2], axis=1)
        across = axle[:, :2] / np.maximum(axle_len[:, None], 1e-9)
        across[axle_len < 1e-9] = [0., 1.]
        along = across[:, ::-1] * [1., -1.]
        width = TIRE_HALF_WIDTH * axle_len + TIRE_RADIUS * np.abs(axle[:, 2])
        tire_corners = (data.xpos[self.tire_bodies, None, :2]
                        + across[:, None] * width[:, None, None] * CORNERS[None, :, :1]
                        + TIRE_RADIUS * along[:, None] * CORNERS[None, :, 1:])
        return np.vstack((data.qpos[:2], chassis, tire_corners.reshape(-1, 2)))

    def contacts(self):
        peak_grip = peak_slip = risk = load = 0.
        for i, contact in enumerate(self.data.contact):
            if contact.efc_address < 0:
                continue
            tire = contact.geom2 if contact.geom1 == self.floor else contact.geom1
            if self.floor not in (contact.geom1, contact.geom2) or tire not in self.tire_body:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, self.contact_force)
            normal, x, y = self.contact_force[:3]
            if normal < .10:
                continue
            grip = math.hypot(x, y) / (contact.friction[0] * normal)
            peak_grip = max(peak_grip, grip)
            risk += normal * min(1., max(0., (grip - .85) / .15)) ** 2
            load += normal
            if self.diagnostics:
                body = self.tire_body[tire]
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                         body, self.wheel_velocity, 0)
                angular, linear = self.wheel_velocity[:3], self.wheel_velocity[3:]
                offset = contact.pos - self.data.xipos[body]
                velocity = linear + np.cross(angular, offset)
                tangent = contact.frame.reshape(3, 3)[1:] @ velocity
                peak_slip = max(peak_slip, float(np.linalg.norm(tangent)))
        return risk, load, peak_grip, peak_slip
