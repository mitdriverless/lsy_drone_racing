"""Fake sensors and simple estimators for the controls tutorial.

FakeSensors turns the sim's ground truth into A2RL-like measurements:
  - IMU (accel + gyro) every env step
  - attitude every step (near-truth, like Betaflight's onboard estimate)
  - a low-rate, delayed, noisy position fix (stand-in for camera/gate PnP)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorConfig:
    env_freq: float = 50.0          # Hz, must match config.env.freq
    fix_every_n_steps: int = 3      # ~17 Hz position fix at 50 Hz env
    fix_delay_steps: int = 1        # 20 ms latency at 50 Hz
    fix_pos_std: float = 0.03       # m
    fix_dropout_prob: float = 0.05  # chance a fix is lost
    acc_std: float = 0.2            # m/s^2
    acc_bias_std: float = 0.05      # m/s^2, constant per episode
    gyro_std: float = 0.01          # rad/s
    quat_noise_std: float = 0.0     # leave clean for now
    seed: int | None = None


@dataclass
class Measurement:
    t: float
    acc: np.ndarray                 # (3,) WORLD frame, gravity removed (simplification)
    gyro: np.ndarray                # (3,) body rates
    quat: np.ndarray                # (4,) xyzw
    pos_fix: np.ndarray | None      # (3,) or None if no new fix this step
    pos_fix_t: float | None         # time the fix was CAPTURED (t - delay)


class FakeSensors:
    def __init__(self, cfg: SensorConfig = SensorConfig()):
        self.cfg = cfg
        self.dt = 1.0 / cfg.env_freq
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.cfg.seed)
        self.acc_bias = self.rng.normal(0.0, self.cfg.acc_bias_std, 3)
        self._prev_vel = None
        self._step = 0
        self._pending = deque()     # (deliver_step, capture_t, noisy_pos)

    def step(self, obs: dict, t: float) -> Measurement:
        c = self.cfg
        vel = np.asarray(obs["vel"], dtype=float)

        # Accelerometer: finite-difference true velocity, add bias + noise
        acc_true = np.zeros(3) if self._prev_vel is None else (vel - self._prev_vel) / self.dt
        self._prev_vel = vel.copy()
        acc = acc_true + self.acc_bias + self.rng.normal(0.0, c.acc_std, 3)

        gyro = np.asarray(obs["ang_vel"], dtype=float) + self.rng.normal(0.0, c.gyro_std, 3)
        quat = np.asarray(obs["quat"], dtype=float)  # TODO later: add small attitude noise

        # Capture a position fix now; deliver it after the delay
        if self._step % c.fix_every_n_steps == 0 and self.rng.random() > c.fix_dropout_prob:
            noisy = np.asarray(obs["pos"], dtype=float) + self.rng.normal(0.0, c.fix_pos_std, 3)
            self._pending.append((self._step + c.fix_delay_steps, t, noisy))

        pos_fix, pos_fix_t = None, None
        if self._pending and self._pending[0][0] <= self._step:
            _, pos_fix_t, pos_fix = self._pending.popleft()

        self._step += 1
        return Measurement(t, acc, gyro, quat, pos_fix, pos_fix_t)


# ---------------------------------------------------------------------------
# Estimate interface: the controller only ever reads a StateEstimate.
# ---------------------------------------------------------------------------

@dataclass
class StateEstimate:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    t: float


class TruthEstimator:
    """Session 1: pass through ground truth."""

    def reset(self):
        pass

    def step(self, obs: dict, t: float) -> StateEstimate:
        return StateEstimate(np.asarray(obs["pos"]), np.asarray(obs["vel"]),
                             np.asarray(obs["quat"]), t)


class NaiveEstimator:
    """Demo: hold the last noisy fix, finite-difference it for velocity. No filtering."""

    def __init__(self, sensors: FakeSensors):
        self.sensors = sensors
        self.reset()

    def reset(self):
        self.sensors.reset()
        self.pos, self.vel = None, np.zeros(3)
        self._last_fix_t = None

    def step(self, obs: dict, t: float) -> StateEstimate:
        m = self.sensors.step(obs, t)
        if m.pos_fix is not None:
            if self.pos is not None and self._last_fix_t is not None:
                self.vel = (m.pos_fix - self.pos) / max(m.pos_fix_t - self._last_fix_t, 1e-3)
            self.pos, self._last_fix_t = m.pos_fix, m.pos_fix_t
        if self.pos is None:  # before the first fix arrives
            self.pos = np.asarray(obs["pos"], dtype=float)
        return StateEstimate(self.pos, self.vel, m.quat, t)


# Session 2: class KalmanEstimator with the same reset()/step(obs, t) -> StateEstimate