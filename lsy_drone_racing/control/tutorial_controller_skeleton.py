"""Tutorial: waypoint-following position PID in attitude mode.

Fill in every TODO. Action order returned to the env: [roll, pitch, yaw, thrust].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from crazyflow.dynamics import load_params as load_dynamics_params
from crazyflow.sim.visualize import draw_line, draw_points

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.tutorial_utils.fake_sensors import TruthEstimator
from lsy_drone_racing.control.tutorial_utils.geometry import force_to_action
from lsy_drone_racing.control.tutorial_utils.history import History
from lsy_drone_racing.control.tutorial_utils.plotting import save_plot

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class TutorialController(Controller):
    """Fly to a list of waypoints, one after another, with your own position controller."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Set up the controller. Called once at the start of every episode."""
        super().__init__(obs, info, config)
        self.dt = 1.0 / config.env.freq
        self.mass = load_dynamics_params(config.sim.dynamics, config.sim.drone)["mass"]
        self.estimator = TruthEstimator()
        self.history = History()

        # TODO: choose waypoints in world coordinates (m). Hint: the start is obs["pos"].
        #       Stay inside the arena: x, y in [-3, 3] m, z below 2.5 m.
        self.waypoints: list[list[float]] = [[0.0, 0.0, 1.0]]
        self.epsilon = 0.1  # TODO: switching radius (m)
        self.wp_idx = 0

        # TODO: add whatever controller state you need (gains, integrator, ...)

        self._t = 0.0

    def current_target(self) -> np.ndarray:
        """The waypoint currently being flown to, always as a float array of shape (3,)."""
        return np.asarray(self.waypoints[self.wp_idx], dtype=float)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next action. Called once per env step (50 Hz)."""
        est = self.estimator.step(obs, self._t)  # use est.pos / est.vel / est.quat only

        # TODO: if within self.epsilon of the current waypoint and it isn't the last one,
        #       advance self.wp_idx. The last waypoint is held forever.

        # TODO: compute f_des, the desired world-frame force on the drone (N).
        #       Use self.current_target(), est.pos, est.vel, self.dt and self.mass.
        f_des = np.zeros(3)
        yaw_des = 0.0

        action = force_to_action(f_des, yaw_des, est.quat)

        # Required: one log entry per step. Add any extra fields you like.
        self.history.log(
            t=self._t,
            pos=est.pos,
            vel=est.vel,
            quat=est.quat,
            target=self.current_target(),
            action=action,
        )
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Called after every env step. Return True to end the episode early."""
        self._t += self.dt
        return False  # TODO (optional): return True after holding the last waypoint for a while

    def episode_callback(self):
        """Called once at the end of every episode. Saves the flight log and its plot to logs/."""
        log_path = self.history.save("tutorial_log.npz")
        if log_path is not None:
            print(f"Saved log:  {log_path}\nSaved plot: {save_plot(log_path)}")
        # TODO: reset any controller state you added
        self.history.reset()
        self.estimator.reset()
        self.wp_idx = 0
        self._t = 0.0

    def render_callback(self, sim: Sim):
        """Draw the waypoint path (green) and the current target (red) in the sim window."""
        if self.waypoints:
            pts = np.asarray(self.waypoints, dtype=float)
            if len(pts) > 1:
                draw_line(sim, pts, rgba=(0, 1, 0, 1))
            draw_points(sim, self.current_target().reshape(1, -1), rgba=(1, 0, 0, 1), size=0.03)