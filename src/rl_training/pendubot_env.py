import gymnasium as gym
from gymnasium import spaces
import numpy as np
import scipy.integrate
import traceback

from src.core_logic.pendubot_dynamics import pendubot_dynamics, G

VALID_TARGET_CONFIGS = [
    (np.pi, 0.0),  # L1 Up, L2 Align (Visually Up-Up)
    (0.0, np.pi),  # L1 Down, L2 World Up (Visually Down-Up)
    (3 * np.pi / 4, np.pi / 4),  # L1 at 3pi/4, L2 World Up
    (-3 * np.pi / 4, -np.pi / 4)  # L1 at -3pi/4, L2 World Up
]
MAX_TORQUE = 20.0
ABNORMAL_ACCELERATION_THRESHOLD_ENV = 1000.0


class PendubotEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, render_mode=None):
        super().__init__()

        self.dt = 0.02
        self.max_episode_seconds = 5.0
        self.max_episode_steps = int(self.max_episode_seconds / self.dt)
        self.current_step = 0
        self.simulation_time = 0.0

        self.m1 = 0.8
        self.l1 = 1.0
        self.m2 = 0.2
        self.l2 = 0.5
        self.lc1 = self.l1  # Assuming I1 calculation uses this for rod about end
        self.I1 = (1 / 3) * self.m1 * self.l1 ** 2
        self.lc2 = self.l2 / 2.0
        self.I2 = (1 / 12) * self.m2 * self.l2 ** 2

        # Velocity limits can be adjusted based on expected operational range
        vel_limit = 25.0
        obs_low = np.array([-1.0] * 4 + [-vel_limit] * 2 + [-1.0] * 4, dtype=np.float32)
        obs_high = np.array([1.0] * 4 + [vel_limit] * 2 + [1.0] * 4, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: continuous torque tau1
        self.action_space = spaces.Box(low=-MAX_TORQUE, high=MAX_TORQUE, shape=(1,), dtype=np.float32)

        self.state = None  # [q1, q2, dq1, dq2] (standard convention)
        self.target_state_angles = None  # [target_q1, target_q2] (standard convention)

        self.render_mode = render_mode
        self.viewer = None
        self.fig_render = None
        self.ax_render = None
        self.line1_render = None
        self.line2_render = None

    def _randomize_target_and_initial_state(self):
        target_idx = self.np_random.integers(len(VALID_TARGET_CONFIGS))
        self.target_state_angles = np.array(VALID_TARGET_CONFIGS[target_idx], dtype=np.float32)

        perturb_angle = 0.05  # radians
        perturb_vel = 0.01  # rad/s

        q1_init = self.target_state_angles[0] + self.np_random.uniform(-perturb_angle, perturb_angle)
        q2_init = self.target_state_angles[1] + self.np_random.uniform(-perturb_angle, perturb_angle)
        dq1_init = self.np_random.uniform(-perturb_vel, perturb_vel)
        dq2_init = self.np_random.uniform(-perturb_vel, perturb_vel)
        self.state = np.array([q1_init, q2_init, dq1_init, dq2_init], dtype=np.float32)

    def _get_observation(self):
        if self.state is None or self.target_state_angles is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        q1, q2, dq1, dq2 = self.state
        tq1, tq2 = self.target_state_angles

        obs = np.array([
            np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2),
            np.clip(dq1, -25.0, 25.0), np.clip(dq2, -25.0, 25.0),
            np.cos(tq1), np.sin(tq1), np.cos(tq2), np.sin(tq2)
        ], dtype=np.float32)
        return obs

    def _calculate_reward(self, action_tau1):
        q1, q2, dq1, dq2 = self.state
        tq1, tq2 = self.target_state_angles
        q1_err_cost = 1.0 - np.cos(q1 - tq1)
        q2_err_cost = 1.0 - np.cos(q2 - tq2)
        angle_penalty = 1.0 * q1_err_cost + 1.5 * q2_err_cost
        velocity_penalty = 0.05 * (dq1 ** 2 + dq2 ** 2)
        action_penalty = 0.001 * (action_tau1 ** 2)
        reward = - (angle_penalty + velocity_penalty + action_penalty)
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._randomize_target_and_initial_state()
        self.current_step = 0
        self.simulation_time = 0.0
        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human": self._render_frame()
        return observation, info

    def step(self, action):
        tau1_action_value = float(action[0])

        def check_acceleration_event_local_env(t, y,
                                               m1_arg, l1_arg, m2_arg, l2_arg,
                                               lc1_arg, lc2_arg, I1_arg, I2_arg,
                                               control_func_for_event, controller_state_for_event):
            q1_s, q2_s, dq1_s, dq2_s = y
            _tau1 = control_func_for_event(t, y, controller_state_for_event)
            s1, c1 = np.sin(q1_s), np.cos(q1_s)
            s2, c2 = np.sin(q2_s), np.cos(q2_s)
            s12 = np.sin(q1_s + q2_s)
            m11 = m1_arg * lc1_arg ** 2 + m2_arg * (
                        l1_arg ** 2 + lc2_arg ** 2 + 2 * l1_arg * lc2_arg * c2) + I1_arg + I2_arg
            m12 = m2_arg * (lc2_arg ** 2 + l1_arg * lc2_arg * c2) + I2_arg
            m22 = m2_arg * lc2_arg ** 2 + I2_arg
            M_event = np.array([[m11, m12], [m12, m22]])
            c1t = -m2_arg * l1_arg * lc2_arg * s2 * dq2_s ** 2 - 2 * m2_arg * l1_arg * lc2_arg * s2 * dq1_s * dq2_s
            c2t = m2_arg * l1_arg * lc2_arg * s2 * dq1_s ** 2
            C_event = np.array([c1t, c2t])
            g1t = (m1_arg * lc1_arg + m2_arg * l1_arg) * G * s1 + m2_arg * lc2_arg * G * s12
            g2t = m2_arg * lc2_arg * G * s12
            Grav_event = np.array([g1t, g2t])
            Torques_event = np.array([_tau1, 0.0])
            try:
                RHS_event = Torques_event - C_event - Grav_event
                ddq_event = np.linalg.solve(M_event, RHS_event)
            except np.linalg.LinAlgError:
                return -1.0
            return ABNORMAL_ACCELERATION_THRESHOLD_ENV - max(abs(ddq_event[0]), abs(ddq_event[1]))

        check_acceleration_event_local_env.terminal = True
        check_acceleration_event_local_env.direction = -1

        current_step_control_func = lambda t_dyn, x_dyn, cs_dyn: tau1_action_value
        dummy_controller_state = {}

        args_for_dynamics_and_event = (
            self.m1, self.l1, self.m2, self.l2,
            self.lc1, self.lc2, self.I1, self.I2,
            current_step_control_func,
            dummy_controller_state
        )

        terminated = False
        try:
            sol = scipy.integrate.solve_ivp(
                pendubot_dynamics,
                [self.simulation_time, self.simulation_time + self.dt],
                self.state, method='RK45', t_eval=[self.simulation_time + self.dt],
                args=args_for_dynamics_and_event, events=check_acceleration_event_local_env)
        except Exception as e:
            print(f"RL ENV CRITICAL ERROR during solve_ivp: {e}")
            traceback.print_exc()
            terminated = True
            self.state = np.array([self.target_state_angles[0], self.target_state_angles[1], 0, 0], dtype=np.float32)

        if not terminated and sol.status == 1:
            if sol.t_events and sol.t_events[0].size > 0: terminated = True
        if not terminated and sol.status != 0:
            print(f"RL ENV Warning: ODE solver failed (Status {sol.status}) - {sol.message}")
            terminated = True
        if not terminated and sol.y.shape[1] > 0:
            self.state = sol.y[:, -1].astype(np.float32)
        elif not terminated:
            terminated = True

        if not np.all(np.isfinite(self.state)):
            print(f"RL ENV WARNING: Invalid state after step: {self.state}")
            terminated = True
            self.state = np.nan_to_num(self.state, nan=0.0, posinf=self.observation_space.high[4],
                                       neginf=self.observation_space.low[4])

        self.simulation_time += self.dt
        self.current_step += 1

        if not terminated:
            q1, q2, _, _ = self.state
            q1_err = (q1 - self.target_state_angles[0] + np.pi) % (2 * np.pi) - np.pi
            q2_err = (q2 - self.target_state_angles[1] + np.pi) % (2 * np.pi) - np.pi
            if abs(q2_err) > np.pi * 0.7 or abs(q1_err) > np.pi: terminated = True

        truncated = self.current_step >= self.max_episode_steps
        reward = self._calculate_reward(tau1_action_value)
        if terminated: reward -= 100.0

        observation = self._get_observation()
        info = self._get_info()
        if self.render_mode == "human": self._render_frame()
        return observation, reward, terminated, truncated, info

    def _get_info(self):
        # Return fixed params for info if needed
        return {"simulation_time": self.simulation_time, "target_angles": self.target_state_angles,
                "m1": self.m1, "l1": self.l1, "m2": self.m2, "l2": self.l2}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame(is_rgb_array=True)
        elif self.render_mode == "human":
            self._render_frame()

    def _render_frame(self, is_rgb_array=False):
        if self.fig_render is None and (self.render_mode == "human" or is_rgb_array):
            import matplotlib.pyplot as plt
            plt.ioff()
            self.fig_render = plt.Figure(figsize=(5, 4))
            self.ax_render = self.fig_render.add_subplot(111, aspect='equal')
            self.line1_render, = self.ax_render.plot([], [], 'o-', lw=2, color='blue')
            self.line2_render, = self.ax_render.plot([], [], 'o-', lw=2, color='red')
            self.ax_render.grid(True)
            max_len_render = self.l1 + self.l2 + 0.5
            self.ax_render.set_xlim(-max_len_render, max_len_render)
            self.ax_render.set_ylim(-max_len_render, max_len_render)
            if self.render_mode == "human": plt.ion()
            plt.show(block=False)
        if self.state is None: return None
        q1, q2 = self.state[0], self.state[1]
        x0, y0 = 0, 0
        x1_vis = self.l1 * np.sin(q1)
        y1_vis = -self.l1 * np.cos(q1)
        x2_vis = x1_vis + self.l2 * np.sin(q1 + q2)
        y2_vis = y1_vis - self.l2 * np.cos(q1 + q2)
        self.line1_render.set_data([x0, x1_vis], [y0, y1_vis])
        self.line2_render.set_data([x1_vis, x2_vis], [y1_vis, y2_vis])
        if self.render_mode == "human":
            self.fig_render.canvas.draw_idle()
            self.fig_render.canvas.flush_events()
        elif is_rgb_array:
            self.fig_render.canvas.draw()
            image = np.frombuffer(self.fig_render.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig_render.canvas.get_width_height()[::-1] + (3,))
            return image
        return None

    def close(self):
        if self.fig_render is not None and self.render_mode == "human":
            import matplotlib.pyplot as plt
            plt.close(self.fig_render)
            self.fig_render = None
            self.ax_render = None
            self.viewer = None