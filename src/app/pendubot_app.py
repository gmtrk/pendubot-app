import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.integrate
import matplotlib

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None  # Placeholder if not installed, GUI will show error
    print("Warning: stable-baselines3 not found. PPO RL Agent will not be available.")


matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traceback

from src.core_logic.pendubot_dynamics import pendubot_dynamics, G
from src.core_logic.controllers import (control_none, control_pid, control_lqr,
                         calculate_lqr_gain, DEFAULT_CONTROLLER_PARAMS,
                         calculate_equilibrium_torque)
from gui_components import (setup_visualization_area, setup_plotting_area,
                            setup_control_panel)


class PendubotApp:
    ABNORMAL_ACCELERATION_THRESHOLD = 1000.0

    def __init__(self, root):
        self.root = root
        self.root.title("Pendubot Simulation")
        self.root.geometry("1200x700")

        self.m1 = 0.8
        self.l1 = 1.0
        self.m2 = 0.2
        self.l2 = 0.5
        self.lc1 = self.l1
        self.lc2 = self.l2 / 2.0
        self.I1 = (1 / 3) * self.m1 * self.l1 ** 2
        self.I2 = (1 / 12) * self.m2 * self.l2 ** 2
        self.dt = 0.02
        self.simulation_time = 0.0
        self.state = np.zeros(4)
        self.time_history = []
        self.q1_history = []
        self.q2_history = []
        self.dq1_history = []
        self.dq2_history = []
        self.history_length = 250
        self.plot_time_window_start = 0.0
        self.plot_time_window_duration = 5.0
        self.control_method_var = tk.StringVar(value="LQR")
        self.control_func = control_none
        self.current_controller_state = {}
        self.target_state = np.zeros(4)

        # Initialize all PID gain StringVars and corresponding float _val attributes
        pid_defaults = DEFAULT_CONTROLLER_PARAMS['pid']
        self.kp_q1_val = pid_defaults['Kp_q1']
        self.ki_q1_val = pid_defaults['Ki_q1']
        self.kd_q1_val = pid_defaults['Kd_q1']
        self.kp_q2_val = pid_defaults['Kp_q2']
        self.ki_q2_val = pid_defaults['Ki_q2']
        self.kd_q2_val = pid_defaults['Kd_q2']

        self.kp_q1_var = tk.StringVar(value=str(self.kp_q1_val))
        self.ki_q1_var = tk.StringVar(value=str(self.ki_q1_val))
        self.kd_q1_var = tk.StringVar(value=str(self.kd_q1_val))
        self.kp_q2_var = tk.StringVar(value=str(self.kp_q2_val))
        self.ki_q2_var = tk.StringVar(value=str(self.ki_q2_val))
        self.kd_q2_var = tk.StringVar(value=str(self.kd_q2_val))
        # Renaming old vars for clarity if they were used by gui_components.py (now they use direct _q2)
        self.kp_var = self.kp_q2_var
        self.ki_var = self.ki_q2_var
        self.kd_var = self.kd_q2_var

        self.ppo_model = None  # Placeholder for the loaded PPO model

        self.current_tau1 = 0.0
        self.tau1_display_var = tk.StringVar(value="0.00 Nm")

        self.anim = None
        self.running = False
        self._applying_params = False
        self.pid_tune_frame = None
        self.setup_gui()
        self.update_pid_widgets_visibility()
        self.apply_parameters()  # Initial run

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.columnconfigure(0, weight=3)
        main_frame.rowconfigure(0, weight=1)
        right_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)
        setup_visualization_area(left_frame, self)
        setup_plotting_area(left_frame, self)
        setup_control_panel(right_frame, self)
        left_frame.rowconfigure(0, weight=2)
        left_frame.rowconfigure(1, weight=3)
        left_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready. Select target/control.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def update_pid_widgets_visibility(self):
        if hasattr(self, 'pid_tune_frame') and self.pid_tune_frame:
            if self.control_method_var and self.control_method_var.get() == "PID":
                self.pid_tune_frame.grid()
            else:
                self.pid_tune_frame.grid_remove()

    def apply_parameters_pid_tune(self):
        if self.control_method_var and self.control_method_var.get() == "PID":
            if not self._applying_params:
                self.apply_parameters()

    def adjust_pid_param(self, param_key_full, amount):
        var_to_update_str = param_key_full + "_var"  # e.g., "kp_q1_var"
        val_attr_str = param_key_full + "_val"  # e.g., "kp_q1_val"

        if not (hasattr(self, var_to_update_str) and hasattr(self, val_attr_str)):
            print(f"Error: Unknown PID parameter key for adjustment: {param_key_full}")
            return

        var_to_update = getattr(self, var_to_update_str)

        try:
            current_value = float(var_to_update.get())
            new_value = round(current_value + amount, 4)
            if new_value < 0.0:
                new_value = 0.0

            var_to_update.set(str(new_value))
            setattr(self, val_attr_str, new_value)

            self.apply_parameters_pid_tune()
        except ValueError:
            messagebox.showerror("PID Tuning Error", f"Current value for {param_key_full.upper()} is invalid.")
            # Restore from internal float value
            var_to_update.set(str(getattr(self, val_attr_str)))

    def get_cartesian_coords(self, q1, q2):
        x0, y0 = 0, 0
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1 + q2)
        return x0, y0, x1, y1, x2, y2

    def _get_ppo_observation(self):
        """ Constructs the observation vector for the PPO agent.
            Physical parameters are now fixed and NOT part of the observation.
        """
        q1, q2, dq1, dq2 = self.state
        tq1, tq2 = self.target_state[0], self.target_state[1]

        obs = np.array([
            np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), dq1, dq2,
            np.cos(tq1), np.sin(tq1), np.cos(tq2), np.sin(tq2)
        ], dtype=np.float32)
        return obs

    def control_ppo(self, t, x, controller_state):
        if self.ppo_model is None:
            print("PPO model not loaded, returning zero torque.")  # Reduce noise
            return 0.0

        observation = self._get_ppo_observation()  # self.state is current, self.target_state for target
        action, _states = self.ppo_model.predict(observation, deterministic=True)
        return float(action[0])

    def apply_parameters(self):
        if self._applying_params:
            return
        self._applying_params = True
        if self.anim and self.running:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
            self.anim = None
        self.running = False
        try:
            self.m1 = float(self.m1_var.get())
            self.l1 = float(self.l1_var.get())
            self.m2 = float(self.m2_var.get())
            self.l2 = float(self.l2_var.get())
            if self.m1 <= 0 or self.l1 <= 0 or self.m2 <= 0 or self.l2 <= 0:
                raise ValueError("Params > 0.")
            self.lc1 = self.l1
            self.lc2 = self.l2 / 2.0
            self.I1 = (1 / 3) * self.m1 * self.l1 ** 2
            self.I2 = (1 / 12) * self.m2 * self.l2 ** 2

            start_pos = self.start_pos_var.get()
            q1_i, q2_i, dq1_i, dq2_i = 0, 0, 0, 0
            tq1, tq2 = 0, 0
            p = 0.05
            if start_pos == "Target (pi, 0) [L1 Up, L2 Align]":
                q1_i, q2_i = np.pi + p, 0.0 + p
                tq1, tq2 = np.pi, 0.0
            elif start_pos == "Target (0, pi) [L1 Down, L2 World Up]":
                q1_i, q2_i = 0.0 + p, np.pi - p
                tq1, tq2 = 0.0, np.pi
            elif start_pos == "Target (3pi/4, pi/4) [L2 World Up]":
                q1_i, q2_i = (3 * np.pi / 4) + p, (np.pi / 4) - p
                tq1, tq2 = (3 * np.pi / 4), (np.pi / 4)
            elif start_pos == "Target (-3pi/4, -pi/4) [L2 World Up]":
                q1_a = -3 * np.pi / 4
                q1_i, q2_i = q1_a + p, (-np.pi / 4) - p
                tq1, tq2 = q1_a, (-np.pi / 4)
            elif start_pos == "Target (0, 0) [Fully Down]":
                q1_i, q2_i = 0.0 + p, 0.0 + p
                tq1, tq2 = 0.0, 0.0
            else:
                q1_i, q2_i = np.pi + p, 0.0 + p
                tq1, tq2 = np.pi, 0.0
            self.state = np.array([q1_i, q2_i, dq1_i, dq2_i])
            self.target_state = np.array([tq1, tq2, 0, 0])

            self.simulation_time = 0.0
            self.plot_time_window_start = 0.0
            self.time_history = []
            self.q1_history = []
            self.q2_history = []
            self.dq1_history = []
            self.dq2_history = []
            method = self.control_method_var.get()
            self.current_controller_state = {}
            u_eq = calculate_equilibrium_torque(tq1, tq2, self.m1, self.l1, self.m2, self.l2, self.lc1, self.lc2)

            if method == "PID":
                self.control_func = control_pid
                try:  # Read all 6 gains
                    kp1 = float(self.kp_q1_var.get())
                    ki1 = float(self.ki_q1_var.get())
                    kd1 = float(self.kd_q1_var.get())
                    kp2 = float(self.kp_q2_var.get())
                    ki2 = float(self.ki_q2_var.get())
                    kd2 = float(self.kd_q2_var.get())
                    self.kp_q1_val, self.ki_q1_val, self.kd_q1_val = kp1, ki1, kd1
                    self.kp_q2_val, self.ki_q2_val, self.kd_q2_val = kp2, ki2, kd2
                except ValueError:
                    messagebox.showerror("Invalid Input", "PID gains invalid.")
                    self._applying_params = False
                    return

                self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
                # Set all 6 gains in controller state
                self.current_controller_state['Kp_q1'] = self.kp_q1_val
                self.current_controller_state['Ki_q1'] = self.ki_q1_val
                self.current_controller_state['Kd_q1'] = self.kd_q1_val
                self.current_controller_state['Kp_q2'] = self.kp_q2_val
                self.current_controller_state['Ki_q2'] = self.ki_q2_val
                self.current_controller_state['Kd_q2'] = self.kd_q2_val

                self.current_controller_state['dt'] = self.dt
                self.current_controller_state['integral_error_q1'] = 0.0  # Initialize both integrals
                self.current_controller_state['integral_error_q2'] = 0.0
                self.current_controller_state['target_state'] = self.target_state  # Pass full target state
                self.current_controller_state['u_eq'] = u_eq
                self.status_var.set(
                    f"State PID+FF (Kpq1={kp1:.1f},Kiq1={ki1:.1f},Kdq1={kd1:.1f} | Kpq2={kp2:.1f},Kiq2={ki2:.1f},Kdq2={kd2:.1f})")
            elif method == "LQR":
                self.control_func = control_lqr
                Q = DEFAULT_CONTROLLER_PARAMS['lqr']['Q']
                R = DEFAULT_CONTROLLER_PARAMS['lqr']['R']
                K, u_eq_lqr = calculate_lqr_gain(self.m1, self.l1, self.m2, self.l2, self.lc1, self.lc2, self.I1,
                                                 self.I2, Q, R, q1_eq=tq1, q2_eq=tq2)
                if K is not None:
                    self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['lqr'].copy()
                    self.current_controller_state['K'] = K
                    self.current_controller_state['target_state'] = self.target_state
                    self.current_controller_state['u_eq'] = u_eq_lqr
                    self.status_var.set(f"LQR (Tq1={tq1:.2f},Tq2={tq2:.2f})")
                else:
                    self.control_func = control_none
                    self.control_method_var.set("None")
                    self.status_var.set("LQR Fail")
            elif method == "PPO RL Agent":
                if PPO is None:
                    messagebox.showerror("Error", "Stable Baselines3 library not found. Cannot use PPO agent.")
                    self.control_func = control_none
                    self.control_method_var.set("None")  # Revert selection
                    self.status_var.set("PPO Failed: Library missing.")
                else:
                    try:
                        model_path = "../../data/ppo_pendubot_model.zip"  # Assuming it's in the same directory
                        if not os.path.exists(model_path):
                            # Try looking in the log directory as a fallback
                            log_model_path = os.path.join("../../data/ppo_pendubot_logs", "ppo_pendubot_model.zip")
                            if os.path.exists(log_model_path):
                                model_path = log_model_path
                            else:
                                messagebox.showerror("Error",
                                                     f"PPO model file '{model_path}' (and in logs) not found. Please train it first.")
                                self.control_func = control_none
                                self.control_method_var.set("None")
                                self.status_var.set("PPO Failed: Model not found.")
                                self.ppo_model = None  # Ensure it's cleared
                                # Must re-assign self._applying_params = False and return
                                self._applying_params = False
                                return

                        if self.ppo_model is None:  # Load only if not already loaded or path changed
                            print(f"Loading PPO model from: {model_path}")
                            self.ppo_model = PPO.load(model_path)
                        self.control_func = self.control_ppo  # Assign the method
                        self.status_var.set("Running PPO RL Agent.")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to load PPO model: {e}")
                        traceback.print_exc()
                        self.control_func = control_none
                        self.control_method_var.set("None")
                        self.status_var.set("PPO Failed: Model load error.")
                        self.ppo_model = None
            else:  # None
                self.control_func = control_none
                self.status_var.set("Running with no control.")
                self.ppo_model = None

            if self.anim is None:
                self.start_animation()
            else:
                self.init_animation()
                self.start_animation()
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.status_var.set(f"Error: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Unexpected error: {e}")
            self.status_var.set(f"Error: {e}")
            traceback.print_exc()
        finally:
            self._applying_params = False

    def show_abnormal_acceleration_popup(self):
        self.running = False
        if self.anim:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        popup = tk.Toplevel(self.root)
        popup.title("Simulation Error")
        popup.geometry("350x100")
        popup.grab_set()
        popup.focus_set()
        popup.transient(self.root)
        label = ttk.Label(popup, text="Oops, the robot reached abnormal acceleration and broke :(", wraplength=330,
                          justify=tk.CENTER)
        label.pack(pady=10, padx=10)

        def restart_and_close():
            popup.destroy()
            self.apply_parameters()

        restart_button = ttk.Button(popup, text="Restart", command=restart_and_close)
        restart_button.pack(pady=5)
        self.root.wait_window(popup)

    def simulation_step(self):
        if not self.running:
            return
        try:
            # --- Calculate and store current_tau1 for display ---
            if self.control_func and hasattr(self,
                                             'current_controller_state') and self.current_controller_state is not None:
                # --- ADD DEBUG PRINT BEFORE CALLING self.control_func ---
                # print(f"DEBUG sim_step: About to call self.control_func: {self.control_func.__name__ if hasattr(self.control_func, '__name__') else 'Unknown'}")
                self.current_tau1 = self.control_func(self.simulation_time, self.state, self.current_controller_state)
                # --- ADD DEBUG PRINT AFTER CALLING self.control_func ---
                # print(f"DEBUG sim_step: self.current_tau1 set to: {self.current_tau1} by {self.control_func.__name__ if hasattr(self.control_func, '__name__') else 'Unknown'}")
            else:
                self.current_tau1 = 0.0
                # print(f"DEBUG sim_step: self.current_tau1 set to 0.0 due to condition fail.")

            def check_acceleration_event_local(t, y, m1_a, l1_a, m2_a, l2_a, lc1_a, lc2_a, I1_a, I2_a, ctrl_f_ref,
                                               ctrl_s_ref):
                q1s, q2s, dq1s, dq2s = y
                tau1 = ctrl_f_ref(t, y, ctrl_s_ref)
                s1, c1 = np.sin(q1s), np.cos(q1s)
                s2, c2 = np.sin(q2s), np.cos(q2s)
                s12 = np.sin(q1s + q2s)
                m11 = m1_a * lc1_a ** 2 + m2_a * (l1_a ** 2 + lc2_a ** 2 + 2 * l1_a * lc2_a * c2) + I1_a + I2_a
                m12 = m2_a * (lc2_a ** 2 + l1_a * lc2_a * c2) + I2_a
                m22 = m2_a * lc2_a ** 2 + I2_a
                M = np.array([[m11, m12], [m12, m22]])
                c1t = -m2_a * l1_a * lc2_a * s2 * dq2s ** 2 - 2 * m2_a * l1_a * lc2_a * s2 * dq1s * dq2s
                c2t = m2_a * l1_a * lc2_a * s2 * dq1s ** 2
                C = np.array([c1t, c2t])
                g1t = (m1_a * lc1_a + m2_a * l1_a) * G * s1 + m2_a * lc2_a * G * s12
                g2t = m2_a * lc2_a * G * s12
                Grav = np.array([g1t, g2t])
                Torques = np.array([tau1, 0.0])
                try:
                    RHS = Torques - C - Grav
                    ddq = np.linalg.solve(M, RHS)
                except np.linalg.LinAlgError:
                    return -1.0
                return self.ABNORMAL_ACCELERATION_THRESHOLD - max(abs(ddq[0]), abs(ddq[1]))

            check_acceleration_event_local.terminal = True
            check_acceleration_event_local.direction = -1
            common_args = (self.m1, self.l1, self.m2, self.l2, self.lc1, self.lc2, self.I1, self.I2, self.control_func,
                           self.current_controller_state)
            sol = scipy.integrate.solve_ivp(pendubot_dynamics, [self.simulation_time, self.simulation_time + self.dt],
                                            self.state, method='RK45', t_eval=[self.simulation_time + self.dt],
                                            args=common_args, events=check_acceleration_event_local)
            if sol.status == 1:
                if sol.t_events and sol.t_events[0].size > 0:
                    self.show_abnormal_acceleration_popup()
                    return
            if sol.status != 0 and sol.status != 1:
                print(f"Solver fail (Status {sol.status}): {sol.message}")
            else:
                if sol.y.shape[1] > 0:
                    self.state = sol.y[:, -1]
            if not np.all(np.isfinite(self.state)):
                print(f"WARNING: Invalid state t={self.simulation_time}: {self.state}")
                self.running = False
                return
            self.simulation_time += self.dt
            self.time_history.append(self.simulation_time)
            self.q1_history.append(self.state[0])
            self.q2_history.append(self.state[1])
            self.dq1_history.append(self.state[2])
            self.dq2_history.append(self.state[3])
            while len(self.time_history) > self.history_length + 50:
                self.time_history.pop(0)
                self.q1_history.pop(0)
                self.q2_history.pop(0)
                self.dq1_history.pop(0)
                self.dq2_history.pop(0)
        except Exception as e:
            print(f"Error during simulation step: {e}")
            traceback.print_exc()
            self.running = False

    def init_animation(self):
        if hasattr(self, 'line1') and self.line1:
            self.line1.set_data([], [])
        if hasattr(self, 'line2') and self.line2:
            self.line2.set_data([], [])
        if hasattr(self, 'plot_q1') and self.plot_q1:
            self.plot_q1.set_data([], [])
        if hasattr(self, 'plot_q2') and self.plot_q2:
            self.plot_q2.set_data([], [])
        if hasattr(self, 'plot_dq1') and self.plot_dq1:
            self.plot_dq1.set_data([], [])
        if hasattr(self, 'plot_dq2') and self.plot_dq2:
            self.plot_dq2.set_data([], [])
        self.update_visualization(initial_draw=True)
        self.update_plots(initial_draw=True)
        artists = []
        if hasattr(self, 'line1') and self.line1:
            artists.append(self.line1)
        if hasattr(self, 'line2') and self.line2:
            artists.append(self.line2)
        if hasattr(self, 'plot_q1') and self.plot_q1:
            artists.append(self.plot_q1)
        if hasattr(self, 'plot_q2') and self.plot_q2:
            artists.append(self.plot_q2)
        if hasattr(self, 'plot_dq1') and self.plot_dq1:
            artists.append(self.plot_dq1)
        if hasattr(self, 'plot_dq2') and self.plot_dq2:
            artists.append(self.plot_dq2)
        return artists

    def update_visualization(self, initial_draw=False):
        if not hasattr(self, 'line1') or not hasattr(self, 'ax_vis'):
            return
        if not self.running and not initial_draw:
            return
        q1, q2 = self.state[0], self.state[1] if self.state is not None and len(self.state) >= 2 else (0, 0)
        try:
            x0, y0, x1, y1, x2, y2 = self.get_cartesian_coords(q1, q2)
            self.line1.set_data([x0, x1], [y0, y1])
            self.line2.set_data([x1, x2], [y1, y2])
            if initial_draw:
                max_len = (self.l1 if hasattr(self, 'l1') else 1.0) + (self.l2 if hasattr(self, 'l2') else 1.0) + 0.5
                self.ax_vis.set_xlim(-max_len, max_len)
                self.ax_vis.set_ylim(-max_len, max_len)
                self.ax_vis.figure.canvas.draw()
        except Exception as e:
            print(f"Error in update_visualization: {e}")
            traceback.print_exc()
            self.running = False

    def update_plots(self, initial_draw=False):
        if not hasattr(self, 'plot_q1') or not hasattr(self, 'axs_plots'):
            return
        if not self.running and not initial_draw:
            return
        if not self.time_history and not initial_draw:
            return
        try:
            time_hist = self.time_history if self.time_history else [0.0]
            q1_hist = self.q1_history if self.q1_history else [0.0]
            q2_hist = self.q2_history if self.q2_history else [0.0]
            dq1_hist = self.dq1_history if self.dq1_history else [0.0]
            dq2_hist = self.dq2_history if self.dq2_history else [0.0]
            t = time_hist[-self.history_length:]
            q1 = q1_hist[-self.history_length:]
            q2 = q2_hist[-self.history_length:]
            dq1 = dq1_hist[-self.history_length:]
            dq2 = dq2_hist[-self.history_length:]
        except Exception as e:
            print(f"Error accessing history: {e}")
            self.running = False
            return
        plot_lines = [self.plot_q1, self.plot_q2, self.plot_dq1, self.plot_dq2]
        plot_data_y = [q1, q2, dq1, dq2]
        is_effectively_empty = len(t) == 0 or (len(t) == 1 and t[0] == 0.0 and not any(
            y_data[0] for y_data in plot_data_y if len(y_data) > 0 and not np.isnan(y_data[0])))
        if initial_draw or is_effectively_empty:
            for line in plot_lines:
                line.set_data([], [])
        else:
            for i, line in enumerate(plot_lines):
                line.set_data(t, plot_data_y[i])
        try:
            needs_canvas_draw = initial_draw
            for i in range(4):
                ax = self.axs_plots[i]
                old_ylim = ax.get_ylim()
                ax.relim()
                ax.autoscale_view(True, False, True)
                if old_ylim != ax.get_ylim():
                    needs_canvas_draw = True
            current_plot_xlim_end = self.plot_time_window_start + self.plot_time_window_duration
            new_xlim_start = self.plot_time_window_start
            new_xlim_end = current_plot_xlim_end
            if self.simulation_time >= current_plot_xlim_end:
                self.plot_time_window_start = np.floor(
                    self.simulation_time / self.plot_time_window_duration) * self.plot_time_window_duration
                new_xlim_start = self.plot_time_window_start
                new_xlim_end = self.plot_time_window_start + self.plot_time_window_duration
            current_actual_xlim = self.axs_plots[3].get_xlim()
            if initial_draw or current_actual_xlim != (new_xlim_start, new_xlim_end):
                for ax_plot in self.axs_plots:
                    ax_plot.set_xlim(new_xlim_start, new_xlim_end)
                needs_canvas_draw = True
            if needs_canvas_draw:
                self.canvas_plots.draw()
        except Exception as e:
            print(f"Error updating plot limits: {e}")
            traceback.print_exc()
            self.running = False

    def animation_frame(self, frame):
        if not self.running:
            if self.anim:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass
            return []
        self.simulation_step()
        updated_artists = []
        if self.running:
            self.update_visualization()
            self.update_plots()
            if hasattr(self, 'tau1_display_var'):  # Check if it exists
                self.tau1_display_var.set(f"{self.current_tau1:.2f} Nm")
            if hasattr(self, 'line1') and self.line1:
                updated_artists.append(self.line1)
            if hasattr(self, 'line2') and self.line2:
                updated_artists.append(self.line2)
            if hasattr(self, 'plot_q1') and self.plot_q1:
                updated_artists.append(self.plot_q1)
            if hasattr(self, 'plot_q2') and self.plot_q2:
                updated_artists.append(self.plot_q2)
            if hasattr(self, 'plot_dq1') and self.plot_dq1:
                updated_artists.append(self.plot_dq1)
            if hasattr(self, 'plot_dq2') and self.plot_dq2:
                updated_artists.append(self.plot_dq2)
        if not self.running and self.anim:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        return updated_artists

    def start_animation(self):
        if self.anim:
            try:
                self.anim.event_source.stop()
            except Exception:
                pass
        self.running = True
        interval_ms = max(1, int(self.dt * 1000))
        if not hasattr(self, 'fig_vis') or not hasattr(self, 'canvas_vis'):
            print("ERROR: Vis figure not init.")
            self.running = False
            return
        if not hasattr(self, 'fig_plots') or not hasattr(self, 'canvas_plots'):
            print("ERROR: Plot figure/canvas not init.")
            self.running = False
            return
        self.anim = FuncAnimation(self.fig_vis, self.animation_frame, init_func=self.init_animation, frames=None,
                                  interval=interval_ms, blit=True, repeat=True, cache_frame_data=False, save_count=0)


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PendubotApp(root)


    def on_closing():
        app.running = False
        if app.anim:
            try:
                app.anim.event_source.stop()
            except Exception:
                pass
        root.quit()
        root.destroy()


    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()