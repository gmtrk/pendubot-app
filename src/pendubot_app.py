# pendubot_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.integrate
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traceback

from pendubot_dynamics import pendubot_dynamics
from controllers import (control_none, control_pid, control_lqr,
                         calculate_lqr_gain, DEFAULT_CONTROLLER_PARAMS,
                         calculate_equilibrium_torque)
from gui_components import (setup_visualization_area, setup_plotting_area,
                            setup_control_panel)

class PendubotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pendubot Simulation")
        self.root.geometry("1200x700")

        # --- Default Parameters & State ---
        self.m1 = 0.8; self.l1 = 1.0; self.m2 = 0.2; self.l2 = 0.5
        self.lc1 = self.l1; self.lc2 = self.l2 / 2.0
        self.I1 = (1/3) * self.m1 * self.l1**2
        self.I2 = (1/12) * self.m2 * self.l2**2

        self.dt = 0.02
        self.simulation_time = 0.0
        self.state = np.zeros(4) # Standard convention

        self.time_history = []; self.q1_history = []; self.q2_history = []
        self.dq1_history = []; self.dq2_history = []
        self.history_length = 250

        # --- Controller & Target ---
        self.control_method_var = None # Set by GUI setup
        self.control_func = control_none
        self.current_controller_state = {}
        self.target_state = np.zeros(4) # Standard convention

        # --- PID Tuning Variables ---
        self.kp_val = DEFAULT_CONTROLLER_PARAMS['pid']['Kp']
        self.ki_val = DEFAULT_CONTROLLER_PARAMS['pid']['Ki']
        self.kd_val = DEFAULT_CONTROLLER_PARAMS['pid']['Kd']
        self.kp_var = tk.StringVar(value=str(self.kp_val))
        self.ki_var = tk.StringVar(value=str(self.ki_val))
        self.kd_var = tk.StringVar(value=str(self.kd_val))

        # Animation control
        self.anim = None
        self.running = False

        # --- GUI Setup ---
        self.pid_tune_frame = None # Placeholder for the PID tuning frame
        self.setup_gui()

        # --- Finalize Setup ---
        self.update_pid_widgets_visibility()
        # apply_parameters() called by GUI radiobutton/spinbox commands

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.columnconfigure(0, weight=3); main_frame.rowconfigure(0, weight=1)
        right_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)

        setup_visualization_area(left_frame, self)
        setup_plotting_area(left_frame, self)
        setup_control_panel(right_frame, self) # Creates PID frame and vars

        left_frame.rowconfigure(0, weight=2); left_frame.rowconfigure(1, weight=3)
        left_frame.columnconfigure(0, weight=1)
        self.status_var = tk.StringVar(value="Ready. Select target/control.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

    def update_pid_widgets_visibility(self):
        if hasattr(self, 'pid_tune_frame') and self.pid_tune_frame:
            if self.control_method_var and self.control_method_var.get() == "PID":
                self.pid_tune_frame.grid() # Show the frame
            else:
                self.pid_tune_frame.grid_remove() # Hide the frame

    def apply_parameters_pid_tune(self):
        if self.control_method_var and self.control_method_var.get() == "PID":
            self.apply_parameters()

    def get_cartesian_coords(self, q1, q2):
        x0, y0 = 0, 0
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1)
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1 + q2)
        return x0, y0, x1, y1, x2, y2

    def apply_parameters(self):
        if hasattr(self, '_applying_params') and self._applying_params:
             return
        self._applying_params = True

        # --- Stop animation - Corrected Syntax ---
        if self.anim and self.running:
            try:
                self.anim.event_source.stop()
            except Exception as e:
                print(f"DEBUG: Error stopping animation: {e}")
            self.anim = None
        self.running = False
        # --- End Stop animation ---

        try:
            self.m1 = float(self.m1_var.get())
            self.l1 = float(self.l1_var.get())
            self.m2 = float(self.m2_var.get())
            self.l2 = float(self.l2_var.get())
            if self.m1 <= 0 or self.l1 <= 0 or self.m2 <= 0 or self.l2 <= 0:
                raise ValueError("Masses and lengths must be positive.")
            self.lc1 = self.l1; self.lc2 = self.l2 / 2.0
            self.I1 = (1/3) * self.m1 * self.l1**2
            self.I2 = (1/12) * self.m2 * self.l2**2
            # print(f"INFO: Using Parameters m1={self.m1}, l1={self.l1}, m2={self.m2}, l2={self.l2}")

            start_pos = self.start_pos_var.get()
            q1_init, q2_init, dq1_init, dq2_init = 0.0, 0.0, 0.0, 0.0
            target_q1, target_q2 = 0.0, 0.0
            perturb = 0.05

            # Define initial and target states based on GUI selection
            if start_pos == "Target (pi, 0) [L1 Up, L2 Align]":
                q1_init, q2_init = np.pi + perturb, 0.0 + perturb
                target_q1, target_q2 = np.pi, 0.0
            elif start_pos == "Target (0, pi) [L1 Down, L2 World Up]":
                q1_init, q2_init = 0.0 + perturb, np.pi - perturb
                target_q1, target_q2 = 0.0, np.pi
            elif start_pos == "Target (3pi/4, pi/4) [L2 World Up]":
                q1_init, q2_init = (3*np.pi/4) + perturb, (np.pi/4) - perturb
                target_q1, target_q2 = (3*np.pi/4), (np.pi/4)
            elif start_pos == "Target (-3pi/4, -pi/4) [L2 World Up]":
                q1_init_angle = -3*np.pi/4
                q1_init, q2_init = q1_init_angle + perturb, (-np.pi/4) - perturb
                target_q1, target_q2 = q1_init_angle, (-np.pi/4)
            elif start_pos == "Target (0, 0) [Fully Down]":
                 q1_init, q2_init = 0.0 + perturb, 0.0 + perturb
                 target_q1, target_q2 = 0.0, 0.0
            else: # Default case
                q1_init, q2_init = np.pi + perturb, 0.0 + perturb
                target_q1, target_q2 = np.pi, 0.0

            self.state = np.array([q1_init, q2_init, dq1_init, dq2_init])
            self.target_state = np.array([target_q1, target_q2, 0.0, 0.0])
            # print(f"INFO: Target state (q1_down, q2_rel) set to: {self.target_state}")

            self.simulation_time = 0.0
            self.time_history = []; self.q1_history = []; self.q2_history = []
            self.dq1_history = []; self.dq2_history = []

            method = self.control_method_var.get()
            self.current_controller_state = {}

            u_eq = calculate_equilibrium_torque(target_q1, target_q2,
                                                self.m1, self.l1, self.m2, self.l2,
                                                self.lc1, self.lc2)
            # print(f"INFO: Calculated u_eq = {u_eq:.4f} for target state.")

            if method == "PID":
                self.control_func = control_pid
                try:
                    kp = float(self.kp_var.get()); ki = float(self.ki_var.get()); kd = float(self.kd_var.get())
                    self.kp_val, self.ki_val, self.kd_val = kp, ki, kd # Update stored values
                except ValueError:
                    messagebox.showerror("Invalid Input", "PID gains must be valid numbers.")
                    self._applying_params = False; return

                self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
                self.current_controller_state['Kp'] = self.kp_val
                self.current_controller_state['Ki'] = self.ki_val
                self.current_controller_state['Kd'] = self.kd_val
                self.current_controller_state['dt'] = self.dt
                self.current_controller_state['integral_error'] = 0.0
                self.current_controller_state['target_q2'] = target_q2
                self.current_controller_state['u_eq'] = u_eq
                self.status_var.set(f"Running PID+FF (Kp={kp:.1f}, Ki={ki:.1f}, Kd={kd:.1f}, Targets q2={target_q2:.2f})")

            elif method == "LQR":
                self.control_func = control_lqr
                Q = DEFAULT_CONTROLLER_PARAMS['lqr']['Q']
                R = DEFAULT_CONTROLLER_PARAMS['lqr']['R']
                K, u_eq_lqr = calculate_lqr_gain(self.m1, self.l1, self.m2, self.l2,
                                          self.lc1, self.lc2, self.I1, self.I2,
                                          Q, R, q1_eq=target_q1, q2_eq=target_q2)
                if K is not None:
                    self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['lqr'].copy()
                    self.current_controller_state['K'] = K
                    self.current_controller_state['target_state'] = self.target_state
                    self.current_controller_state['u_eq'] = u_eq_lqr
                    self.status_var.set(f"Running LQR (Targets q1={target_q1:.2f}, q2={target_q2:.2f}).")
                else:
                    self.control_func = control_none; self.control_method_var.set("None")
                    self.status_var.set("LQR Failed. No control applied.")
            else: # None
                self.control_func = control_none
                self.status_var.set("Running with no control.")

            self.update_visualization()
            self.update_plots()
            self.start_animation()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.status_var.set(f"Error: Invalid parameter values. {e}")
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred: {e}")
             self.status_var.set(f"Error: {e}")
             print(f"Detailed Error: {e}", flush=True)
             traceback.print_exc()
        finally:
            self._applying_params = False # Reset flag


    def update_visualization(self):
        if not hasattr(self, 'line1') or not self.running: return
        q1, q2 = self.state[0], self.state[1]
        try:
            x0, y0, x1, y1, x2, y2 = self.get_cartesian_coords(q1, q2)
            self.line1.set_data([x0, x1], [y0, y1])
            self.line2.set_data([x1, x2], [y1, y2])
            max_len = self.l1 + self.l2 + 0.5; current_xlim = self.ax_vis.get_xlim()
            if current_xlim[0] != -max_len or current_xlim[1] != max_len:
                 self.ax_vis.set_xlim(-max_len, max_len); self.ax_vis.set_ylim(-max_len, max_len)
            self.canvas_vis.draw()
        except Exception as e: print(f"Error in update_visualization: {e}"); traceback.print_exc(); self.running = False

    def update_plots(self):
        if not hasattr(self, 'plot_q1') or not self.running or not self.time_history: return
        try:
            time_hist=self.time_history; q1_hist=self.q1_history; q2_hist=self.q2_history
            dq1_hist=self.dq1_history; dq2_hist=self.dq2_history; current_len=len(time_hist)
            if not (current_len == len(q1_hist) == len(q2_hist) == len(dq1_hist) == len(dq2_hist)):
                print("CRITICAL ERROR: History list lengths out of sync!"); self.running = False; return
            t=time_hist[-self.history_length:]; q1=q1_hist[-self.history_length:]
            q2=q2_hist[-self.history_length:]; dq1=dq1_hist[-self.history_length:]; dq2=dq2_hist[-self.history_length:]
        except Exception as e: print(f"Error accessing history: {e}"); self.running = False; return
        if len(t) == 0:
            self.plot_q1.set_data([], []); self.plot_q2.set_data([], []); self.plot_dq1.set_data([], []); self.plot_dq2.set_data([], [])
        else:
            self.plot_q1.set_data(t, q1); self.plot_q2.set_data(t, q2); self.plot_dq1.set_data(t, dq1); self.plot_dq2.set_data(t, dq2)
        try:
            needs_redraw = False
            for i in range(4):
                current_ax = self.axs_plots[i]; old_lim = current_ax.get_ylim()
                current_ax.relim(); current_ax.autoscale_view(True, False, True)
                if old_lim != current_ax.get_ylim(): needs_redraw = True
            if len(t) > 0:
                min_time=t[0]; max_time=t[-1]; time_range = max(1.0, max_time - min_time if max_time > min_time else 1.0)
                xlim_min = max(min_time, max_time - time_range); current_xlim = self.axs_plots[3].get_xlim()
                if current_xlim != (xlim_min, max_time):
                    for ax in self.axs_plots: ax.set_xlim(xlim_min, max_time)
                    needs_redraw = True
            if needs_redraw or len(t) == 0: self.canvas_plots.draw()
        except Exception as e: print(f"Error updating plot limits/drawing: {e}"); traceback.print_exc(); self.running = False

    def simulation_step(self):
        if not self.running: return
        try:
            control_args = (self.m1, self.l1, self.m2, self.l2, self.lc1, self.lc2, self.I1, self.I2, self.control_func, self.current_controller_state)
            sol = scipy.integrate.solve_ivp( pendubot_dynamics, [self.simulation_time, self.simulation_time + self.dt], self.state, method='RK45', t_eval=[self.simulation_time + self.dt], args=control_args )
            if sol.status != 0: self.state = self.state
            else: self.state = sol.y[:, -1]
            if not np.all(np.isfinite(self.state)): print(f"WARNING: Invalid state t={self.simulation_time}: {self.state}"); self.running = False; return
            self.simulation_time += self.dt
            self.time_history.append(self.simulation_time); self.q1_history.append(self.state[0])
            self.q2_history.append(self.state[1]); self.dq1_history.append(self.state[2]); self.dq2_history.append(self.state[3])
            while len(self.time_history) > self.history_length + 50: self.time_history.pop(0); self.q1_history.pop(0); self.q2_history.pop(0); self.dq1_history.pop(0); self.dq2_history.pop(0)
        except Exception as e: print(f"Error during simulation step: {e}"); traceback.print_exc(); self.running = False

    def animation_frame(self, frame):
        if not self.running:
            # --- Stop animation - Corrected Syntax ---
            if self.anim:
                try:
                    self.anim.event_source.stop()
                except Exception:
                    pass # Ignore errors stopping timer
            # --- End Stop animation ---
            return []

        self.simulation_step()

        artists = []
        if self.running: # Check again, simulation_step might have set running=False
            self.update_visualization()
            self.update_plots()
            artists.extend([self.line1, self.line2, self.plot_q1, self.plot_q2, self.plot_dq1, self.plot_dq2])

        # --- Stop animation if flag became false - Corrected Syntax ---
        if not self.running and self.anim:
             try:
                 self.anim.event_source.stop()
             except Exception:
                 pass # Ignore errors stopping timer
        # --- End Stop animation ---
        return artists

    def start_animation(self):
        # --- Stop animation - Corrected Syntax ---
        if self.anim:
             try:
                 self.anim.event_source.stop()
             except Exception:
                 pass # Ignore errors stopping timer
        # --- End Stop animation ---

        self.running = True
        interval_ms = max(1, int(self.dt * 1000))
        if not hasattr(self, 'fig_vis') or not hasattr(self, 'canvas_vis'): print("ERROR: Vis figure not init."); self.running = False; return
        self.anim = FuncAnimation(self.fig_vis, self.animation_frame, frames=None, interval=interval_ms, blit=False, repeat=True, cache_frame_data=False)
        try: self.canvas_vis.draw_idle(); self.canvas_plots.draw_idle()
        except Exception as e: print(f"Error during initial draw: {e}"); self.running = False


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PendubotApp(root)
    def on_closing():
        print("Closing application...")
        app.running = False
        # --- Stop animation - Corrected Syntax ---
        if app.anim:
            try:
                app.anim.event_source.stop()
            except Exception:
                pass # Ignore errors stopping timer
        # --- End Stop animation ---
        root.quit(); root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()