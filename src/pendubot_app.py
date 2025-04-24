# pendubot_app.py

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.integrate
import matplotlib
matplotlib.use('TkAgg') # Ensure TkAgg backend is used
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import traceback # For printing tracebacks

# Import components from other files
from pendubot_dynamics import pendubot_dynamics # Assumes this uses q1 from downward vertical
from controllers import (control_none, control_pid, control_lqr,
                         calculate_lqr_gain, DEFAULT_CONTROLLER_PARAMS)
from gui_components import (setup_visualization_area, setup_plotting_area,
                            setup_control_panel)

class PendubotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pendubot Simulation")
        self.root.geometry("1200x700")

        # --- Default Simulation Parameters ---
        self.m1 = 1.0
        self.l1 = 1.0
        self.m2 = 1.0
        self.l2 = 1.0
        self.lc1 = self.l1
        self.lc2 = self.l2 / 2.0
        self.I1 = (1/3) * self.m1 * self.l1**2
        self.I2 = (1/12) * self.m2 * self.l2**2

        self.dt = 0.02
        self.simulation_time = 0.0
        self.state = np.zeros(4) # Uses STANDARD angle convention (q1 from downward vertical)

        # Data storage for plots
        self.time_history = []
        self.q1_history = []
        self.q2_history = []
        self.dq1_history = []
        self.dq2_history = []
        self.history_length = 250

        # Controller selection and state
        self.control_method_var = None # Set by GUI setup
        self.control_func = control_none
        self.current_controller_state = {}
        self.target_state = np.zeros(4) # Uses STANDARD angle convention

        # Animation control
        self.anim = None
        self.running = False

        # --- GUI Setup ---
        self.setup_gui() # Calls functions from gui_components

        # --- Finalize Setup ---
        # Command is now set on radio buttons in gui_components
        # self.apply_button.config(command=self.apply_parameters)

        # Apply initial default parameters and start simulation
        # This will be called by the default radio button selection via its command
        # self.apply_parameters() # No longer needed here if radio buttons have commands


    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        # Left frame
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.columnconfigure(0, weight=3)
        main_frame.rowconfigure(0, weight=1)
        # Right frame
        right_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1)
        # Populate Frames using functions from gui_components
        # These functions need access to self, so they are called correctly below
        setup_visualization_area(left_frame, self)
        setup_plotting_area(left_frame, self)
        setup_control_panel(right_frame, self) # This now sets commands on radio buttons
        # Configure row weights for left_frame
        left_frame.rowconfigure(0, weight=2)
        left_frame.rowconfigure(1, weight=3)
        left_frame.columnconfigure(0, weight=1)
        # Status Bar
        self.status_var = tk.StringVar(value="Ready. Select target/control.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))


    def get_cartesian_coords(self, q1, q2):
        """
        Calculate Cartesian coordinates based on STANDARD angle convention:
        q1=0 is DOWN, q1=pi is UP (angle from downward vertical).
        q2=0 is aligned with link 1.
        Plot assumes y is positive UP.
        """
        x0, y0 = 0, 0
        # Kinematics using q1 angle from downward vertical, y pointing UP on plot
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1) # Negative cos for y UP relative to origin
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1 + q2)
        return x0, y0, x1, y1, x2, y2


    def apply_parameters(self):
        """Read parameters from GUI, reset simulation, and restart."""
        # print("DEBUG: apply_parameters called.") # Optional debug
        if self.anim and self.running: # Only stop if animation is actually running
            # print("DEBUG: Stopping existing animation...") # Optional debug
            try:
                self.anim.event_source.stop()
            except Exception as e:
                print(f"DEBUG: Error stopping animation: {e}")
            self.anim = None # Ensure old animation object is cleared
        self.running = False # Stop simulation loop if it was running

        try:
            # Read and validate parameters
            m1 = float(self.m1_var.get())
            l1 = float(self.l1_var.get())
            m2 = float(self.m2_var.get())
            l2 = float(self.l2_var.get())
            if m1 <= 0 or l1 <= 0 or m2 <= 0 or l2 <= 0:
                raise ValueError("Masses and lengths must be positive.")
            self.m1, self.l1, self.m2, self.l2 = m1, l1, m2, l2
            self.lc1 = self.l1
            self.lc2 = self.l2 / 2.0
            self.I1 = (1/3) * self.m1 * self.l1**2
            self.I2 = (1/12) * self.m2 * self.l2**2

            # --- Set Initial State and Target State (STANDARD CONVENTION) ---
            # q1=0 DOWN, q1=pi UP. q2=0 aligned.
            start_pos = self.start_pos_var.get()
            q1_init, q2_init, dq1_init, dq2_init = 0.0, 0.0, 0.0, 0.0
            target_q1, target_q2 = 0.0, 0.0 # Targets in standard convention
            perturb = 0.05

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
                 # Use equivalent positive angle for q1 if preferred: 5pi/4 = -3pi/4
                q1_init_angle = -3*np.pi/4 # Or 5*np.pi/4
                q1_init, q2_init = q1_init_angle + perturb, (-np.pi/4) - perturb
                target_q1, target_q2 = q1_init_angle, (-np.pi/4)
            elif start_pos == "Target (0, 0) [Fully Down]":
                 q1_init, q2_init = 0.0 + perturb, 0.0 + perturb
                 target_q1, target_q2 = 0.0, 0.0
            else: # Default case if selection invalid
                print(f"Warning: Unknown start_pos '{start_pos}'. Defaulting to Target (pi, 0).")
                q1_init, q2_init = np.pi + perturb, 0.0 + perturb
                target_q1, target_q2 = np.pi, 0.0

            self.state = np.array([q1_init, q2_init, dq1_init, dq2_init])
            self.target_state = np.array([target_q1, target_q2, 0.0, 0.0])
            print(f"INFO: Target state (q1_down, q2_rel) set to: {self.target_state}")

            # --- Reset simulation time and history ---
            # print("DEBUG: Resetting history lists...") # Optional debug
            self.simulation_time = 0.0
            self.time_history = []
            self.q1_history = []
            self.q2_history = []
            self.dq1_history = []
            self.dq2_history = []
            # print(f"DEBUG apply_parameters - AFTER RESET: Lengths: t=...") # Optional debug

            # --- Select Control Function ---
            method = self.control_method_var.get()
            self.current_controller_state = {}

            if method == "PID":
                self.control_func = control_pid
                self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
                self.current_controller_state['dt'] = self.dt
                self.current_controller_state['integral_error'] = 0.0
                self.current_controller_state['target_q2'] = target_q2 # PID target q2
                pid_params = self.current_controller_state
                self.status_var.set(f"Running PID (Targets q2={target_q2:.2f}; ...).")

            elif method == "LQR":
                self.control_func = control_lqr
                Q = DEFAULT_CONTROLLER_PARAMS['lqr']['Q']
                R = DEFAULT_CONTROLLER_PARAMS['lqr']['R']

                # Calculate LQR gain using target state in STANDARD convention
                K, u_eq = calculate_lqr_gain(self.m1, self.l1, self.m2, self.l2,
                                          self.lc1, self.lc2, self.I1, self.I2,
                                          Q, R, q1_eq=target_q1, q2_eq=target_q2)

                if K is not None:
                    self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['lqr'].copy()
                    self.current_controller_state['K'] = K
                    # Store the target state in STANDARD convention
                    self.current_controller_state['target_state'] = self.target_state
                    self.current_controller_state['u_eq'] = u_eq
                    self.status_var.set(f"Running LQR (Targets q1={target_q1:.2f}, q2={target_q2:.2f}).")
                else:
                    self.control_func = control_none
                    self.control_method_var.set("None")
                    self.status_var.set("LQR Failed. No control applied.")

            else: # None
                self.control_func = control_none
                self.status_var.set("Running with no control.")

            # --- Update visuals and restart animation ---
            # print("DEBUG: Updating plots/visualization before restart.") # Optional debug
            self.update_visualization()
            self.update_plots() # Clear plots with new data state

            # print("DEBUG: apply_parameters - Before start_animation.") # Optional debug
            self.start_animation() # This restarts the animation loop
            # print("DEBUG: apply_parameters - After start_animation.") # Optional debug

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.status_var.set(f"Error: Invalid parameter values. {e}")
            self.running = False # Stop if error during setup
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred during setup: {e}")
             self.status_var.set(f"Error: {e}")
             print(f"Detailed Error: {e}", flush=True)
             traceback.print_exc()
             self.running = False # Stop if error during setup


    def update_visualization(self):
        """Update the Matplotlib visualization."""
        if not hasattr(self, 'line1') or not self.running: return

        q1, q2 = self.state[0], self.state[1] # Standard convention angles
        try:
            x0, y0, x1, y1, x2, y2 = self.get_cartesian_coords(q1, q2)
            self.line1.set_data([x0, x1], [y0, y1])
            self.line2.set_data([x1, x2], [y1, y2])

            max_len = self.l1 + self.l2 + 0.5
            current_xlim = self.ax_vis.get_xlim()
            # Avoid unnecessary limit setting if possible
            if current_xlim[0] != -max_len or current_xlim[1] != max_len:
                 self.ax_vis.set_xlim(-max_len, max_len)
                 self.ax_vis.set_ylim(-max_len, max_len)
            self.canvas_vis.draw() # Use draw() for potentially more immediate updates
        except Exception as e:
            print(f"Error during visualization update: {e}")
            self.running = False # Stop on error


    def update_plots(self):
        """Update the Matplotlib plots."""
        if not hasattr(self, 'plot_q1') or not self.running or not self.time_history:
            return

        try:
            # Use local variables for safety within try block
            time_hist = self.time_history
            q1_hist = self.q1_history
            q2_hist = self.q2_history
            dq1_hist = self.dq1_history
            dq2_hist = self.dq2_history

            current_len = len(time_hist)
            # Check list sync (important if errors occur during append)
            if not (current_len == len(q1_hist) == len(q2_hist) == len(dq1_hist) == len(dq2_hist)):
                print("CRITICAL ERROR: History list lengths are out of sync!")
                self.running = False; return

            t = time_hist[-self.history_length:]
            q1 = q1_hist[-self.history_length:]
            q2 = q2_hist[-self.history_length:]
            dq1 = dq1_hist[-self.history_length:]
            dq2 = dq2_hist[-self.history_length:]

        except Exception as e:
            print(f"Error accessing or slicing history lists: {e}")
            self.running = False; return

        if len(t) == 0: # Handle empty data after reset
            self.plot_q1.set_data([], []); self.plot_q2.set_data([], [])
            self.plot_dq1.set_data([], []); self.plot_dq2.set_data([], [])
        else:
            self.plot_q1.set_data(t, q1); self.plot_q2.set_data(t, q2)
            self.plot_dq1.set_data(t, dq1); self.plot_dq2.set_data(t, dq2)

        try: # Update limits and draw
            needs_redraw = False
            for i in range(4):
                current_ax = self.axs_plots[i]
                old_lim = current_ax.get_ylim()
                current_ax.relim()
                current_ax.autoscale_view(True, False, True) # Autoscale Y only
                if old_lim != current_ax.get_ylim(): needs_redraw = True

            if len(t) > 0:
                min_time = t[0]; max_time = t[-1]
                time_range = max(1.0, max_time - min_time if max_time > min_time else 1.0)
                xlim_min = max(min_time, max_time - time_range)
                current_xlim = self.axs_plots[3].get_xlim()
                if current_xlim != (xlim_min, max_time):
                    for ax in self.axs_plots: ax.set_xlim(xlim_min, max_time)
                    needs_redraw = True

            if needs_redraw or len(t) == 0: # Redraw if limits changed or plots cleared
                self.canvas_plots.draw()

        except Exception as e: # Catch drawing/limit errors
            print(f"Error updating plot limits or drawing: {e}")
            self.running = False


    def simulation_step(self):
        """Perform one simulation step using solve_ivp."""
        # State `self.state` uses STANDARD angle convention
        if not self.running: return # Exit if stop requested
        try:
            # Controller function needs state in STANDARD convention now
            control_args = (self.m1, self.l1, self.m2, self.l2,
                            self.lc1, self.lc2, self.I1, self.I2,
                            self.control_func, self.current_controller_state)

            # Dynamics function expects STANDARD convention
            sol = scipy.integrate.solve_ivp(
                pendubot_dynamics,
                [self.simulation_time, self.simulation_time + self.dt],
                self.state, # Pass state directly (standard convention)
                method='RK45',
                t_eval=[self.simulation_time + self.dt],
                args=control_args
            )

            if sol.status != 0:
                 print(f"Warning: ODE solver failed (Status {sol.status}) - {sol.message}")
                 self.state = self.state # Keep last state
            else:
                self.state = sol.y[:, -1] # Result is in standard coords

            if not np.all(np.isfinite(self.state)):
                print(f"WARNING: Invalid state detected after step t={self.simulation_time}: {self.state}")
                self.running = False; return

            self.simulation_time += self.dt

            # Record history (standard convention)
            self.time_history.append(self.simulation_time)
            self.q1_history.append(self.state[0])
            self.q2_history.append(self.state[1])
            self.dq1_history.append(self.state[2])
            self.dq2_history.append(self.state[3])

            # Limit history buffer size
            while len(self.time_history) > self.history_length + 50:
                self.time_history.pop(0); self.q1_history.pop(0)
                self.q2_history.pop(0); self.dq1_history.pop(0)
                self.dq2_history.pop(0)

        except Exception as e:
            print(f"Error during simulation step: {e}")
            traceback.print_exc()
            self.status_var.set(f"Simulation Error: {e}")
            self.running = False


    def animation_frame(self, frame):
        """Update function for Matplotlib animation."""
        # Ensure self.running is checked *before* calling simulation_step
        # to prevent issues if simulation stopped mid-frame previously.
        if not self.running:
            # Stop the animation explicitly if running flag is false
            if self.anim:
                try: self.anim.event_source.stop()
                except: pass
            return [] # Return empty list if not running

        self.simulation_step() # Call simulation step first

        artists = []
        # Update visuals only if still running after simulation step
        if self.running:
            self.update_visualization()
            self.update_plots()
            artists.extend([self.line1, self.line2, self.plot_q1, self.plot_q2, self.plot_dq1, self.plot_dq2])

        # If self.running became False during simulation_step or updates, stop animation
        if not self.running and self.anim:
             try: self.anim.event_source.stop()
             except: pass

        return artists


    def start_animation(self):
        """Starts the Matplotlib animation."""
        # Stop existing timer if somehow present
        if self.anim:
             try: self.anim.event_source.stop()
             except: pass

        self.running = True # Set running flag
        interval_ms = max(1, int(self.dt * 1000))

        # Ensure figures/canvases exist before creating animation
        if not hasattr(self, 'fig_vis') or not hasattr(self, 'canvas_vis'):
             print("ERROR: Visualization figure/canvas not initialized.")
             self.running = False
             return

        self.anim = FuncAnimation(self.fig_vis, self.animation_frame, frames=None,
                                  interval=interval_ms, blit=False, repeat=True, cache_frame_data=False)

        # Perform an initial draw to ensure canvas is ready
        try:
            self.canvas_vis.draw_idle()
            self.canvas_plots.draw_idle()
        except Exception as e:
             print(f"Error during initial draw: {e}")
             self.running = False # Stop if initial draw fails


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PendubotApp(root)

    def on_closing():
        print("Closing application...")
        app.running = False # Stop simulation loop flag
        if app.anim:
            try: app.anim.event_source.stop()
            except: pass
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()