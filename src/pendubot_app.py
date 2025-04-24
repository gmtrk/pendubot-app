import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Import components from other files
from pendubot_dynamics import pendubot_dynamics
from controllers import (control_none, control_pid, control_lqr,
                         calculate_lqr_gain, DEFAULT_CONTROLLER_PARAMS)
from gui_components import (setup_visualization_area, setup_plotting_area,
                            setup_control_panel)

class PendubotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pendubot Simulation")
        self.root.geometry("1200x700") # Width x Height

        # --- Default Simulation Parameters ---
        self.m1 = 1.0
        self.l1 = 1.0
        self.m2 = 1.0
        self.l2 = 1.0
        # These will be updated by apply_parameters based on m1, l1, m2, l2
        self.lc1 = self.l1
        self.lc2 = self.l2 / 2.0
        self.I1 = (1/3) * self.m1 * self.l1**2
        self.I2 = (1/12) * self.m2 * self.l2**2

        self.dt = 0.02  # Simulation time step (s) -> 50 Hz
        self.simulation_time = 0.0
        self.state = np.zeros(4) # Initial state [q1, q2, dq1, dq2]

        # Data storage for plots
        self.time_history = []
        self.q1_history = []
        self.q2_history = []
        self.dq1_history = []
        self.dq2_history = []
        self.history_length = 250 # Number of points to show (~5 seconds at 50Hz)

        # Controller selection and state
        # GUI variable is created in setup_control_panel
        self.control_method_var = None # Placeholder, will be set by setup_control_panel
        self.control_func = control_none
        self.current_controller_state = {} # Holds runtime state (PID integral, LQR gain K)

        # Animation control
        self.anim = None
        self.running = False

        # --- GUI Setup ---
        self.setup_gui() # Calls functions from gui_components

        # --- Finalize Setup ---
        # Set the command for the apply button (created in gui_components)
        self.apply_button.config(command=self.apply_parameters)

        # Apply initial default parameters and start simulation
        self.apply_parameters()


    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Left frame for visualization and plots
        left_frame = ttk.Frame(main_frame, padding="5")
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        main_frame.columnconfigure(0, weight=3) # Give more space to viz/plots
        main_frame.rowconfigure(0, weight=1)

        # Right frame for controls
        right_frame = ttk.Frame(main_frame, padding="10", borderwidth=2, relief="groove")
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)
        main_frame.columnconfigure(1, weight=1) # Control panel width

        # --- Populate Frames using imported functions ---
        setup_visualization_area(left_frame, self)
        setup_plotting_area(left_frame, self)
        setup_control_panel(right_frame, self)

        # Configure row weights for left_frame (viz/plots)
        left_frame.rowconfigure(0, weight=2) # Give more vertical space to viz
        left_frame.rowconfigure(1, weight=3) # Give more space to plots
        left_frame.columnconfigure(0, weight=1)


        # --- Status Bar ---
        self.status_var = tk.StringVar(value="Ready.")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))


    def get_cartesian_coords(self, q1, q2):
        """Calculate Cartesian coordinates of the joints."""
        x0, y0 = 0, 0
        # Use current link lengths
        x1 = self.l1 * np.sin(q1)
        y1 = -self.l1 * np.cos(q1) # Negative cos for standard plot orientation (y up)
        x2 = x1 + self.l2 * np.sin(q1 + q2)
        y2 = y1 - self.l2 * np.cos(q1 + q2)
        return x0, y0, x1, y1, x2, y2

    def update_visualization(self):
        """Update the Matplotlib visualization."""
        if not hasattr(self, 'line1'): return # Check if GUI setup is complete

        q1, q2 = self.state[0], self.state[1]
        x0, y0, x1, y1, x2, y2 = self.get_cartesian_coords(q1, q2)

        self.line1.set_data([x0, x1], [y0, y1])
        self.line2.set_data([x1, x2], [y1, y2])

        # Update limits based on current lengths
        max_len = self.l1 + self.l2 + 0.5
        self.ax_vis.set_xlim(-max_len, max_len)
        self.ax_vis.set_ylim(-max_len, max_len)
        # Redraw the axes elements like grid and labels if limits changed significantly
        # self.ax_vis.figure.canvas.draw_idle() # More efficient than full canvas draw
        self.canvas_vis.draw_idle()


    def update_plots(self):
        """Update the Matplotlib plots."""
        if not hasattr(self, 'plot_q1') or not self.time_history: return # Check if GUI setup is complete and data exists

        # Limit history shown
        t = self.time_history[-self.history_length:]
        q1 = self.q1_history[-self.history_length:]
        q2 = self.q2_history[-self.history_length:]
        dq1 = self.dq1_history[-self.history_length:]
        dq2 = self.dq2_history[-self.history_length:]

        self.plot_q1.set_data(t, q1)
        self.plot_q2.set_data(t, q2)
        self.plot_dq1.set_data(t, dq1)
        self.plot_dq2.set_data(t, dq2)

        # Adjust plot limits automatically
        for i in range(4):
            self.axs_plots[i].relim()
            self.axs_plots[i].autoscale_view(True, True, True) # Rescale x and y

        # Keep x-axis synchronized and showing the latest time window
        min_time = t[0]
        max_time = t[-1]
        # Ensure a minimum time range is visible, e.g., 1 second
        time_range = max(1.0, max_time - min_time)
        self.axs_plots[3].set_xlim(max_time - time_range, max_time)

        self.canvas_plots.draw_idle()


    def apply_parameters(self):
        """Read parameters from GUI, reset simulation, and restart."""
        if self.anim:
            self.anim.event_source.stop()
            self.anim = None
            self.running = False

        try:
            # Read and validate parameters from GUI variables
            m1 = float(self.m1_var.get())
            l1 = float(self.l1_var.get())
            m2 = float(self.m2_var.get())
            l2 = float(self.l2_var.get())
            if m1 <= 0 or l1 <= 0 or m2 <= 0 or l2 <= 0:
                raise ValueError("Masses and lengths must be positive.")

            self.m1, self.l1, self.m2, self.l2 = m1, l1, m2, l2

            # Update dependent params (CoM, Inertia)
            self.lc1 = self.l1         # CoM at end of link 1
            self.lc2 = self.l2 / 2.0    # CoM at middle of link 2
            self.I1 = (1/3) * self.m1 * self.l1**2 # Rod about end
            self.I2 = (1/12) * self.m2 * self.l2**2 # Rod about center

            # Set Initial State based on selection
            start_pos = self.start_pos_var.get()
            # Angles relative to **vertical downward** for q1, relative to **link 1** for q2
            # Goal: Stabilize q2 near 0 (second link pointing "up" relative to first link)
            q1_init, q2_init, dq1_init, dq2_init = 0.0, 0.0, 0.0, 0.0
            perturb = 0.05 # Small perturbation

            if start_pos == "Near Down-Up":         # q1=pi (Down), q2=0 (Aligned Upward relative to link 1)
                q1_init, q2_init = np.pi + perturb, 0.0 + perturb
            elif start_pos == "Near Side-Up (R)":   # q1=pi/2 (Right), q2=0 (Aligned Upward rel to link 1)
                q1_init, q2_init = np.pi/2 + perturb, 0.0 + perturb
            elif start_pos == "Near Side-Up (L)":   # q1=-pi/2 (Left), q2=0 (Aligned Upward rel to link 1)
                q1_init, q2_init = -np.pi/2 + perturb, 0.0 + perturb
            elif start_pos == "Fully Down":         # q1=pi (Down), q2=pi (Down rel to link 1) -> hangs down
                q1_init, q2_init = np.pi, np.pi
            elif start_pos == "Near Up-Up (Unstable)": # q1=0 (Up), q2=0 (Aligned Upward rel to link 1) -> unstable point
                 q1_init, q2_init = 0.0 + perturb, 0.0 + perturb
            else: # Default
                q1_init, q2_init = np.pi + perturb, 0.0 + perturb

            self.state = np.array([q1_init, q2_init, dq1_init, dq2_init])

            # Reset simulation time and history
            self.simulation_time = 0.0
            self.time_history = []
            self.q1_history = []
            self.q2_history = []
            self.dq1_history = []
            self.dq2_history = []

            # --- Select Control Function and Reset/Initialize Controller State ---
            method = self.control_method_var.get()
            self.current_controller_state = {} # Reset state for new controller

            if method == "PID":
                self.control_func = control_pid
                # Initialize PID state using defaults, pass dt
                self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
                self.current_controller_state['dt'] = self.dt
                self.current_controller_state['integral_error'] = 0.0
                # Initialize last error for derivative calculation if using finite diff
                # self.current_controller_state['last_error'] = (self.current_controller_state['target_q2'] - q2_init + np.pi) % (2*np.pi) - np.pi
                pid_params = self.current_controller_state
                self.status_var.set(f"Running PID (Kp={pid_params['Kp']}, Ki={pid_params['Ki']}, Kd={pid_params['Kd']}).")

            elif method == "LQR":
                self.control_func = control_lqr
                # Get Q, R from defaults
                Q = DEFAULT_CONTROLLER_PARAMS['lqr']['Q']
                R = DEFAULT_CONTROLLER_PARAMS['lqr']['R']
                # Calculate LQR gain based on current physical parameters
                K = calculate_lqr_gain(self.m1, self.l1, self.m2, self.l2, self.lc1, self.lc2, self.I1, self.I2, Q, R)

                if K is not None:
                     # Initialize LQR state using defaults and calculated K
                    self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['lqr'].copy()
                    self.current_controller_state['K'] = K
                    # Target state is already in defaults
                    self.status_var.set("Running LQR.")
                else:
                    # Fallback if LQR calculation failed
                    self.control_func = control_none
                    self.control_method_var.set("None") # Update radio button display
                    self.status_var.set("LQR Failed. No control applied.")

            else: # None
                self.control_func = control_none
                self.status_var.set("Running with no control.")

            # Clear plots and redraw axes before starting animation
            self.update_visualization() # Set initial position display
            self.update_plots() # Clear plots

            # Restart animation
            self.start_animation()

        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.status_var.set(f"Error: Invalid parameter values. {e}")
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred during setup: {e}")
             self.status_var.set(f"Error: {e}")
             print(f"Detailed Error: {e}", flush=True) # Print details to console


    def simulation_step(self):
        """Perform one simulation step using solve_ivp."""
        try:
            sol = scipy.integrate.solve_ivp(
                pendubot_dynamics,
                [self.simulation_time, self.simulation_time + self.dt],
                self.state,
                method='RK45', # Robust solver
                t_eval=[self.simulation_time + self.dt], # Evaluate only at the end
                args=(self.m1, self.l1, self.m2, self.l2,
                      self.lc1, self.lc2, self.I1, self.I2,
                      self.control_func, self.current_controller_state) # Pass controller state dict
            )

            if sol.status != 0:
                 print(f"Warning: ODE solver failed (Status {sol.status}) - {sol.message}")
                 # Option: Stop simulation or keep last state
                 self.state = self.state # Keep last known state
            else:
                self.state = sol.y[:, -1]

            # Optional: Wrap angles if needed for interpretation, but dynamics handle periodicity
            # self.state[0] = (self.state[0] + np.pi) % (2 * np.pi) - np.pi
            # self.state[1] = (self.state[1] + np.pi) % (2 * np.pi) - np.pi

            self.simulation_time += self.dt

            # Record history
            self.time_history.append(self.simulation_time)
            self.q1_history.append(self.state[0])
            self.q2_history.append(self.state[1])
            self.dq1_history.append(self.state[2])
            self.dq2_history.append(self.state[3])

            # Limit history buffer size efficiently using deque would be better, but list pop is ok here
            while len(self.time_history) > self.history_length + 50: # Keep a bit more than needed for plotting
                self.time_history.pop(0)
                self.q1_history.pop(0)
                self.q2_history.pop(0)
                self.dq1_history.pop(0)
                self.dq2_history.pop(0)

        except Exception as e:
            print(f"Error during simulation step: {e}")
            self.status_var.set(f"Simulation Error: {e}")
            self.running = False # Stop on error


    def animation_frame(self, frame):
        """Update function for Matplotlib animation."""
        artists = []
        if self.running:
            self.simulation_step()
            self.update_visualization()
            self.update_plots()

            # Collect artists that were updated for blitting (if used)
            artists.extend([self.line1, self.line2, self.plot_q1, self.plot_q2, self.plot_dq1, self.plot_dq2])
            # May need to include axes text labels if limits change drastically, but autoscale handles most cases.
        return artists


    def start_animation(self):
        """Starts the Matplotlib animation."""
        if self.anim: # Stop existing animation if restarting
             self.anim.event_source.stop()

        self.running = True
        # Interval is in milliseconds. dt*1000 aims for real-time.
        interval_ms = max(1, int(self.dt * 1000)) # Ensure interval is at least 1ms

        # Create or restart the animation object
        # blit=False is generally safer with TkAgg and complex updates.
        self.anim = FuncAnimation(self.fig_vis, self.animation_frame, frames=None,
                                  interval=interval_ms, blit=False, repeat=True, cache_frame_data=False)

        self.canvas_vis.draw_idle()
        self.canvas_plots.draw_idle()


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PendubotApp(root)

    # Ensure clean exit when window is closed
    def on_closing():
        if app.anim:
            app.anim.event_source.stop() # Stop animation timer
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()