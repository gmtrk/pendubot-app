# gui_components.py

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

def setup_visualization_area(parent_frame, app_instance):
    """Creates the Matplotlib visualization canvas and axes."""
    fig_vis = plt.Figure(figsize=(5, 4))
    ax_vis = fig_vis.add_subplot(111, aspect='equal')
    canvas_vis = FigureCanvasTkAgg(fig_vis, master=parent_frame)
    canvas_vis_widget = canvas_vis.get_tk_widget()
    canvas_vis_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Pendulum lines (initially empty)
    line1, = ax_vis.plot([], [], 'o-', lw=3, color='blue', markersize=8) # Link 1
    line2, = ax_vis.plot([], [], 'o-', lw=3, color='red', markersize=8)   # Link 2
    ax_vis.grid(True)
    ax_vis.set_title("Pendubot")
    ax_vis.set_xlabel("x (m)")
    ax_vis.set_ylabel("y (m)")

    # Store references in the app instance
    app_instance.fig_vis = fig_vis
    app_instance.ax_vis = ax_vis
    app_instance.canvas_vis = canvas_vis
    app_instance.line1 = line1
    app_instance.line2 = line2

    # Initial limits based on default params (will be updated)
    l1, l2 = 1.0, 1.0 # Use defaults initially
    max_len = l1 + l2 + 0.5
    ax_vis.set_xlim(-max_len, max_len)
    ax_vis.set_ylim(-max_len, max_len)


def setup_plotting_area(parent_frame, app_instance):
    """Creates the Matplotlib plotting canvases and axes."""
    fig_plots, axs_plots = plt.subplots(4, 1, sharex=True, figsize=(5, 4))
    canvas_plots = FigureCanvasTkAgg(fig_plots, master=parent_frame)
    canvas_plots_widget = canvas_plots.get_tk_widget()
    canvas_plots_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

    plot_q1, = axs_plots[0].plot([], [], 'b-')
    axs_plots[0].set_ylabel('q1 (rad)')
    axs_plots[0].grid(True)
    axs_plots[0].set_ylim(-np.pi - 0.5, np.pi + 0.5) # Initial guess

    plot_q2, = axs_plots[1].plot([], [], 'r-')
    axs_plots[1].set_ylabel('q2 (rad)')
    axs_plots[1].grid(True)
    axs_plots[1].set_ylim(-np.pi - 0.5, np.pi + 0.5) # Initial guess

    plot_dq1, = axs_plots[2].plot([], [], 'b-')
    axs_plots[2].set_ylabel('dq1 (rad/s)')
    axs_plots[2].grid(True)
    axs_plots[2].set_ylim(-15, 15) # Initial guess

    plot_dq2, = axs_plots[3].plot([], [], 'r-')
    axs_plots[3].set_ylabel('dq2 (rad/s)')
    axs_plots[3].set_xlabel('Time (s)')
    axs_plots[3].grid(True)
    axs_plots[3].set_ylim(-15, 15) # Initial guess

    fig_plots.tight_layout()

    # Store references in the app instance
    app_instance.fig_plots = fig_plots
    app_instance.axs_plots = axs_plots
    app_instance.canvas_plots = canvas_plots
    app_instance.plot_q1 = plot_q1
    app_instance.plot_q2 = plot_q2
    app_instance.plot_dq1 = plot_dq1
    app_instance.plot_dq2 = plot_dq2

def setup_control_panel(parent_frame, app_instance):
    """Creates the control panel widgets, including dynamic PID tuning."""
    row_idx = 0

    # --- Starting Position / Target Configuration ---
    ttk.Label(parent_frame, text="Target Configuration:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=5)
    row_idx += 1
    app_instance.start_pos_var = tk.StringVar(value="Target (pi, 0) [L1 Up, L2 Align]")
    start_positions = [
        "Target (pi, 0) [L1 Up, L2 Align]",
        "Target (0, pi) [L1 Down, L2 World Up]",
        "Target (3pi/4, pi/4) [L2 World Up]",
        "Target (-3pi/4, -pi/4) [L2 World Up]",
        "Target (0, 0) [Fully Down]"
        ]
    for i, pos in enumerate(start_positions):
        rb = ttk.Radiobutton(parent_frame, text=pos, variable=app_instance.start_pos_var, value=pos,
                              command=app_instance.apply_parameters) # Trigger reset
        rb.grid(row=row_idx + i // 2, column=i % 2, sticky=tk.W, padx=5)
    row_idx += (len(start_positions) + 1) // 2

    # --- Parameters ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Parameters:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
    row_idx += 1

    param_frame = ttk.Frame(parent_frame)
    param_frame.grid(row=row_idx, column=0, columnspan=2, sticky='ew')
    param_labels = ["m1 (kg):", "l1 (m):", "m2 (kg):", "l2 (m):"]
    # Use getattr to safely get initial values from app_instance
    param_vars = [tk.StringVar(value=str(getattr(app_instance, 'm1', 1.0))),
                   tk.StringVar(value=str(getattr(app_instance, 'l1', 1.0))),
                   tk.StringVar(value=str(getattr(app_instance, 'm2', 1.0))),
                   tk.StringVar(value=str(getattr(app_instance, 'l2', 1.0)))]
    app_instance.m1_var, app_instance.l1_var, app_instance.m2_var, app_instance.l2_var = param_vars

    for i, label in enumerate(param_labels):
        ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=1)
        entry = ttk.Entry(param_frame, textvariable=param_vars[i], width=10)
        entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=1)
        entry.bind('<Return>', lambda event: app_instance.apply_parameters())
        # Optional: Add trace to update on any change, not just Enter/Apply button
        # param_vars[i].trace_add("write", lambda *args: app_instance.apply_parameters())
    row_idx += 1

    # --- Control Method ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Control Method:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
    row_idx += 1

    app_instance.control_method_var = tk.StringVar(value="LQR")
    control_methods = ["None", "PID", "LQR"]
    for method in control_methods:
        # --- Link radio button command to update PID visibility AND apply params ---
        rb = ttk.Radiobutton(parent_frame, text=method, variable=app_instance.control_method_var, value=method,
                              command=lambda: (app_instance.update_pid_widgets_visibility(), app_instance.apply_parameters()))
        rb.grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, padx=5)
        row_idx += 1

    # --- PID Tuning Frame (Initially hidden potentially) ---
    # Create a frame to hold PID widgets for easy show/hide
    app_instance.pid_tune_frame = ttk.Frame(parent_frame)
    # Place it in the grid, visibility controlled later
    app_instance.pid_tune_frame.grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=(5,0))
    row_idx += 1

    # Define PID default values safely (get from app_instance if already set)
    default_kp = getattr(app_instance, 'kp_val', 30.0)
    default_ki = getattr(app_instance, 'ki_val', 5.0)
    default_kd = getattr(app_instance, 'kd_val', 10.0)

    # StringVars for PID gains (initialize in app_instance.__init__)
    if not hasattr(app_instance, 'kp_var'): app_instance.kp_var = tk.StringVar(value=str(default_kp))
    if not hasattr(app_instance, 'ki_var'): app_instance.ki_var = tk.StringVar(value=str(default_ki))
    if not hasattr(app_instance, 'kd_var'): app_instance.kd_var = tk.StringVar(value=str(default_kd))

    # Create PID Spinboxes and Labels inside the frame
    ttk.Label(app_instance.pid_tune_frame, text="Kp:").grid(row=0, column=0, sticky=tk.W, padx=5)
    kp_spinbox = tk.Spinbox(app_instance.pid_tune_frame, from_=0.0, to=1000.0, increment=1.0, width=8,
                            textvariable=app_instance.kp_var, command=app_instance.apply_parameters_pid_tune) # Trigger update on arrow click
    kp_spinbox.grid(row=0, column=1, sticky=tk.W, padx=5)
    kp_spinbox.bind('<Return>', lambda event: app_instance.apply_parameters()) # Trigger update on Enter

    ttk.Label(app_instance.pid_tune_frame, text="Ki:").grid(row=1, column=0, sticky=tk.W, padx=5)
    ki_spinbox = tk.Spinbox(app_instance.pid_tune_frame, from_=0.0, to=1000.0, increment=0.5, width=8,
                            textvariable=app_instance.ki_var, command=app_instance.apply_parameters_pid_tune)
    ki_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5)
    ki_spinbox.bind('<Return>', lambda event: app_instance.apply_parameters())

    ttk.Label(app_instance.pid_tune_frame, text="Kd:").grid(row=2, column=0, sticky=tk.W, padx=5)
    kd_spinbox = tk.Spinbox(app_instance.pid_tune_frame, from_=0.0, to=1000.0, increment=0.5, width=8,
                            textvariable=app_instance.kd_var, command=app_instance.apply_parameters_pid_tune)
    kd_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5)
    kd_spinbox.bind('<Return>', lambda event: app_instance.apply_parameters())

    # --- Apply Button ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    app_instance.apply_button = ttk.Button(parent_frame, text="Apply & Reset", command=app_instance.apply_parameters)
    app_instance.apply_button.grid(row=row_idx, column=0, columnspan=2, pady=10)
    row_idx += 1