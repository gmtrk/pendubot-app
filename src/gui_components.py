import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

from src.controllers import DEFAULT_CONTROLLER_PARAMS


def setup_visualization_area(parent_frame, app_instance):
    """Creates the Matplotlib visualization canvas and axes."""
    app_instance.fig_vis = plt.Figure(figsize=(5, 4))
    app_instance.ax_vis = app_instance.fig_vis.add_subplot(111, aspect='equal')
    app_instance.canvas_vis = FigureCanvasTkAgg(app_instance.fig_vis, master=parent_frame)
    canvas_vis_widget = app_instance.canvas_vis.get_tk_widget()
    canvas_vis_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    app_instance.line1, = app_instance.ax_vis.plot([], [], 'o-', lw=3, color='blue', markersize=8) # Link 1
    app_instance.line2, = app_instance.ax_vis.plot([], [], 'o-', lw=3, color='red', markersize=8)   # Link 2
    app_instance.ax_vis.grid(True)
    app_instance.ax_vis.set_title("Pendubot")
    app_instance.ax_vis.set_xlabel("x (m)")
    app_instance.ax_vis.set_ylabel("y (m)")

    l1_val = getattr(app_instance, 'l1', 1.0)
    l2_val = getattr(app_instance, 'l2', 0.5)
    max_len = l1_val + l2_val + 0.5
    app_instance.ax_vis.set_xlim(-max_len, max_len)
    app_instance.ax_vis.set_ylim(-max_len, max_len)


def setup_plotting_area(parent_frame, app_instance):
    """Creates the Matplotlib plotting canvases and axes."""
    app_instance.fig_plots, app_instance.axs_plots = plt.subplots(4, 1, sharex=True, figsize=(5, 4))
    app_instance.canvas_plots = FigureCanvasTkAgg(app_instance.fig_plots, master=parent_frame)
    canvas_plots_widget = app_instance.canvas_plots.get_tk_widget()
    canvas_plots_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

    app_instance.plot_q1, = app_instance.axs_plots[0].plot([], [], 'b-')
    app_instance.axs_plots[0].set_ylabel('q1 (rad)')
    app_instance.axs_plots[0].grid(True)
    app_instance.axs_plots[0].set_ylim(-np.pi - 0.5, np.pi + 0.5)

    app_instance.plot_q2, = app_instance.axs_plots[1].plot([], [], 'r-')
    app_instance.axs_plots[1].set_ylabel('q2 (rad)')
    app_instance.axs_plots[1].grid(True)
    app_instance.axs_plots[1].set_ylim(-np.pi - 0.5, np.pi + 0.5)

    app_instance.plot_dq1, = app_instance.axs_plots[2].plot([], [], 'b-')
    app_instance.axs_plots[2].set_ylabel('dq1 (rad/s)')
    app_instance.axs_plots[2].grid(True)
    app_instance.axs_plots[2].set_ylim(-15, 15)

    app_instance.plot_dq2, = app_instance.axs_plots[3].plot([], [], 'r-')
    app_instance.axs_plots[3].set_ylabel('dq2 (rad/s)')
    app_instance.axs_plots[3].set_xlabel('Time (s)')
    app_instance.axs_plots[3].grid(True)
    app_instance.axs_plots[3].set_ylim(-15, 15)

    app_instance.fig_plots.tight_layout()


def setup_control_panel(parent_frame, app_instance):
    """Creates the control panel widgets, including expanded PID tuning."""
    row_idx = 0
    main_colspan = 3  # To accommodate potentially wider button rows

    # --- Starting Position / Target Configuration ---
    ttk.Label(parent_frame, text="Target Configuration:").grid(row=row_idx, column=0, columnspan=main_colspan,
                                                               sticky=tk.W, pady=5)
    row_idx += 1
    if not hasattr(app_instance, 'start_pos_var') or app_instance.start_pos_var is None:
        app_instance.start_pos_var = tk.StringVar(value="Target (pi, 0) [L1 Up, L2 Align]")
    start_positions = [
        "Target (pi, 0) [L1 Up, L2 Align]", "Target (0, pi) [L1 Down, L2 World Up]",
        "Target (3pi/4, pi/4) [L2 World Up]", "Target (-3pi/4, -pi/4) [L2 World Up]",
        "Target (0, 0) [Fully Down]"]
    for i, pos in enumerate(start_positions):
        rb = ttk.Radiobutton(parent_frame, text=pos, variable=app_instance.start_pos_var, value=pos,
                             command=app_instance.apply_parameters)
        rb.grid(row=row_idx + i // 2, column=i % 2, columnspan=1, sticky=tk.W, padx=5)
    row_idx += (len(start_positions) + 1) // 2

    # --- Parameters ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=main_colspan, sticky="ew",
                                                           pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Parameters:").grid(row=row_idx, column=0, columnspan=main_colspan, sticky=tk.W,
                                                     pady=(5, 2))
    row_idx += 1
    param_frame = ttk.Frame(parent_frame)
    param_frame.grid(row=row_idx, column=0, columnspan=main_colspan, sticky='ew')
    param_labels = ["m1 (kg):", "l1 (m):", "m2 (kg):", "l2 (m):"]
    param_vars_data = [('m1', 1.0), ('l1', 1.0), ('m2', 1.0), ('l2', 1.0)]  # Default values if not on app_instance
    app_instance.m1_var, app_instance.l1_var, app_instance.m2_var, app_instance.l2_var = [
        tk.StringVar(value=str(getattr(app_instance, k, v))) for k, v in param_vars_data]
    param_gui_vars = [app_instance.m1_var, app_instance.l1_var, app_instance.m2_var, app_instance.l2_var]

    for i, label_text in enumerate(param_labels):
        ttk.Label(param_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=1)
        entry = ttk.Entry(param_frame, textvariable=param_gui_vars[i], width=10)
        entry.grid(row=i, column=1, sticky=tk.W, padx=5, pady=1)
        entry.bind('<Return>', lambda event, app=app_instance: app.apply_parameters())
    row_idx += 1

    # --- Control Method ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=main_colspan, sticky="ew",
                                                           pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Control Method:").grid(row=row_idx, column=0, columnspan=main_colspan, sticky=tk.W,
                                                         pady=(5, 2))
    row_idx += 1
    if not hasattr(app_instance, 'control_method_var') or app_instance.control_method_var is None:
        app_instance.control_method_var = tk.StringVar(value="LQR")
    control_methods = ["None", "PID", "LQR"]
    for method in control_methods:
        rb = ttk.Radiobutton(parent_frame, text=method, variable=app_instance.control_method_var, value=method,
                             command=lambda app=app_instance: (
                             app.update_pid_widgets_visibility(), app.apply_parameters()))
        rb.grid(row=row_idx, column=0, columnspan=main_colspan, sticky=tk.W, padx=5)
        row_idx += 1

    # --- PID Tuning Frame (Expanded) ---
    app_instance.pid_tune_frame = ttk.LabelFrame(parent_frame, text="State-Feedback PID Gains")
    app_instance.pid_tune_frame.grid(row=row_idx, column=0, columnspan=main_colspan, sticky="ew", pady=(5, 0), padx=5)
    # This frame is now managed by grid, its internal row_idx starts from 0 for its children

    pid_params_config = [
        # (key_prefix, label_text, default_val_attr, string_var_attr, increments)
        ('kp_q1', "Kp (q1):", 'kp_q1_val', 'kp_q1_var', [1.0, 0.5, 0.1]),
        ('ki_q1', "Ki (q1):", 'ki_q1_val', 'ki_q1_var', [0.5, 0.1, 0.01]),
        ('kd_q1', "Kd (q1):", 'kd_q1_val', 'kd_q1_var', [0.5, 0.1, 0.01]),
        ('kp_q2', "Kp (q2):", 'kp_q2_val', 'kp_q2_var', [1.0, 0.5, 0.1]),  # Using existing kp_val etc.
        ('ki_q2', "Ki (q2):", 'ki_q2_val', 'ki_q2_var', [0.5, 0.1, 0.01]),
        ('kd_q2', "Kd (q2):", 'kd_q2_val', 'kd_q2_var', [0.5, 0.1, 0.01]),
    ]

    pid_frame_row = 0
    for key, label, val_attr, var_attr, incs in pid_params_config:
        # Ensure StringVars are initialized on app_instance if not already
        # (app_instance.__init__ should handle this based on DEFAULT_CONTROLLER_PARAMS)
        if not hasattr(app_instance, var_attr):
            # Fallback initialization (ideally done in app_instance.__init__)
            default_val = DEFAULT_CONTROLLER_PARAMS['pid'].get(key.replace('_var', '').replace('_val', '').upper(),
                                                               0.0)  # Kp_q1 from kp_q1_val
            setattr(app_instance, var_attr, tk.StringVar(value=str(default_val)))

        row_frame = ttk.Frame(app_instance.pid_tune_frame)
        row_frame.pack(fill=tk.X, padx=5, pady=2)  # Use pack for rows within LabelFrame

        ttk.Label(row_frame, text=label, width=8).pack(side=tk.LEFT, padx=(0, 2))

        # Use the correct StringVar from app_instance
        spinbox_var = getattr(app_instance, var_attr)
        spinbox_increment = incs[1]  # Use middle increment for spinbox arrows
        spinbox = tk.Spinbox(row_frame, from_=0.0, to=1000.0, increment=spinbox_increment, width=7,
                             textvariable=spinbox_var, command=app_instance.apply_parameters_pid_tune)
        spinbox.pack(side=tk.LEFT, padx=(0, 5))
        spinbox.bind('<Return>', lambda event, app=app_instance: app.apply_parameters())

        for inc in incs:
            ttk.Button(row_frame, text=f"+{inc:.2f}".rstrip('0').rstrip('.'), width=5,
                       command=lambda i=inc, k=key, app=app_instance: app.adjust_pid_param(k, i)).pack(side=tk.LEFT)
            ttk.Button(row_frame, text=f"-{inc:.2f}".rstrip('0').rstrip('.'), width=5,
                       command=lambda i=inc, k=key, app=app_instance: app.adjust_pid_param(k, -i)).pack(side=tk.LEFT,
                                                                                                        padx=(0, 2))
        pid_frame_row += 1

    row_idx += 1  # After the pid_tune_frame

    # --- Apply Button ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=main_colspan, sticky="ew",
                                                           pady=10)
    row_idx += 1
    app_instance.apply_button = ttk.Button(parent_frame, text="Apply All & Reset Sim",
                                           command=app_instance.apply_parameters)
    app_instance.apply_button.grid(row=row_idx, column=0, columnspan=main_colspan, pady=10)
