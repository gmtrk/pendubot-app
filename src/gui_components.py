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
    """Creates the control panel widgets."""
    row_idx = 0

    # --- Starting Position ---
    ttk.Label(parent_frame, text="Starting Position:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=5)
    row_idx += 1
    app_instance.start_pos_var = tk.StringVar(value="Near Down-Up") # Default near stable point
    start_positions = ["Near Down-Up", "Near Side-Up (R)", "Near Side-Up (L)", "Fully Down", "Near Up-Up (Unstable)"]
    for i, pos in enumerate(start_positions):
        rb = ttk.Radiobutton(parent_frame, text=pos, variable=app_instance.start_pos_var, value=pos)
        # Grid layout wraps radio buttons nicely
        rb.grid(row=row_idx + i // 2, column=i % 2, sticky=tk.W, padx=5)
    row_idx += (len(start_positions) + 1) // 2


    # --- Parameters ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Parameters:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
    row_idx += 1

    param_frame = ttk.Frame(parent_frame) # Frame for better alignment
    param_frame.grid(row=row_idx, column=0, columnspan=2, sticky='ew')
    param_labels = ["m1 (kg):", "l1 (m):", "m2 (kg):", "l2 (m):"]
    param_vars = [tk.StringVar(value=str(app_instance.m1)),
                   tk.StringVar(value=str(app_instance.l1)),
                   tk.StringVar(value=str(app_instance.m2)),
                   tk.StringVar(value=str(app_instance.l2))]
    app_instance.m1_var, app_instance.l1_var, app_instance.m2_var, app_instance.l2_var = param_vars

    for i, label in enumerate(param_labels):
        ttk.Label(param_frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=1)
        ttk.Entry(param_frame, textvariable=param_vars[i], width=10).grid(row=i, column=1, sticky=tk.W, padx=5, pady=1)
    row_idx += 1


    # --- Control Method ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    ttk.Label(parent_frame, text="Control Method:").grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, pady=(5,2))
    row_idx += 1

    app_instance.control_method_var = tk.StringVar(value="LQR") # Default to LQR
    control_methods = ["None", "PID", "LQR"]
    for method in control_methods:
        rb = ttk.Radiobutton(parent_frame, text=method, variable=app_instance.control_method_var, value=method)
        rb.grid(row=row_idx, column=0, columnspan=2, sticky=tk.W, padx=5)
        row_idx += 1


    # --- Apply Button ---
    ttk.Separator(parent_frame, orient=tk.HORIZONTAL).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=10)
    row_idx += 1
    # Assign button to app instance so command can be set later
    app_instance.apply_button = ttk.Button(parent_frame, text="Apply & Reset")
    app_instance.apply_button.grid(row=row_idx, column=0, columnspan=2, pady=10)
    row_idx += 1