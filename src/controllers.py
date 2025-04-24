# controllers.py

import numpy as np
import control # For LQR
from pendubot_dynamics import pendubot_dynamics, G # Use G for equilibrium torque

# --- Default Controller Parameters ---
DEFAULT_CONTROLLER_PARAMS = {
    'pid': {'Kp': 25.0, 'Ki': 2.0, 'Kd': 5.0, 'max_integral': 10.0, 'max_torque': 15.0},
    'lqr': {
            'Q': np.diag([0.5, 10.0, 0.1, 1.0]), # Weights tuned for q2 stability mostly
            'R': np.array([[0.05]]),
            'max_torque': 20.0
           }
}

# --- Control Function Implementations ---

def control_none(t, x, controller_state):
    """No control input."""
    return 0.0

def control_pid(t, x, controller_state):
    """PID controller aiming for target_q2 (relative angle)."""
    q1, q2, dq1, dq2 = x # Standard convention state
    params = controller_state

    Kp = params.get('Kp', 0.0)
    Ki = params.get('Ki', 0.0)
    Kd = params.get('Kd', 0.0)
    target_q2 = params.get('target_q2', 0.0) # Target relative angle
    dt = params.get('dt', 0.02)
    max_integral = params.get('max_integral', 10.0)
    max_torque = params.get('max_torque', 15.0)

    # Error calculation based on q2 (relative angle)
    error = target_q2 - q2
    error = (error + np.pi) % (2 * np.pi) - np.pi # Normalize error

    # Integral term
    current_integral = params.get('integral_error', 0.0)
    current_integral += error * dt
    current_integral = np.clip(current_integral, -max_integral, max_integral)
    params['integral_error'] = current_integral # Update state

    # Derivative term (using velocity dq2)
    derivative_error = -dq2 # d(target-q2)/dt = -dq2

    # PID control law
    tau1 = Kp * error + Ki * current_integral + Kd * derivative_error
    tau1 = np.clip(tau1, -max_torque, max_torque)
    return float(tau1)

def control_lqr(t, x, controller_state):
    """
    LQR controller stabilizing around target_state (standard convention).
    Uses gain K derived from linearization around target_state.
    Applies feedback and feedforward torque: u = -K*(x - target_state) + u_eq.
    """
    params = controller_state

    K = params.get('K')
    target_state = params.get('target_state') # Target state in standard convention
    u_eq = params.get('u_eq', 0.0)
    max_torque = params.get('max_torque', 20.0)

    if K is None or target_state is None:
        print("Warning: LQR gain K or target_state not available.")
        return 0.0

    # Calculate error state (deviation from target) in standard convention
    error_x = x - target_state
    # Wrap angle errors
    error_x[0] = (error_x[0] + np.pi) % (2 * np.pi) - np.pi # q1 error wrap
    error_x[1] = (error_x[1] + np.pi) % (2 * np.pi) - np.pi # q2 error wrap

    # LQR control law: u = -K * error_x + u_eq
    if K.shape[1] != len(error_x):
         print(f"ERROR: LQR K shape {K.shape} incompatible with error_x shape {error_x.shape}")
         feedback_torque = 0.0
    else:
         feedback_torque_array = -K @ error_x
         feedback_torque = feedback_torque_array[0] # Extract scalar

    tau1 = feedback_torque + u_eq # Add feedforward term
    tau1 = np.clip(tau1, -max_torque, max_torque)
    return float(tau1)


# --- LQR Gain Calculation ---

def calculate_lqr_gain(m1, l1, m2, l2, lc1, lc2, I1, I2, Q, R, q1_eq, q2_eq):
    """
    Linearizes the system around the equilibrium (q1_eq, q2_eq, 0, 0)
    expressed in STANDARD angle convention (q1 from downward vertical).
    Computes the LQR gain K and equilibrium torque u_eq.
    Returns K, u_eq, or None, None if calculation fails.
    """
    # Equilibrium state in STANDARD convention (q1 from downward vertical)
    x_eq = np.array([q1_eq, q2_eq, 0.0, 0.0])
    print(f"DEBUG: Linearizing LQR around state (q1_down={q1_eq:.2f}, q2_rel={q2_eq:.2f})")

    # --- Calculate equilibrium torque u_eq using STANDARD angles ---
    s1 = np.sin(q1_eq)
    s12 = np.sin(q1_eq + q2_eq)
    g1_eq = (m1 * lc1 + m2 * l1) * G * s1 + m2 * lc2 * G * s12
    g2_eq = m2 * lc2 * G * s12

    if not np.isclose(g2_eq, 0.0, atol=1e-6):
        print(f"WARNING: Target state q1={q1_eq:.2f}, q2={q2_eq:.2f} is not a true passive equilibrium (g2={g2_eq:.4f} != 0).")

    u_eq = g1_eq # Equilibrium torque is g1 term
    print(f"INFO: Equilibrium torque u_eq for target is approx {u_eq:.2f}")
    u_eq_lin = np.array([0.0]) # Use u=0 relative to u_eq for feedback gain K

    # --- Numerical linearization using STANDARD state ---
    epsilon = 1e-6
    def dynamics_wrapper(x_standard, u_val):
        # Assumes pendubot_dynamics expects standard q1 (from downward vertical)
        return pendubot_dynamics(0, x_standard, m1, l1, m2, l2, lc1, lc2, I1, I2,
                                 lambda t, state, ctrl_state: u_val, {})[:4]

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    for i in range(4):
        x_plus = x_eq.copy(); x_plus[i] += epsilon
        x_minus = x_eq.copy(); x_minus[i] -= epsilon
        A[:, i] = (dynamics_wrapper(x_plus, u_eq_lin[0]) - dynamics_wrapper(x_minus, u_eq_lin[0])) / (2 * epsilon)

    u_plus = u_eq_lin[0] + epsilon
    u_minus = u_eq_lin[0] - epsilon
    B[:, 0] = (dynamics_wrapper(x_eq, u_plus) - dynamics_wrapper(x_eq, u_minus)) / (2 * epsilon)

    # --- Compute LQR gain K ---
    try:
        K, S, E = control.lqr(A, B, Q, R)
        print(f"LQR Gain K computed for target q1={q1_eq:.2f}, q2={q2_eq:.2f}: {K}")
        if K.ndim == 1: K = K.reshape(1, -1) # Ensure K is 2D (1x4)
        return K, u_eq # Return gain K and scalar equilibrium torque u_eq
    except Exception as e:
        print(f"ERROR: Error computing LQR gain for target q1={q1_eq}, q2={q2_eq}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for LQR errors
        return None, None