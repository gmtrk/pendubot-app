# controllers.py

import numpy as np
import control # For LQR
from pendubot_dynamics import pendubot_dynamics, G # Use G for equilibrium torque
import traceback

# --- Default Controller Parameters ---
DEFAULT_CONTROLLER_PARAMS = {
    # PID Gains might need significant tuning after adding gravity compensation
    'pid': {'Kp': 30.0, 'Ki': 5.0, 'Kd': 10.0, 'max_integral': 10.0, 'max_torque': 15.0},
    'lqr': {
            'Q': np.diag([5.0, 20.0, 0.1, 1.0]), # Weights for q1, q2, dq1, dq2
            'R': np.array([[0.02]]),            # Weight for control effort tau1
            'max_torque': 20.0
           }
}

# --- Helper: Calculate Equilibrium Torque ---
def calculate_equilibrium_torque(q1, q2, m1, l1, m2, l2, lc1, lc2):
    """Calculates the feedforward torque needed to counteract gravity at state (q1, q2).
       Assumes q1 is angle from downward vertical (standard convention)."""
    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    g1_eq = (m1 * lc1 + m2 * l1) * G * s1 + m2 * lc2 * G * s12
    # g2_eq = m2 * lc2 * G * s12 # We don't need g2 here
    return g1_eq

# --- Control Function Implementations ---

def control_none(t, x, controller_state):
    """No control input."""
    return 0.0

def control_pid(t, x, controller_state):
    """PID controller aiming for target_q2, with gravity compensation."""
    q1, q2, dq1, dq2 = x # Standard convention state
    params = controller_state

    Kp = params.get('Kp', 0.0)
    Ki = params.get('Ki', 0.0)
    Kd = params.get('Kd', 0.0)
    target_q2 = params.get('target_q2', 0.0) # Target relative angle
    u_eq = params.get('u_eq', 0.0)           # Gravity compensation torque
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

    # PID feedback calculation
    pid_feedback = Kp * error + Ki * current_integral + Kd * derivative_error

    # --- Final Torque = PID Feedback + Gravity Feedforward ---
    tau1 = pid_feedback + u_eq

    # Limit total torque
    tau1 = np.clip(tau1, -max_torque, max_torque)
    return float(tau1)

def control_lqr(t, x, controller_state):
    """LQR controller (Full state feedback + feedforward)."""
    params = controller_state
    K = params.get('K')
    target_state = params.get('target_state') # Target state in standard convention
    u_eq = params.get('u_eq', 0.0)
    max_torque = params.get('max_torque', 20.0)

    if K is None or target_state is None: return 0.0

    error_x = x - target_state
    error_x[0] = (error_x[0] + np.pi) % (2 * np.pi) - np.pi
    error_x[1] = (error_x[1] + np.pi) % (2 * np.pi) - np.pi

    if K.shape[1] != len(error_x):
         print(f"ERROR: LQR K shape {K.shape} incompatible with error_x shape {error_x.shape}")
         feedback_torque = 0.0
    else:
         feedback_torque_array = -K @ error_x
         feedback_torque = feedback_torque_array[0]

    tau1 = feedback_torque + u_eq
    tau1 = np.clip(tau1, -max_torque, max_torque)
    return float(tau1)

# --- LQR Gain Calculation ---
def calculate_lqr_gain(m1, l1, m2, l2, lc1, lc2, I1, I2, Q, R, q1_eq, q2_eq):
    """Linearizes around (q1_eq, q2_eq, 0, 0) standard convention, returns K and u_eq."""
    x_eq = np.array([q1_eq, q2_eq, 0.0, 0.0])
    # Calculate equilibrium torque u_eq using standard angles
    u_eq = calculate_equilibrium_torque(q1_eq, q2_eq, m1, l1, m2, l2, lc1, lc2)
    # print(f"INFO: Equilibrium torque u_eq for target is approx {u_eq:.2f}") # Reduce noise

    # --- Numerical linearization ---
    epsilon = 1e-6
    u_eq_lin = np.array([0.0]) # Linearize around zero input relative to u_eq
    def dynamics_wrapper(x_standard, u_val):
        return pendubot_dynamics(0, x_standard, m1, l1, m2, l2, lc1, lc2, I1, I2,
                                 lambda t, state, ctrl_state: u_val, {})[:4]

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    try:
        for i in range(4):
            x_plus = x_eq.copy(); x_plus[i] += epsilon
            x_minus = x_eq.copy(); x_minus[i] -= epsilon
            A[:, i] = (dynamics_wrapper(x_plus, u_eq_lin[0]) - dynamics_wrapper(x_minus, u_eq_lin[0])) / (2 * epsilon)
        u_plus = u_eq_lin[0] + epsilon; u_minus = u_eq_lin[0] - epsilon
        B[:, 0] = (dynamics_wrapper(x_eq, u_plus) - dynamics_wrapper(x_eq, u_minus)) / (2 * epsilon)
    except Exception as e:
         print(f"ERROR during numerical differentiation for LQR A/B matrices: {e}")
         traceback.print_exc(); return None, None

    # --- Compute LQR gain K ---
    try:
        K, S, E = control.lqr(A, B, Q, R)
        print(f"LQR Gain K computed for target q1={q1_eq:.2f}, q2={q2_eq:.2f}: {K}") # Reduce noise
        if K.ndim == 1: K = K.reshape(1, -1)
        return K, u_eq # Return gain K and scalar equilibrium torque u_eq
    except Exception as e:
        print(f"ERROR: Error computing LQR gain (control.lqr) for target q1={q1_eq}, q2={q2_eq}: {e}")
        traceback.print_exc(); return None, None