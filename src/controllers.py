import numpy as np
import control  # For LQR
from pendubot_dynamics import pendubot_dynamics, G  # Use G for equilibrium torque
import traceback

# --- Default Controller Parameters ---
DEFAULT_CONTROLLER_PARAMS = {
    'pid': {
        # User will tune these as POSITIVE values in GUI.
        'Kp_q1': 16.0, 'Ki_q1': 0.0, 'Kd_q1': 0.0,
        'Kp_q2': 26.0, 'Ki_q2': 0.0, 'Kd_q2': 0.0,
        'max_integral_q1': 5.0,
        'max_integral_q2': 10.0,
        'max_torque': 20.0
    },
    'lqr': {
        'Q': np.diag([5.0, 20.0, 0.1, 1.0]),
        'R': np.array([[0.02]]),
        'max_torque': 20.0
    }
}


# --- Helper: Calculate Equilibrium Torque ---
def calculate_equilibrium_torque(q1, q2, m1, l1, m2, l2, lc1, lc2):
    s1 = np.sin(q1)
    s12 = np.sin(q1 + q2)
    g1_eq = (m1 * lc1 + m2 * l1) * G * s1 + m2 * lc2 * G * s12
    return g1_eq


# --- Control Function Implementations ---

def control_none(t, x, controller_state):
    return 0.0


def control_pid(t, x, controller_state):
    """Full State Feedback PID controller with gravity compensation.
       Proportional terms are applied with an inverted sign based on empirical findings."""
    q1, q2, dq1, dq2 = x  # Current state in standard convention
    params = controller_state

    target_state = params.get('target_state', np.zeros(4))
    target_q1, target_q2, target_dq1, target_dq2 = target_state

    Kp_q1 = params.get('Kp_q1', 0.0)
    Ki_q1 = params.get('Ki_q1', 0.0)
    Kd_q1 = params.get('Kd_q1', 0.0)
    Kp_q2 = params.get('Kp_q2', 0.0)
    Ki_q2 = params.get('Ki_q2', 0.0)
    Kd_q2 = params.get('Kd_q2', 0.0)

    u_eq = params.get('u_eq', 0.0)
    dt = params.get('dt', 0.02)
    max_integral_q1 = params.get('max_integral_q1', 5.0)
    max_integral_q2 = params.get('max_integral_q2', 10.0)
    max_torque = params.get('max_torque', 15.0)

    # Error Calculation (error = target - current)
    error_q1 = target_q1 - q1
    error_q1 = (error_q1 + np.pi) % (2 * np.pi) - np.pi

    error_q2 = target_q2 - q2
    error_q2 = (error_q2 + np.pi) % (2 * np.pi) - np.pi

    # Integral Terms
    integral_error_q1 = params.get('integral_error_q1', 0.0) + error_q1 * dt
    integral_error_q1 = np.clip(integral_error_q1, -max_integral_q1, max_integral_q1)
    params['integral_error_q1'] = integral_error_q1

    integral_error_q2 = params.get('integral_error_q2', 0.0) + error_q2 * dt
    integral_error_q2 = np.clip(integral_error_q2, -max_integral_q2, max_integral_q2)
    params['integral_error_q2'] = integral_error_q2

    # Derivative Terms (de/dt = d(target)/dt - d(current)/dt = 0 - dq = -dq)
    # This provides damping if Kd is positive.
    derivative_error_q1 = target_dq1 - dq1  # Effectively -dq1
    derivative_error_q2 = target_dq2 - dq2  # Effectively -dq2

    # --- PID Feedback Calculation ---
    # Apply Kp terms with a NEGATIVE sign
    feedback_q1 = (-Kp_q1 * error_q1) + (Ki_q1 * integral_error_q1) + (Kd_q1 * derivative_error_q1)
    feedback_q2 = (-Kp_q2 * error_q2) + (Ki_q2 * integral_error_q2) + (Kd_q2 * derivative_error_q2)

    pid_feedback = feedback_q1 + feedback_q2

    tau1 = pid_feedback + u_eq
    tau1 = np.clip(tau1, -max_torque, max_torque)
    return float(tau1)


def control_lqr(t, x, controller_state):
    params = controller_state
    K = params.get('K')
    target_state = params.get('target_state')
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


def calculate_lqr_gain(m1, l1, m2, l2, lc1, lc2, I1, I2, Q, R, q1_eq, q2_eq):
    x_eq = np.array([q1_eq, q2_eq, 0.0, 0.0])
    u_eq = calculate_equilibrium_torque(q1_eq, q2_eq, m1, l1, m2, l2, lc1, lc2)
    epsilon = 1e-6
    u_eq_lin = np.array([0.0])

    def dynamics_wrapper(x_standard, u_val):
        return pendubot_dynamics(0, x_standard, m1, l1, m2, l2, lc1, lc2, I1, I2,
                                 lambda t, state, ctrl_state: u_val, {})[:4]

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))
    try:
        for i in range(4):
            x_plus = x_eq.copy()
            x_plus[i] += epsilon
            x_minus = x_eq.copy()
            x_minus[i] -= epsilon
            A[:, i] = (dynamics_wrapper(x_plus, u_eq_lin[0]) - dynamics_wrapper(x_minus, u_eq_lin[0])) / (2 * epsilon)
        u_plus = u_eq_lin[0] + epsilon
        u_minus = u_eq_lin[0] - epsilon
        B[:, 0] = (dynamics_wrapper(x_eq, u_plus) - dynamics_wrapper(x_eq, u_minus)) / (2 * epsilon)
    except Exception as e:
        print(f"ERROR during LQR A/B differentiation: {e}")
        traceback.print_exc()
        return None, None
    try:
        K, S, E = control.lqr(A, B, Q, R)
        if K.ndim == 1: K = K.reshape(1, -1)
        return K, u_eq
    except Exception as e:
        print(f"ERROR computing LQR gain: {e}")
        traceback.print_exc()
        return None, None