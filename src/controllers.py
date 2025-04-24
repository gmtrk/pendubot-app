import numpy as np
import control # For LQR
from tkinter import messagebox # For showing LQR calculation errors

# Import dynamics function only needed for linearization in LQR gain calculation
from pendubot_dynamics import pendubot_dynamics

# --- Default Controller Parameters (Structure & Default Gains) ---
# The actual runtime state (like integral error) will be managed by the app class.
DEFAULT_CONTROLLER_PARAMS = {
    'pid': {'Kp': 25.0, 'Ki': 2.0, 'Kd': 5.0, 'target_q2': 0.0, 'max_integral': 10.0, 'max_torque': 15.0},
    'lqr': {'target_state': np.array([np.pi, 0.0, 0.0, 0.0]), # Target: q1=pi (down), q2=0 (up)
            'Q': np.diag([0.1, 10.0, 0.1, 1.0]), # Weights for q1, q2, dq1, dq2 deviations
            'R': np.array([[0.1]]),             # Weight for control effort tau1
            'max_torque': 20.0
           }
}

# --- Control Function Implementations ---

def control_none(t, x, controller_state):
    """No control input."""
    return 0.0

def control_pid(t, x, controller_state):
    """
    PID controller to stabilize q2 around target_q2.
    controller_state is a dictionary holding runtime PID state like:
    {'integral_error': value, 'last_error': value, 'dt': value,
     'Kp', 'Ki', 'Kd', 'target_q2', 'max_integral', 'max_torque'}
    """
    q1, q2, dq1, dq2 = x
    params = controller_state # Use the runtime state/parameters

    # Extract params safely
    Kp = params.get('Kp', 0.0)
    Ki = params.get('Ki', 0.0)
    Kd = params.get('Kd', 0.0)
    target_q2 = params.get('target_q2', 0.0)
    dt = params.get('dt', 0.02) # Default dt if not provided
    max_integral = params.get('max_integral', 10.0)
    max_torque = params.get('max_torque', 15.0)


    # Wrap q2 error correctly if target is 0 (upright)
    # Error: target - current. Angle difference needs care.
    error = target_q2 - q2
    error = (error + np.pi) % (2 * np.pi) - np.pi # Normalize error to [-pi, pi]

    # Integral term
    current_integral = params.get('integral_error', 0.0)
    current_integral += error * dt
    # Clamp integral term to prevent windup
    current_integral = np.clip(current_integral, -max_integral, max_integral)
    params['integral_error'] = current_integral # Update state

    # Derivative term (using velocity directly is often better than finite diff)
    # Use -dq2 as derivative of error (d(target-q2)/dt = -dq2 if target is constant)
    derivative_error = -dq2
    # Or using finite difference (less ideal usually):
    # last_error = params.get('last_error', error) # Get last error or use current if first step
    # derivative_error = (error - last_error) / dt if dt > 0 else 0.0
    # params['last_error'] = error # Update state

    # PID control law
    tau1 = Kp * error + Ki * current_integral + Kd * derivative_error

    # Limit torque
    tau1 = np.clip(tau1, -max_torque, max_torque)

    return tau1

def control_lqr(t, x, controller_state):
    """
    LQR controller to stabilize around target_state.
    controller_state holds runtime LQR state: {'K': gain_matrix, 'target_state': target, 'max_torque': value}
    """
    params = controller_state

    K = params.get('K')
    target_state = params.get('target_state', np.zeros(4))
    max_torque = params.get('max_torque', 20.0)

    if K is None or K.size == 0:
        print("Warning: LQR gain K not available in controller_state.")
        return 0.0 # No gain available

    # Calculate error state (deviation from target)
    error_x = x - target_state
    # Wrap angle errors correctly
    error_x[0] = (error_x[0] + np.pi) % (2 * np.pi) - np.pi # q1 error wrap
    error_x[1] = (error_x[1] + np.pi) % (2 * np.pi) - np.pi # q2 error wrap

    # LQR control law: u = -K * error_x
    # Note: u is a vector, we need tau1 which is u[0]
    tau1 = (-K @ error_x)[0]

    # Limit torque
    tau1 = np.clip(tau1, -max_torque, max_torque)

    return tau1


# --- LQR Gain Calculation ---

def calculate_lqr_gain(m1, l1, m2, l2, lc1, lc2, I1, I2, Q, R):
    """
    Linearizes the system around the unstable upright equilibrium (q1=pi, q2=0)
    and computes the LQR gain K using the provided Q and R matrices.
    Returns the gain matrix K or None if calculation fails.
    """
    # Define the upright equilibrium state [q1, q2, dq1, dq2] for linearization
    x_eq = np.array([np.pi, 0.0, 0.0, 0.0])
    u_eq = np.array([0.0]) # Equilibrium torque is 0

    # Numerical linearization using finite differences
    epsilon = 1e-6

    def dynamics_wrapper(x, u_val):
        # Wrapper for pendubot_dynamics suitable for linearization
        # Uses a dummy control function that just returns the input 'u_val'.
        # State dimension is 4.
        return pendubot_dynamics(0, x, m1, l1, m2, l2, lc1, lc2, I1, I2,
                                 lambda t, state, ctrl_state: u_val, {})[:4]

    A = np.zeros((4, 4))
    B = np.zeros((4, 1))

    # Calculate A matrix (df/dx)
    for i in range(4):
        x_plus = x_eq.copy()
        x_plus[i] += epsilon
        x_minus = x_eq.copy()
        x_minus[i] -= epsilon
        A[:, i] = (dynamics_wrapper(x_plus, u_eq[0]) - dynamics_wrapper(x_minus, u_eq[0])) / (2 * epsilon)

    # Calculate B matrix (df/du) - u is scalar tau1 here
    u_plus = u_eq[0] + epsilon
    u_minus = u_eq[0] - epsilon
    B[:, 0] = (dynamics_wrapper(x_eq, u_plus) - dynamics_wrapper(x_eq, u_minus)) / (2 * epsilon)

    # Compute LQR gain K
    try:
        K, S, E = control.lqr(A, B, Q, R)
        print(f"LQR Gain K computed: {K}")
        return K
    except Exception as e:
        print(f"Error computing LQR gain: {e}")
        messagebox.showerror("LQR Error", f"Could not compute LQR gain. Check parameters or system stability.\nError: {e}")
        return None