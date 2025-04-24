import numpy as np

G = 9.81 # Gravity

def pendubot_dynamics(t, x, m1, l1, m2, l2, lc1, lc2, I1, I2, tau_func, current_controller_state):
    """
    Calculates the state derivative dx/dt for the Pendubot.
    x = [q1, q2, dq1, dq2]
    tau_func is a function tau_func(t, x, current_controller_state) -> tau1
    current_controller_state holds controller-specific runtime data (e.g., PID integral).
    """
    q1, q2, dq1, dq2 = x
    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s12 = np.sin(q1 + q2)
    # c12 = np.cos(q1 + q2) # Not strictly needed for EOM below

    # --- Calculate Mass Matrix M ---
    # Corrected terms based on common Pendubot derivations
    m11 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2) + I1 + I2
    m12 = m2 * (lc2**2 + l1 * lc2 * c2) + I2
    m22 = m2 * lc2**2 + I2
    M = np.array([[m11, m12],
                  [m12, m22]])

    # --- Calculate Coriolis and Centrifugal Term C ---
    c1_term = -m2 * l1 * lc2 * s2 * dq2**2 - 2 * m2 * l1 * lc2 * s2 * dq1 * dq2
    c2_term = m2 * l1 * lc2 * s2 * dq1**2
    C = np.array([c1_term, c2_term])

    # --- Calculate Gravity Term G ---
    g1_term = (m1 * lc1 + m2 * l1) * G * s1 + m2 * lc2 * G * s12
    g2_term = m2 * lc2 * G * s12
    Grav = np.array([g1_term, g2_term])

    # --- Control Input ---
    # Calculate torque using the selected control function
    tau1 = tau_func(t, x, current_controller_state)
    Torques = np.array([tau1, 0]) # Torque only on the first joint

    # --- Calculate Acceleration ddq ---
    try:
        # Use solve for M*ddq = RHS instead of inverse for numerical stability
        RHS = Torques - C - Grav
        ddq = np.linalg.solve(M, RHS)
        # M_inv = np.linalg.inv(M) # Less stable method
        # ddq = M_inv @ (Torques - C - Grav)
    except np.linalg.LinAlgError:
        print("Warning: Mass matrix became singular. Setting acceleration to zero.")
        ddq = np.zeros(2) # Avoid crashing if matrix is singular

    # --- State Derivative ---
    dx = np.array([dq1, dq2, ddq[0], ddq[1]])
    return dx