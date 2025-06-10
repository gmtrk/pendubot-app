"""
This script uses a Genetic Algorithm (GA) to tune a conventional PID controller.

The fitness function is adapted from the multiobjective optimization concept
described in the paper "Auto-Tuning Fuzzy PID Control of a Pendubot System."
It optimizes for two criteria simultaneously:
1. Performance: Fast settling time and low error, using a time-weighted integral.
2. Effort: Minimizing the required motor torque.
"""
import numpy as np
import scipy.integrate
import pygad
import time
import os
try:
    from src.core_logic.pendubot_dynamics import G
    from src.core_logic.controllers import (control_pid,
                                            calculate_equilibrium_torque)
except ImportError:
    print("ERROR: Could not import project modules (pendubot_dynamics, controllers).")
    print("Ensure this script is in the correct directory or PYTHONPATH is set.")
    exit()

def pendubot_dynamics_with_fixed_tau(t, y, m1, l1, m2, l2, lc1, lc2, I1, I2, tau1):
    q1, q2, dq1, dq2 = y
    s1, c1, s2, c2, s12 = np.sin(q1), np.cos(q1), np.sin(q2), np.cos(q2), np.sin(q1 + q2)
    m11 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I1 + I2
    m12 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2;
    m22 = m2 * lc2 ** 2 + I2
    M = np.array([[m11, m12], [m12, m22]])
    c1_term = -m2 * l1 * lc2 * s2 * dq2 ** 2 - 2 * m2 * l1 * lc2 * s2 * dq1 * dq2
    c2_term = m2 * l1 * lc2 * s2 * dq1 ** 2
    C = np.array([c1_term, c2_term])
    g1_term = (m1 * lc1 + m2 * l1) * G * s1 + m2 * lc2 * G * s12;
    g2_term = m2 * lc2 * G * s12
    Grav = np.array([g1_term, g2_term])
    try:
        ddq = np.linalg.solve(M, np.array([tau1, 0.0]) - C - Grav)
    except np.linalg.LinAlgError:
        return np.array([0, 0, 1e6, 1e6])
    return np.array([dq1, dq2, ddq[0], ddq[1]])


def run_simulation(pid_gains, target_state, p_params, sim_duration, dt):
    m1, l1, m2, l2, lc1, lc2, I1, I2 = p_params
    initial_state = np.array([target_state[0] + 0.05, target_state[1] - 0.05, 0, 0])
    controller_state = {
        'Kp_q1': pid_gains[0], 'Ki_q1': pid_gains[1], 'Kd_q1': pid_gains[2],
        'Kp_q2': pid_gains[3], 'Ki_q2': pid_gains[4], 'Kd_q2': pid_gains[5],
        'dt': dt, 'target_state': target_state,
        'u_eq': calculate_equilibrium_torque(target_state[0], target_state[1], *p_params[:6]),
        'max_integral_q1': 5.0, 'max_integral_q2': 10.0, 'max_torque': 25.0
    }
    time_points = np.arange(0, sim_duration, dt)
    state_history, tau_history = np.zeros((len(time_points), 4)), np.zeros(len(time_points))
    state = initial_state
    for i, t in enumerate(time_points):
        state_history[i, :], tau_history[i] = state, control_pid(t, state, controller_state)
        sol = scipy.integrate.solve_ivp(
            pendubot_dynamics_with_fixed_tau, [t, t + dt], state, t_eval=[t + dt], args=(*p_params, tau_history[i]))
        if sol.status != 0: return None, None, False, None
        state = sol.y[:, -1]
        if any(abs(s) > 30 for s in state): return None, None, False, None
    return state_history, tau_history, True, time_points


class PIDTuner:
    def __init__(self, target_config, physical_params):
        self.target_config = target_config
        self.p_params = physical_params
        self.dt = 0.02
        self.duration = 15.0
        self.objective_weights = {'w1_perf': 0.85, 'w2_effort': 0.15}

    def fitness_func(self, ga_instance, solution, solution_idx):
        target_state_vec = np.array([self.target_config[0], self.target_config[1], 0, 0])
        state_hist, tau_hist, is_stable, time_vec = run_simulation(
            pid_gains=solution, target_state=target_state_vec, p_params=self.p_params,
            sim_duration=self.duration, dt=self.dt
        )
        if not is_stable: return -1e6

        # --- Multiobjective Fitness Calculation ---
        # Objective 1: Performance
        # Inverse of the Integrated Time-multiplied Squared Error (ITSE)
        error_q1 = target_state_vec[0] - np.unwrap(state_hist[:, 0])
        error_q2 = target_state_vec[1] - np.unwrap(state_hist[:, 1])
        itse_cost = np.sum(time_vec * (error_q1 ** 2 + error_q2 ** 2)) * self.dt + 1e-9
        f1_performance = 1.0 / itse_cost

        # Objective 2: Control Effort
        # Inverse of the Integrated Squared Torque
        effort_cost = np.sum(tau_hist ** 2) * self.dt + 1e-9
        f2_effort = 1.0 / effort_cost

        w1 = self.objective_weights['w1_perf']
        w2 = self.objective_weights['w2_effort']

        total_fitness = (w1 * f1_performance) + (w2 * f2_effort * 1e-3)

        return total_fitness


if __name__ == "__main__":
    m1 = 0.8
    l1 = 1.0
    m2 = 0.2
    l2 = 0.5
    lc1 = l1
    lc2 = l2 / 2.0
    I1 = (1 / 3) * m1 * l1 ** 2
    I2 = (1 / 12) * m2 * l2 ** 2
    physical_params = (m1, l1, m2, l2, lc1, lc2, I1, I2)

    targets = {
        "1": {"name": "Up-Up", "config": (np.pi, 0.0)},
        "2": {"name": "Down-Up", "config": (0.0, np.pi)},
        "3": {"name": "3pi/4, pi/4", "config": (3 * np.pi / 4, np.pi / 4)},
        "4": {"name": "-3pi/4, -pi/4", "config": (-3 * np.pi / 4, -np.pi / 4)}
    }
    num_threads = os.cpu_count()

    ga_params = {
        'num_generations': 10, 'num_parents_mating': 30, 'sol_per_pop': 240,
        'parent_selection_type': "sss", 'crossover_type': "scattered",
        'mutation_type': "adaptive", 'mutation_num_genes': (4, 2),
        'keep_elitism': 5,
    }

    gene_space = [
        {'low': 0, 'high': 300}, {'low': 0, 'high': 200}, {'low': 0, 'high': 100},  # q1 gains
        {'low': 0, 'high': 300}, {'low': 0, 'high': 200}, {'low': 0, 'high': 100}  # q2 gains
    ]

    print("--- Starting PID Tuning with Paper-Inspired Multiobjective Fitness ---")

    print("\nAvailable target configurations:")
    for key, val in targets.items():
        print(f"  {key}: {val['name']} (q1={val['config'][0]:.2f}, q2={val['config'][1]:.2f})")

    while True:
        choice = input("Enter the number of the configuration you want to tune (e.g., 1 for Up-Up): ")
        if choice in targets:
            selected_target = targets[choice]
            name = selected_target['name']
            config = selected_target['config']
            break
        else:
            print("Invalid choice. Please enter a valid number.")

    print(f"\n🚀 Tuning for Target '{name}': q1={config[0]:.2f}, q2={config[1]:.2f}")
    start_time = time.time()

    tuner = PIDTuner(target_config=config, physical_params=physical_params)
    ga_instance = pygad.GA(
        num_genes=6, gene_space=gene_space, fitness_func=tuner.fitness_func,
        on_generation=lambda ga: print(
            f"  Gen {ga.generations_completed:03d} | Best Fitness: {ga.best_solution(pop_fitness=ga.last_generation_fitness)[1]:.4f}"),
        **ga_params
    )
    ga_instance.run()
    end_time = time.time()
    solution, solution_fitness, _ = ga_instance.best_solution()

    print(f"   Tuning complete in {end_time - start_time:.2f} seconds.")
    print(f"   Best Fitness Score: {solution_fitness:.4f}")
    print("   Optimal PID Gains Found:")
    print(f"   - Kp_q1: {solution[0]:.4f}, Ki_q1: {solution[1]:.4f}, Kd_q1: {solution[2]:.4f}")
    print(f"   - Kp_q2: {solution[3]:.4f}, Ki_q2: {solution[4]:.4f}, Kd_q2: {solution[5]:.4f}")

    print("\n--- Tuning task completed. ---")