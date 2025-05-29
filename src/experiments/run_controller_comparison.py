import numpy as np
import scipy.integrate
import os
import pandas as pd
import traceback

try:
    from src.core_logic.pendubot_dynamics import pendubot_dynamics, G
    from src.core_logic.controllers import (control_none, control_pid, control_lqr,
                                            calculate_lqr_gain, DEFAULT_CONTROLLER_PARAMS,
                                            calculate_equilibrium_torque)
except ImportError:
    print("ERROR: Could not import project modules (pendubot_dynamics, controllers).")
    print("Ensure this script is in the correct directory or PYTHONPATH is set.")
    exit()

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None
    print("Warning: stable-baselines3 not found. PPO RL Agent will not be tested.")

ABNORMAL_ACCELERATION_THRESHOLD_SCRIPT = 1000.0


class HeadlessPendubotSimulator:
    """
    Class for running Pendubot simulations in "headless" mode (no GUI)
    for data collection and analysis.
    """

    def __init__(self):
        self.m1 = 0.8
        self.l1 = 1.0
        self.m2 = 0.2
        self.l2 = 0.5
        self.lc1 = self.l1
        self.lc2 = self.l2 / 2.0
        self.I1 = (1 / 3) * self.m1 * self.l1 ** 2
        self.I2 = (1 / 12) * self.m2 * self.l2 ** 2
        self.g_sim = G

        self.dt = 0.02
        self.simulation_time_limit = 10.0

        # Simulation state variables
        self.simulation_time = 0.0
        self.state = np.zeros(4)  # [q1, q2, dq1, dq2] standard convention
        self.target_state = np.zeros(4)  # [tq1, tq2, 0, 0] standard convention

        self.control_func = control_none
        self.current_controller_state = {}
        self.ppo_model = None

        # Metrics accumulators
        self.iae_q1 = 0.0
        self.iae_q2 = 0.0
        self.control_effort_integral = 0.0  # Sum (tau1^2 * dt)
        self.current_tau1 = 0.0  # Last applied torque

    def setup_episode(self, target_q1, target_q2, controller_type, specific_pid_gains=None):
        """Configures a new simulation episode."""
        self.iae_q1 = 0.0
        self.iae_q2 = 0.0
        self.control_effort_integral = 0.0
        self.current_tau1 = 0.0
        self.simulation_time = 0.0

        self.target_state = np.array([target_q1, target_q2, 0.0, 0.0])

        perturb_angle = 0.05  # Small perturbation from target state
        q1_init = target_q1 + np.random.uniform(-perturb_angle, perturb_angle)
        q2_init = target_q2 + np.random.uniform(-perturb_angle, perturb_angle)
        self.state = np.array([q1_init, q2_init, 0.0, 0.0])  # Zero initial velocities

        self.current_controller_state = {}
        u_eq = calculate_equilibrium_torque(target_q1, target_q2,
                                            self.m1, self.l1, self.m2, self.l2,
                                            self.lc1, self.lc2)

        if controller_type == "PID":
            self.control_func = control_pid
            if specific_pid_gains is None:
                print("    WARNING: No specific PID gains provided, using full defaults.")
                current_gains = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
            else:
                current_gains = DEFAULT_CONTROLLER_PARAMS['pid'].copy()
                current_gains.update(specific_pid_gains)

            self.current_controller_state = current_gains
            self.current_controller_state['dt'] = self.dt
            self.current_controller_state['integral_error_q1'] = 0.0
            self.current_controller_state['integral_error_q2'] = 0.0
            self.current_controller_state['target_state'] = self.target_state
            self.current_controller_state['u_eq'] = u_eq

        elif controller_type == "LQR":
            self.control_func = control_lqr
            Q_lqr = DEFAULT_CONTROLLER_PARAMS['lqr']['Q']
            R_lqr = DEFAULT_CONTROLLER_PARAMS['lqr']['R']
            K, u_eq_lqr = calculate_lqr_gain(self.m1, self.l1, self.m2, self.l2,
                                             self.lc1, self.lc2, self.I1, self.I2,
                                             Q_lqr, R_lqr, q1_eq=target_q1, q2_eq=target_q2)
            if K is None:
                print(f"    LQR Gain calculation failed for target ({target_q1:.2f}, {target_q2:.2f}). Cannot run.")
                return False
            self.current_controller_state = DEFAULT_CONTROLLER_PARAMS['lqr'].copy()
            self.current_controller_state['K'] = K
            self.current_controller_state['target_state'] = self.target_state
            self.current_controller_state['u_eq'] = u_eq_lqr

        elif controller_type == "PPO":
            if PPO is None:
                print("    PPO library (Stable Baselines3) not available. Skipping PPO test.")
                return False
            self.control_func = self._control_ppo_headless
            if self.ppo_model is None:
                try:
                    model_path = "../../data/ppo_pendubot_model.zip"
                    if not os.path.exists(model_path):
                        model_path = os.path.join("../../data/ppo_pendubot_logs", "ppo_pendubot_model.zip")
                    if not os.path.exists(model_path):
                        print(f"    PPO model not found at {model_path} or in logs/. Skipping PPO test.")
                        return False
                    self.ppo_model = PPO.load(model_path)
                    print(f"    PPO Model loaded from {model_path}")
                except Exception as e:
                    print(f"    Error loading PPO model: {e}. Skipping PPO test.")
                    return False
            self.current_controller_state = {}

        elif controller_type == "None":
            self.control_func = control_none
            self.current_controller_state = {}
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")
        return True  # Setup successful

    def _get_ppo_observation_headless(self):
        """ Creates observation vector for PPO (fixed physical params, 10 elements). """
        q1, q2, dq1, dq2 = self.state
        tq1, tq2 = self.target_state[0], self.target_state[1]
        obs = np.array([
            np.cos(q1), np.sin(q1), np.cos(q2), np.sin(q2), dq1, dq2,
            np.cos(tq1), np.sin(tq1), np.cos(tq2), np.sin(tq2)
        ], dtype=np.float32)
        return obs

    def _control_ppo_headless(self, t, x_current_state, controller_state_dict):
        """ Control function for PPO in headless simulation. """
        if self.ppo_model is None: return 0.0
        observation = self._get_ppo_observation_headless()
        action, _ = self.ppo_model.predict(observation, deterministic=True)
        return float(action[0])

    def _run_simulation_step(self):
        """Executes one simulation step and updates metrics."""
        if not self.running: return False

        self.current_tau1 = self.control_func(self.simulation_time, self.state, self.current_controller_state)

        def check_acceleration_event_headless(t, y, m1_arg, l1_arg, m2_arg, l2_arg,
                                              lc1_arg, lc2_arg, I1_arg, I2_arg,
                                              control_func_h, controller_state_h):
            q1s, q2s, dq1s, dq2s = y
            tau1_event = control_func_h(t, y, controller_state_h)
            s1, c1 = np.sin(q1s), np.cos(q1s)
            s2, c2 = np.sin(q2s), np.cos(q2s)
            s12 = np.sin(q1s + q2s)
            m11 = m1_arg * lc1_arg ** 2 + m2_arg * (
                        l1_arg ** 2 + lc2_arg ** 2 + 2 * l1_arg * lc2_arg * c2) + I1_arg + I2_arg
            m12 = m2_arg * (lc2_arg ** 2 + l1_arg * lc2_arg * c2) + I2_arg
            m22 = m2_arg * lc2_arg ** 2 + I2_arg
            M = np.array([[m11, m12], [m12, m22]])
            c1t = -m2_arg * l1_arg * lc2_arg * s2 * dq2s ** 2 - 2 * m2_arg * l1_arg * lc2_arg * s2 * dq1s * dq2s
            c2t = m2_arg * l1_arg * lc2_arg * s2 * dq1s ** 2
            C = np.array([c1t, c2t])
            g1t = (m1_arg * lc1_arg + m2_arg * l1_arg) * self.g_sim * s1 + m2_arg * lc2_arg * self.g_sim * s12
            g2t = m2_arg * lc2_arg * self.g_sim * s12
            Grav = np.array([g1t, g2t])
            Torques = np.array([tau1_event, 0.0])
            try:
                RHS = Torques - C - Grav
                ddq = np.linalg.solve(M, RHS)
            except np.linalg.LinAlgError:
                return -1.0
            return ABNORMAL_ACCELERATION_THRESHOLD_SCRIPT - max(abs(ddq[0]), abs(ddq[1]))

        check_acceleration_event_headless.terminal = True
        check_acceleration_event_headless.direction = -1

        common_args = (self.m1, self.l1, self.m2, self.l2,
                       self.lc1, self.lc2, self.I1, self.I2,
                       self.control_func, self.current_controller_state)

        try:
            sol = scipy.integrate.solve_ivp(
                pendubot_dynamics,
                [self.simulation_time, self.simulation_time + self.dt],
                self.state, method='RK45', t_eval=[self.simulation_time + self.dt],
                args=common_args, events=check_acceleration_event_headless)
        except Exception as e:
            print(f"    CRITICAL ERROR during solve_ivp: {e}")
            traceback.print_exc()
            return False

        terminated_by_event = False
        if sol.status == 1:
            if sol.t_events and sol.t_events[0].size > 0:
                terminated_by_event = True
                print(f"    EVENT: Abnormal acceleration at t={sol.t_events[0][0]:.3f}s.") # Reduce noise

        if sol.status != 0 and not terminated_by_event:
            print(f"    Solver failed (Status {sol.status}): {sol.message}")
            return False

        if isinstance(sol.y, np.ndarray) and sol.y.shape[1] > 0:
            self.state = sol.y[:, -1]
        elif terminated_by_event:
            pass
        else:
            print(f"    Solver returned no solution points or failed at t_start. sol.y: {sol.y}")
            return False  # Indicate step failure

        if not np.all(np.isfinite(self.state)):
            print(f"    WARNING: Invalid state at t={self.simulation_time}: {self.state}")
            return False

        self.simulation_time += self.dt

        q1_curr, q2_curr = self.state[0], self.state[1]
        tq1, tq2 = self.target_state[0], self.target_state[1]
        err_q1 = (tq1 - q1_curr + np.pi) % (2 * np.pi) - np.pi
        err_q2 = (tq2 - q2_curr + np.pi) % (2 * np.pi) - np.pi
        self.iae_q1 += abs(err_q1) * self.dt
        self.iae_q2 += abs(err_q2) * self.dt
        self.control_effort_integral += (self.current_tau1 ** 2) * self.dt

        if terminated_by_event: return False
        return True  # Step successful

    def run_full_episode(self):
        """Runs a full simulation episode and returns collected metrics."""
        self.running = True
        step_count = 0
        # Loop until time limit, ensuring one extra dt doesn't push it over
        while self.simulation_time < self.simulation_time_limit - (self.dt / 2.0):
            if not self.running: break
            step_successful = self._run_simulation_step()
            if not step_successful:
                # print(f"    Episode ended prematurely at t={self.simulation_time:.2f}s.") # Reduce noise
                self.running = False
                return {"iae_q1": self.iae_q1, "iae_q2": self.iae_q2,
                        "control_effort": self.control_effort_integral,
                        "time_completed": self.simulation_time,
                        "final_tau1": self.current_tau1, "status": "Failed"}
            step_count += 1

        self.running = False
        return {"iae_q1": self.iae_q1, "iae_q2": self.iae_q2,
                "control_effort": self.control_effort_integral,
                "time_completed": self.simulation_time,
                "final_tau1": self.current_tau1, "status": "Completed"}


# --- Main Experiment Loop ---
def perform_experiments():
    print("Starting Pendubot controller comparison experiments...\n")

    targets = {
        "L1_Up_L2_Align": (np.pi, 0.0),
        "L1_Down_L2_WorldUp": (0.0, np.pi),
        "L1_3pi4_L2_WorldUp": (3 * np.pi / 4, np.pi / 4),
        "L1_-3pi4_L2_WorldUp": (-3 * np.pi / 4, -np.pi / 4),
    }

    pid_default_gains = DEFAULT_CONTROLLER_PARAMS['pid']
    specific_pid_setups = {
        "L1_Up_L2_Align": {'Kp_q1': 16.0, 'Kp_q2': 26.0},
        "L1_Down_L2_WorldUp": {'Kp_q1': 14.8, 'Kp_q2': 5.1},
        "L1_3pi4_L2_WorldUp": {'Kp_q1': 15.4, 'Kp_q2': 22.5},
        "L1_-3pi4_L2_WorldUp": {'Kp_q1': 19.5, 'Kp_q2': 26.0},
    }

    controllers_to_test = ["PID", "LQR"]
    if PPO is not None:
        controllers_to_test.append("PPO")
    else:
        print("PPO will not be tested - Stable Baselines3 library not available.")

    all_results = []
    simulator = HeadlessPendubotSimulator()
    num_runs_per_scenario = 3

    for target_name, (tq1, tq2) in targets.items():
        print(f"--- Testing Target: {target_name} (q1={tq1:.2f}, q2={tq2:.2f}) ---")
        for controller_name in controllers_to_test:
            print(f"  Controller: {controller_name}")

            run_metrics_list = []  # Store metrics for each run to average later

            for run_num in range(num_runs_per_scenario):
                print(f"    Run {run_num + 1}/{num_runs_per_scenario}...", end=" ")

                current_pid_gains_for_run = None
                if controller_name == "PID":
                    base_pid_gains = pid_default_gains.copy()
                    override_gains = specific_pid_setups.get(target_name)
                    if override_gains: base_pid_gains.update(override_gains)
                    current_pid_gains_for_run = base_pid_gains

                setup_ok = simulator.setup_episode(tq1, tq2, controller_name,
                                                   specific_pid_gains=current_pid_gains_for_run)

                if not setup_ok:
                    print(f"SETUP FAILED. Skipping run.")
                    run_metrics = {"iae_q1": float('nan'), "iae_q2": float('nan'),
                                   "control_effort": float('nan'), "time_completed": 0,
                                   "final_tau1": 0, "status": "SetupFailed"}
                else:
                    run_metrics = simulator.run_full_episode()

                run_metrics_list.append(run_metrics)
                print(f"Status: {run_metrics['status']}, Time: {run_metrics['time_completed']:.2f}s")

            # Calculate averages for this controller-target pair
            successful_runs = [m for m in run_metrics_list if m["status"] == "Completed"]
            num_success = len(successful_runs)

            if num_success > 0:
                avg_iae_q1 = sum(m["iae_q1"] for m in successful_runs) / num_success
                avg_iae_q2 = sum(m["iae_q2"] for m in successful_runs) / num_success
                avg_effort = sum(m["control_effort"] for m in successful_runs) / num_success
                avg_time_success = sum(m["time_completed"] for m in successful_runs) / num_success
            else:
                avg_iae_q1 = float('inf')
                avg_iae_q2 = float('inf')
                avg_effort = float('inf')
                avg_time_success = sum(m["time_completed"] for m in
                                       run_metrics_list) / num_runs_per_scenario if num_runs_per_scenario > 0 else 0

            all_results.append({
                "Target": target_name,
                "Controller": controller_name,
                "Avg_IAE_q1": avg_iae_q1,
                "Avg_IAE_q2": avg_iae_q2,
                "Avg_ControlEffort": avg_effort,
                "Avg_CompletionTime_Success": avg_time_success,  # Avg time for successful runs
                "NumSuccesses": f"{num_success}/{num_runs_per_scenario}"
            })
            print(
                f"    Averaged for {controller_name}: IAE_q1={avg_iae_q1:.2f}, IAE_q2={avg_iae_q2:.2f}, Effort={avg_effort:.2f}, Successes={num_success}/{num_runs_per_scenario}\n")

    results_df = pd.DataFrame(all_results)
    print("\n\n--- === Experiment Results Summary === ---")
    pd.set_option('display.precision', 2)
    pd.set_option('display.width', 120)
    print(results_df.to_string(index=False))

    csv_filename = "../../data/pendubot_controller_comparison_results.csv"
    try:
        results_df.to_csv(csv_filename, index=False, float_format='%.2f')
        print(f"\nResults saved to file: {csv_filename}")
    except Exception as e:
        print(f"\nError saving results to CSV: {e}")


if __name__ == "__main__":
    perform_experiments()