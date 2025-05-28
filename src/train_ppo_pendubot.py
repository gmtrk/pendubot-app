import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import the custom environment
from pendubot_env import PendubotEnv

# --- Parameters ---
LOG_DIR = "ppo_pendubot_logs/"
MODEL_SAVE_PATH = "ppo_pendubot_model"
TENSORBOARD_LOG_DIR = "./ppo_pendubot_tensorboard/"

TOTAL_TIMESTEPS = 10000000  # Adjust as needed (e.g., 1e6, 2e6 for better results)
N_ENVS = 16 # Number of parallel environments
SAVE_FREQ = 100000 # Save a checkpoint every N steps (adjust per N_ENVS)

def train():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

    # Create vectorized environments
    # Using a lambda to ensure fresh environment creation for each process
    env = make_vec_env(lambda: PendubotEnv(), n_envs=N_ENVS)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // N_ENVS, 1), # Adjust save_freq for n_envs
        save_path=LOG_DIR,
        name_prefix="ppo_pendubot_checkpoint"
    )

    # PPO model
    # Hyperparameters can be tuned. These are reasonable defaults.
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        n_steps=1024, # Number of steps to run for each environment per update
        batch_size=64, # Minibatch size
        n_epochs=10,   # Number of epoch when optimizing the surrogate loss
        gamma=0.99,    # Discount factor
        gae_lambda=0.95, # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,  # Clipping parameter PPO
        ent_coef=0.0,   # Entropy coefficient
        vf_coef=0.5,    # Value function coefficient
        max_grad_norm=0.5, # Max value for gradient clipping
        # learning_rate=3e-4, # Can specify learning rate
        device="cpu" # "cuda" if GPU available, else "cpu"
    )

    print("Starting PPO training...")
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print(f"Saving final model to {MODEL_SAVE_PATH}.zip")
        model.save(MODEL_SAVE_PATH)
        env.close() # Important to close vectorized environments

    print("Training finished.")
    print(f"Model saved as {MODEL_SAVE_PATH}.zip")
    print(f"Checkpoints saved in {LOG_DIR}")
    print(f"Tensorboard logs in {TENSORBOARD_LOG_DIR}")
    print(f"To view tensorboard, run: tensorboard --logdir {TENSORBOARD_LOG_DIR}")

if __name__ == "__main__":
    train()