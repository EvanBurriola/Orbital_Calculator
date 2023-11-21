import sys
import numpy as np
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from gym_env import OrekitEnv

# Page 35 RP(A Reinforcement Learning Approach to Spacecraft Trajectory);
# [a(m), e, i(deg), omega/w(deg), Omega/raan(deg), TrueAnomaly(v)]
initial_state = [5500*1e3, 0.20,5.0, 20.0, 20.0,10.0]
target_state = [6300*1e3, 0.23, 5.3, 24.0, 24.0, 10.0]
simulation_date = [2018, 2, 16, 12, 10, 0.0]
simulation_duration = 24.0 * 60.0 ** 2 * 4
spacecraft_mass = [500.0, 150.0]
# Let Spacecraft take an action every (step) amount of seconds
simulation_stepT = 500.0

n = len(sys.argv)
if(n < 2):
    print("Usage: python3 training.py [DDPG/TD3]")
    sys.exit()

# Create environment instance
env = OrekitEnv(initial_state, target_state, simulation_date, simulation_duration, spacecraft_mass, simulation_stepT)
# Get action space from environment
n_actions = env.action_space.shape[-1]
# Define the action noise (continuous action space)
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

alg = sys.argv[1]
if(alg == "DDPG"):
    # Create the TD3 model
    model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cpu", tau=0.01)
elif(alg == "TD3"):
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cpu", tau=0.01)
else:
    print("Usage: python3 training.py [DDPG/TD3]")
    sys.exit()

env.alg = alg

# Options for loading existing model
# model = DDPG.load("ddpg_model", device="cpu")
# model.set_env(env)

# Train & save model
model.learn(total_timesteps=415000, log_interval=10)
model.save('models/'+str(env.id)+"_"+ alg +"_model")

# Generate .txt of reward/episode trained
env.write_reward()
