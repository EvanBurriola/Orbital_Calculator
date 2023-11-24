# Orbital Trajectory Calculator
![](https://github.com/EvanBurriola/Orbital_Calculator/blob/main/results/orbit_visualizer.gif)

## Creating Environment
	 conda env create -f conda_env.yml
## Activate Environment
    conda activate Satmind
## Training
    python3 training.py [DDPG/TD3]
To train with CUDA device, modify the device parameter in the training script:

     model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cuda", tau=0.01)
