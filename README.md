# Orbital Trajectory Calculator

## Creating Environment
	 conda env create -f conda_env.yml
## Activate Environment
    conda activate Satmind
## Training
    python3 training.py
To train with CUDA device, modify the device parameter in the training script:

     model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cuda", tau=0.01)
