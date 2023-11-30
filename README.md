# Orbital Trajectory Calculator

An implementation of a novel approach to the way satellite trajectory changes are performed. Utilizing Reinforcement Learning, specifically the Twin Delayed DDPG (TD3) algorithm, to develop continuous and high precision orbitial transfer actions. An extension of "A Reinforcment Learning Approach to Spacecraft Trajectory Optimization" research by Daniel S. Kolosa.

![](https://github.com/EvanBurriola/Orbital_Calculator/blob/main/results/orbit_visualizer.gif)

## Dependencies

To ensure all required packages are installed, create a virtual environment through Anaconda. Steps to do so are outlined [here](https://varhowto.com/install-miniconda-ubuntu-20-04/).

-   Python - 3.11
-   Orekit - >=12.0
-   Stable-Baselines3 - 2.2.1
-   OpenAI Gynamsium - 0.29.1
-   PyOrb for visualization of training episodes
-   Matplotlib for training plots (states, actions, reward)

## Creating Virtual Environment

Modify the directory where the environment will be located in `environment.yml`: <p>
&ensp; prefix: /home/<b>victoriapham1</b>/miniconda3/envs/otc

```
conda env create -f environment.yml
```

## Activate Environment

```
conda activate otc
```

## Training

```
python3 training.py [DDPG/TD3]
```

To train with CUDA device, modify the device parameter in the training script:

```
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, device="cuda", tau=0.01)
```

## Results

-   Successful termination states during training are found in the <b>results</b> directory
-   A zip of the trained model will appear in the <b>models</b> directory
-   To visualize a successful termination episode: `python3 visual_orbit.py`

## References

A Reinforcement Learning Approach to Spacecraft Trajectory Optimization, [Daniel S. Kolosa](https://github.com/dkolosa/Satmind).
