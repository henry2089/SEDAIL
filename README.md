# SEDAIL:Sample-Efficient Diffusion Adversarial Imitation Learning
![framework](https://github.com/user-attachments/assets/c209ba66-cd89-451d-829b-63644e2e503a)
_(a) The discriminator \(D_{(\phi,\psi)}\) is a classifier guided conditional diffusion model. Each sample first passes through the diffusion chain; the diffusion model \(\phi\) then computes a loss that trains the discriminator to tell apart expert data \(x_{0}\sim\mathcal{D}\) and agent data \(x_{0}\sim\mathcal{R}_{\text{off}}\). (b) The policy \(\pi_{\theta}\) is updated by maximizing the diffusion reward \(R_{(\phi,\psi)}\) produced by \(D_{(\phi,\psi)}\)._

# Requirements

SEDAIL is evaluated  on MuJoCo continuous control tasks in OpenAI gym. It can be set up by following the official installation instructions at https://github.com/openai/mujoco-py.

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

# Dataset

All dataset archive files are located in ./demos/data. They are compressed, please extract them into the ./demos/data directory.

# Running  

Example scripts:

```
python ./run_experiment.py -e "./exp_specs/test_ant.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_half.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_hopper.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_walker.yaml" -g 0
```
