# SEDAIL:Sample-Efficient Diffusion Adversarial Imitation Learning
![framework](https://github.com/user-attachments/assets/c209ba66-cd89-451d-829b-63644e2e503a)
_(a) The discriminator is a classifier guided conditional diffusion model. Each sample first passes through the diffusion chain; the diffusion model then computes a loss that trains the discriminator to tell apart expert data and agent data. (b) The policy is updated by maximizing the diffusion reward produced by discriminator._

# Features
SEDAIL (1) increases the update‑to‑data ratio to improve sample efficiency, (2) replaces the discriminator with a classifier‑guided conditional diffusion loss to mitigate off‑policy distribution shift, (3) adaptively schedules diffusion steps (noise strength) to curb discriminator overfitting. On MuJoCo tasks with scarce demonstrations and high UTD, SEDAIL surpasses existing off‑policy and diffusion‑augmented baselines.

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

Following are some examples of training the SEDAIL algorithm.

```
python ./run_experiment.py -e "./exp_specs/test_ant.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_half.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_hopper.yaml" -g 0
python ./run_experiment.py -e "./exp_specs/test_walker.yaml" -g 0
```

In the above examples, the 'g' option specifies the GPU number to be used, the 'e' option specifies the environment.
