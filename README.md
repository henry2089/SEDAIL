# SEDAIL:Sample-Efficient Diffusion Adversarial Imitation Learning

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
```
