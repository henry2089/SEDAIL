import yaml
import argparse
import numpy as np
import os, sys, inspect
import pickle
import copy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
print(sys.path)

import gym
from rlkit.envs import get_env, get_envs
import random
# noinspection PyPackageRequirements
import rlkit.torch.utils.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger, set_seed
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.torch.common.networks import FlattenMlp
from rlkit.torch.common.policies import ReparamTanhMultivariateGaussianPolicy
from rlkit.torch.algorithms.sac.sac_alpha import (
    SoftActorCritic,
)  # SAC Auto alpha version
from ddpm_disc.ddpm_disc_model import DDPM_Disc
from rlkit.torch.algorithms.adv_irl.UTD_adv_irl import UTDAdvIRL
from rlkit.envs.wrappers import ProxyEnv, ScaledEnv, MinmaxEnv, EPS
from rlkit.weight.mlp import ConcatMlp
from ddpm_disc.Classifier.AdjustedValueClassifier import AdjustedValueClassifier
from ddpm_disc.Classifier.Classifier import TimedClassifier_new
from ddpm_disc.Classifier.Classifier import TimedClassifier
from rlkit.torch.algorithms.REDQ.REDQ import REDQSoftActorCritic
from rlkit.torch.algorithms.sac.sac_delay import SoftActorCritic_dalay
from ddpm_disc.diffusion.diffchain import Diffchain

import numpy as np


def experiment(variant):
    with open("demos_listing.yaml", "r") as f:
        listings = yaml.safe_load(f.read())

    demos_path = listings[variant["expert_name"]]["file_paths"][variant["expert_idx"]]
    """
    Buffer input format
    """

    """
    PKL input format
    """
    print("demos_path", demos_path)

    with open(demos_path, "rb") as f:
        traj_list = pickle.load(f)
    traj_list = random.sample(traj_list, variant["traj_num"])

    obs = np.vstack([traj_list[i]["observations"] for i in range(len(traj_list))])
    obs_mean, obs_std = np.mean(obs, axis=0), np.std(obs, axis=0)
    acts_mean, acts_std = None, None
    obs_min, obs_max = np.min(obs, axis=0), np.max(obs, axis=0)

    env_specs = variant["env_specs"]
    env = get_env(env_specs)
    env.seed(env_specs["eval_env_seed"])

    print("\n\nEnv: {}".format(env_specs["env_name"]))
    print("kwargs: {}".format(env_specs["env_kwargs"]))
    print("Obs Space: {}".format(env.observation_space))
    print("Act Space: {}\n\n".format(env.action_space))

    expert_replay_buffer = EnvReplayBuffer(
        variant["adv_irl_params"]["replay_buffer_size"],
        env,
        random_seed=np.random.randint(10000),
    )

    tmp_env_wrapper = env_wrapper = ProxyEnv  # Identical wrapper
    kwargs = {}
    wrapper_kwargs = {}

    if variant["scale_env_with_demo_stats"]:
        print("\nWARNING: Using scale env wrapper")
        tmp_env_wrapper = env_wrapper = ScaledEnv
        wrapper_kwargs = dict(
            obs_mean=obs_mean,
            obs_std=obs_std,
            acts_mean=acts_mean,
            acts_std=acts_std,
        )
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_mean) / (obs_std + EPS)
            traj_list[i]["next_observations"] = (traj_list[i]["next_observations"] - obs_mean) / (obs_std + EPS)

    elif variant["minmax_env_with_demo_stats"]:
        print("\nWARNING: Using min max env wrapper")
        tmp_env_wrapper = env_wrapper = MinmaxEnv
        wrapper_kwargs = dict(obs_min=obs_min, obs_max=obs_max)
        for i in range(len(traj_list)):
            traj_list[i]["observations"] = (traj_list[i]["observations"] - obs_min) / (obs_max - obs_min + EPS)
            traj_list[i]["next_observations"] = (traj_list[i]["next_observations"] - obs_min) / (
                    obs_max - obs_min + EPS)

    obs_space = env.observation_space
    act_space = env.action_space
    max_action = float(env.action_space.high[0])
    assert not isinstance(obs_space, gym.spaces.Dict)
    assert len(obs_space.shape) == 1
    assert len(act_space.shape) == 1

    env = env_wrapper(env, **wrapper_kwargs)
    training_env = get_envs(
        env_specs, env_wrapper, wrapper_kwargs=wrapper_kwargs, **kwargs
    )
    training_env.seed(env_specs["training_env_seed"])

    obs_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # build the policy models
    net_size = variant["policy_net_size"]
    num_hidden = variant["policy_num_hidden_layers"]

    policy = ReparamTanhMultivariateGaussianPolicy(
        hidden_sizes=num_hidden * [net_size],
        obs_dim=obs_dim,
        action_dim=action_dim,
    )
    if variant["adv_irl_params"]["wrap_absorbing"]:
        obs_dim += 1
    input_dim = obs_dim + action_dim
    if variant["adv_irl_params"]["state_only"]:
        input_dim = obs_dim + obs_dim

    # build the classifier and discriminator model
    """ Initialize classifier """
    if variant["clas_type"] == 'class':
        classifier = TimedClassifier(
            n_timesteps=variant["disc_ddpm_n_timesteps"],
            input_dim=obs_dim + action_dim,
            timestep_embed_dim=32,
            hidden_dim=256,
        )
    else:
        classifier = AdjustedValueClassifier(
            input_size=obs_dim + action_dim,
            output_size=1,
            hidden_dims=[256, 256],
            emb_dim=128,  # Example value; adjust according to model needs
        )
    classifier.to(ptu.device)

    """ Initialize classifier EMA """
    ema_classifier = copy.deepcopy(classifier)
    ema_classifier.eval()  # Set to evaluation mode by default
    for param in ema_classifier.parameters():
        param.requires_grad = False  # Disable gradient computation for EMA classifier

    for i in range(len(traj_list)):
        expert_replay_buffer.add_path(
            traj_list[i], absorbing=variant["adv_irl_params"]["wrap_absorbing"], env=env,
        )

    """ Prepare weight networks """
    weight_net = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[256, 256],
    )
    weight_net.last_fc.bias.data.fill_(1.0)
    weight_net.to(ptu.device)

    # set up the algorithm
    num_update = variant["adv_irl_params"]["num_update_loops_per_train_call"] * variant["UTD"]
    num_update_disc = num_update / variant["disc_divisor"]
    ada_total = num_update_disc * variant["adv_irl_params"]["num_epochs"] * 10.0 * variant["adv_irl_params"]["num_disc_updates_per_loop_iter"]
    ada_divisor = variant["ada_divisor"]
    ada_interval = variant["ada_interval"]
    ada_target = variant["ada_target"]
    ada_unit = ada_total / ada_divisor
    ada_const = ada_interval / ada_unit
    utd = variant["UTD"]
    if utd > 3:
        lambda_factor = variant["lambda_factor"] / utd * variant["x_dim_mult"]
    else:
        lambda_factor = variant["lambda_factor"]

    disc_model = DDPM_Disc(
        input_dim, action_dim, max_action,
        n_timesteps=variant["disc_ddpm_n_timesteps"],
        disc_hid_dim=variant["disc_hid_dim"],
        device=ptu.device,
        lr=variant["adv_irl_params"]["disc_lr"],
        disc_momentum=variant["adv_irl_params"]["disc_momentum"],
        clamp_magnitude=variant["disc_clamp_magnitude"],
        classifier=classifier,
        ema_classifier=ema_classifier,
        lambda_factor=lambda_factor,
        t_stopgrad=variant["t_stopgrad"],
        extend_type=variant["extend_type"],
        cond_dim=variant["cond_dim"],
        condition_mlp=variant["condition_mlp"],
        env_name=variant["env_specs"]["env_name"],
        traj_num=variant["traj_num"],
        trainer_version=variant["trainer_version"],
        beta_schedule=variant["beta_schedule"],
        beta_start=1e-4,
        beta_end=2e-2,
        t_min=variant["t_min"],
        t_max=variant["t_max"],
        is_diff_adapt=variant["is_diff_adapt"],
        t_dim_mult=variant["t_dim_mult"],
        t_dim=variant["t_dim"],
        n_dim=variant["n_dim"],
        x_emb_dim=variant["x_emb_dim"],
        num_mid_layers=variant["num_mid_layers"],
        x_emb_type=variant["x_emb_type"],
        x_dim_mult=variant["x_dim_mult"],
        emb_type=variant["emb_type"],
        drop_rate=variant["drop_rate"],
    )

    if variant["trainer_type"] == 'REDQ':
        trainer = REDQSoftActorCritic(
            policy=policy,
            env=env,
            obs_dim=obs_dim,
            act_dim=action_dim,
            utd_ratio=utd,
            policy_update_delay=utd,
            num_Q=variant["num_Q"],
            num_min=variant["num_min"],
            lr=variant["lr"],
            num_update=num_update,
            **variant["sac_params"]
        )
    elif variant["trainer_type"] == 'sacdelay':
        qf1 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        qf2 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        trainer = SoftActorCritic_dalay(
            num_update=num_update, policy_update_delay=utd, policy=policy, qf1=qf1, qf2=qf2, env=env, **variant["sac_params"]
        )
    else:
        qf1 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        qf2 = FlattenMlp(
            hidden_sizes=num_hidden * [net_size],
            input_size=obs_dim + action_dim,
            output_size=1,
        )
        trainer = SoftActorCritic(
            lr=variant["lr"], policy=policy, qf1=qf1, qf2=qf2, env=env, **variant["sac_params"]
        )

    diffusion_chain = Diffchain(
        beta_schedule=variant["beta_schedule"],
        beta_start=1e-4,
        beta_end=2e-2,
        t_min=variant["t_min"],
        t_max=variant["t_max"],
        ts_dist=variant["ts_dist"],
    )
    algorithm = UTDAdvIRL(
        env=env,
        training_env=training_env,
        exploration_policy=policy,
        discriminator=disc_model,
        policy_trainer=trainer,
        expert_replay_buffer=expert_replay_buffer,
        onpolicy_buffer_size=variant["onpolicy_buffer_size"],
        num_weight_per_train_call=variant["num_weight_per_train_call"],
        num_classifier_per_train_call=variant["num_classifier_per_train_call"],
        trainer_type=variant["trainer_type"],
        num_update=num_update,
        num_update_disc=num_update_disc,
        disc_divisor=variant["disc_divisor"],
        ada_const=ada_const,
        ada_interval=ada_interval,
        ada_target=ada_target,
        weight_net=weight_net,
        diffusion_chain=diffusion_chain,
        ischain=variant["ischain"],
        disc_ddpm=variant["env_specs"]["disc_ddpm"],
        is_classifier_reward=variant["is_classifier_reward"],
        eval_calc_batch_size=variant["eval_calc_batch_size"],
        num_eval_iterations=variant["num_eval_iterations"],
        stop_utd_return_num=variant["stop_utd_return_num"],
        env_name=variant["env_specs"]["env_name"],
        traj_num=variant["traj_num"],
        train_type=variant["train_type"],
        is_reward_smooth=variant["is_reward_smooth"],
        reward_smooth_gamma=variant["reward_smooth_gamma"],
        train_divisor=variant["train_divisor"],
        is_clip=variant["is_clip"],
        utd=variant["UTD"],
        update_type=variant["update_type"],
        clas_type=variant["clas_type"],
        clip_num=variant["clip_num"],
        **variant["adv_irl_params"]
    )

    if ptu.gpu_enabled():
        algorithm.to(ptu.device)
    algorithm.train()

    return 1


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    parser.add_argument("-g", "--gpu", default=2, type=int)
    args = parser.parse_args()
    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # Make all seeds the same.
    exp_specs["env_specs"]["eval_env_seed"] = exp_specs["env_specs"][
        "training_env_seed"
    ] = exp_specs["seed"]

    exp_suffix = ""
    if exp_specs["demo_type"] == 'sample':
        exp_suffix = "--lf{}--utd{}--gp-{}--rs-{}--sample-{}--trajnum-{}".format(
            exp_specs["lambda_factor"],
            exp_specs["UTD"],
            exp_specs["adv_irl_params"]["grad_pen_weight"],
            exp_specs["sac_params"]["reward_scale"],
            exp_specs["sample_length"],
            format(exp_specs["traj_num"]),
        )
    else:
        exp_suffix = "--lf{}--utd{}--gp-{}--rs-{}--trajnum-{}".format(
            exp_specs["lambda_factor"],
            exp_specs["UTD"],
            exp_specs["adv_irl_params"]["grad_pen_weight"],
            exp_specs["sac_params"]["reward_scale"],
            format(exp_specs["traj_num"]),
        )

    if not exp_specs["adv_irl_params"]["no_terminal"]:
        exp_suffix = "--terminal" + exp_suffix

    if exp_specs["adv_irl_params"]["wrap_absorbing"]:
        exp_suffix = "--absorbing" + exp_suffix

    if exp_specs["scale_env_with_demo_stats"]:
        exp_suffix = "--scale" + exp_suffix

    if exp_specs["using_gpus"] > 0:
        print("\n\nUSING GPU\n\n")
        ptu.set_gpu_mode(True, args.gpu)

    exp_id = exp_specs["exp_id"]
    exp_prefix = exp_specs["exp_name"]
    seed = exp_specs["seed"]
    set_seed(seed)

    exp_prefix = exp_prefix + exp_suffix
    setup_logger(exp_prefix=exp_prefix, exp_id=exp_id, variant=exp_specs, seed=seed)

    experiment(exp_specs)
