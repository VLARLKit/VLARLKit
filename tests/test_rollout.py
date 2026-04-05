import hydra
import torch
import numpy as np
import time

from vlarlkit.utils.action_utils import prepare_actions
from vlarlkit.utils.remote_env import RemoteEnv
from vlarlkit.models.openpi import get_model


def rollout_one_epoch(cfg, env, model):
    num_chunk_steps = (
        cfg.env.eval.max_episode_steps
        // cfg.model.num_action_chunks
    )

    obs, _ = env.reset()

    env_step_times, model_inference_times = [], []

    for _ in range(num_chunk_steps):
        start_time = time.time()
        with torch.no_grad():
            actions, info = model.predict_action_batch(obs, mode="eval")
        model_inference_times.append(time.time() - start_time)
        actions = prepare_actions(
            raw_chunk_actions=actions,
            env_type=cfg.env.eval.env_type,
            model_type=cfg.model.model_type, 
            num_action_chunks=cfg.model.num_action_chunks,
            action_dim=cfg.model.action_dim,
            policy=cfg.model.get("policy_setup", None)
        )
        start_time = time.time()
        next_obs, rewards, terminations, truncations, env_info = env.chunk_step(actions)
        env_step_times.append(time.time() - start_time)
        rewards = rewards.sum(-1)
        terminations = terminations.any(-1)
        truncations = truncations.any(-1)

        obs = next_obs.copy()

    env.update_reset_state_ids()

    return env_info["episode"], env_step_times, model_inference_times


@hydra.main(config_path="../examples/configs", config_name="libero_spatial_ppo_pi05.yaml")
def main(cfg):
    model = get_model(cfg.model)
    model.to("cuda:0")
    # print(model)

    host = cfg.env.get("env_client_host", "localhost")
    base_port = int(cfg.env.get("env_client_base_port", 5550))
    env = RemoteEnv(host=host, port=base_port, env_mode="eval")

    rollout_info = rollout_one_epoch(cfg, env, model)
    print(rollout_info)
    print(f"Env step time: {np.mean(rollout_info[1])}")
    print(f"Model inference time: {np.mean(rollout_info[2])}")

if __name__ == "__main__":
    main()