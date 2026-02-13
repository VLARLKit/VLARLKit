import hydra
from omegaconf import OmegaConf

from vlarlkit.envs.libero.libero_env import LiberoEnv
from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.rollouts.rollout import Rollout

@hydra.main(config_path="../examples/configs", config_name="libero_10_ppo_pi05.yaml")
def main(cfg):
    env = LiberoEnv(cfg.env.train, num_envs=8)
    policy = None
    rollout_result = RolloutResult()

    rollout = Rollout(cfg, env, policy, rollout_result)
    rollout.rollout()

if __name__ == "__main__":
    main()