import hydra
from omegaconf import OmegaConf

from vlarlkit.envs.libero.libero_env import LiberoEnv
from vlarlkit.data.io_struct import RolloutResult
from vlarlkit.rollouts.rollout import Rollout
from vlarlkit.models.openpi import get_model


@hydra.main(config_path="../examples/configs", config_name="libero_10_ppo_pi05.yaml")
def main(cfg):
    cfg.model.model_path = "/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT"
    cfg.model.data.assets_dir = "/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero"
    model = get_model(cfg.model)
    model.to("cuda:0")
    # print(model)

    env = LiberoEnv(cfg.env.eval, num_envs=10)
    rollout_result = RolloutResult()

    rollout = Rollout(cfg, env, model, rollout_result)
    rollout_info = rollout.rollout_one_epoch()
    print(rollout_info)

if __name__ == "__main__":
    main()