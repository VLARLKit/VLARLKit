import hydra
from omegaconf import OmegaConf

from vlarlkit.models.openpi import get_model

@hydra.main(config_path="../examples/configs", config_name="libero_10_ppo_pi05.yaml")
def main(cfg):
    cfg.model.model_path = "/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT"
    cfg.model.data.assets_dir = "/network/scratch/s/sunyi/rlinf_sft_models/RLinf-Pi05-LIBERO-SFT/physical-intelligence/libero"
    model = get_model(cfg.model)

if __name__ == "__main__":
    main()