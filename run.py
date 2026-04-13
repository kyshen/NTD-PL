import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from omegaconf import DictConfig
from conf import register_configs
register_configs()

@hydra.main(config_name="config", version_base=None)
def main(cfg: DictConfig):    
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    dataset = hydra.utils.instantiate(cfg.data)
    method = hydra.utils.instantiate(cfg.method)
    task = hydra.utils.instantiate(cfg.task)
    filter = hydra.utils.instantiate(cfg.filter)

    filter(dataset)
    task.setup(dataset, method)
    result = task.run()
    result.save(out_dir)

if __name__ == "__main__":
    main()
