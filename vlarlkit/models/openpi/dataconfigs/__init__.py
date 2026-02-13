from vlarlkit.models.openpi.dataconfigs.libero_dataconfig import LeRobotLiberoDataConfig

DATA_CONFIGS = {
    "libero": LeRobotLiberoDataConfig,
}

def get_data_config(name: str):
    if name not in DATA_CONFIGS:
        raise ValueError(f"Data config {name} not found")
    return DATA_CONFIGS[name]