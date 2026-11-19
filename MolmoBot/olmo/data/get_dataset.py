from olmo.data.dataset import Dataset
from olmo.data.synthmanip_dataset import build_synthmanip_dataset


def get_dataset_by_name(dataset_name, split, *, backend="native") -> Dataset:
    if backend == "lerobot_lance":
        from olmo.data.lerobot_lance_dataset import build_dataset
        return build_dataset(dataset_name, split)
    if backend != "native":
        raise ValueError(f"Unknown data backend: {backend}")
    if dataset_name.startswith("synthmanip/"):
        return build_synthmanip_dataset(dataset_name, split)

    raise NotImplementedError(dataset_name, split)