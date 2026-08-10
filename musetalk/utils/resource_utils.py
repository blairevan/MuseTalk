import gc


def release_torch_resources(*resources):
    """Move Torch resources to CPU and clear cached CUDA memory best-effort."""
    import torch

    errors = []
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.to("cpu")
        except Exception as error:
            errors.append(str(error))

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return errors
