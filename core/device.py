import torch 

def get_device(device_name = "cuda"):
    """
    Get the PyTorch device based on the provided device name.
    Defaults to 'cuda' if available, otherwise 'cpu'.
    """
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")