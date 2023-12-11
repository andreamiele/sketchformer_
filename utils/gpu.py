import torch

def setup_gpu(gpu_ids):
    # Check if GPUs are available
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU...")
        return torch.device('cpu')

    # If specific GPU IDs are provided, set the default CUDA device
    if gpu_ids:
        torch.cuda.set_device(gpu_ids[0])

    # Return the main device
    device = torch.device(f'cuda:{gpu_ids[0]}' if gpu_ids else 'cuda')
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    return device


