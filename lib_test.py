import torch, torchvision
print(f'The version of torch: {torch.__version__}')
print(f'The version of torchvision: {torchvision.__version__}')
print(f'CUDA availibility: {torch.cuda.is_available()}')

if torch.cuda.is_available(): 
    print(f'CUDA is using device {torch.cuda.current_device()}')
    print(f'Current device is {torch.cuda.get_device_name(torch.cuda.current_device())}')
