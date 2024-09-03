import torch.nn as nn

def summarize_model(model):
    print("Model Summary")
    print("="*50)

    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
            num_params = sum(p.numel() for p in module.parameters())
            num_trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            num_non_trainable_params = num_params - num_trainable_params
            
            total_params += num_params
            trainable_params += num_trainable_params
            non_trainable_params += num_non_trainable_params

            print(f"Layer: {name}")
            print(f"  Type: {module.__class__.__name__}")
            print(f"  Parameters: {num_params}")
            print(f"  Trainable Parameters: {num_trainable_params}")
            print(f"  Non-Trainable Parameters: {num_non_trainable_params}")
            print("-"*50)

    print("Total Parameters:", total_params)
    print("Total Trainable Parameters:", trainable_params)
    print("Total Non-Trainable Parameters:", non_trainable_params)
    print("="*50)
