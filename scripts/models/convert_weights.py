import torch
from torch.nn import Module
from torchsparse.nn import Conv3d
from spira.nn import SparseConv
import copy

def torchsparse_reorder(kx, ky, kz):
    stride = ky * kz 
    reorder = []

    for base in range(stride):  
        for step in range(kx):      
            idx = base + step * stride
            reorder.append(idx)
    
    return reorder


def spira_os_reorder(kx, ky, kz):

    stride = kz
    reorder = []

    for base in range(stride):
        for step in range(kx*ky):
            idx = base + step * stride
            reorder.append(idx)
    return reorder

def spira_hs_reorder(kx, ky, kz, threshold):
    kx_min = -(kx // 2) if (kx % 2 == 1) else -(kx // 2) + 1
    ky_min = -(ky // 2) if (ky % 2 == 1) else -(ky // 2) + 1
    kz_min = -(kz // 2) if (kz % 2 == 1) else -(kz // 2) + 1

    kx_max = kx_min + kx - 1
    ky_max = ky_min + ky - 1
    kz_max = kz_min + kz - 1

    front = []
    back = []

    original_idx = 0
    for x in range(kx_min, kx_max + 1):
        for y in range(ky_min, ky_max + 1):
            for z in range(kz_min, kz_max + 1):
                l1 = abs(x) + abs(y) + abs(z)

                if l1 < threshold:
                    front.append(original_idx)
                else:
                    back.append(original_idx)

                original_idx += 1

    return front + back

def layout_torchsparse_weights(model: Module) -> Module:

    """
    Reorder kernel weights of all Spira's Sparse Convolution layers
    so that it matches the layout of Minuet / Spira weight-stat
    """

    new_model = copy.deepcopy(model)

    for name, module in new_model.named_modules():
        if not isinstance(module, Conv3d):
            continue

        kx, ky, kz = module.kernel_size
        kv = kx * ky * kz

        perm = torchsparse_reorder(kx, ky, kz)

        # Sanity
        assert len(perm) == kv

        with torch.no_grad():
            module.kernel.copy_(module.kernel[perm])

        print(f"[✓] TorchSparse layout weights: {name} ({kx}x{ky}x{kz})")

    return new_model


def layout_spira_weights(model: Module) -> Module:
    """
    Reorder kernel weights of all Spira's Sparse Convolution layers
    so that it matches the layout of Minuet / Spira weight-stat
    """
    
    for name, module in model.named_modules():
        if not isinstance(module, SparseConv):
            continue

        kx, ky, kz = module.kernel_size

        if(module._tunable_config['map_dataflow'] == 0 or module._tunable_config['force_os']):
            print(f"[✓] Spira layout weights: {name} (Output Stationary)")
            perm = spira_os_reorder(kx, ky, kz)
            with torch.no_grad():
                module.kernel.copy_(module.kernel[perm])

        elif(module._tunable_config['map_dataflow'] > 2):
            threshold = (module._tunable_config['map_dataflow'] - 1)//2
            print(f"[✓] Spira layout weights: {name} (Hybrid)")
            perm = spira_hs_reorder(kx, ky, kz, threshold)
            with torch.no_grad():
                module.kernel.copy_(module.kernel[perm])

        else: 
            print(f"[✓] Spira layout weights: {name} (Weight Stationary)")  #This is the layout every other schema gets to - nothing to be done
