import argparse
from loaders import make_dataset, torchsparse_dataloader, minuet_dataloader, spira_dataloader, spira_64_dataloader
import os
import torch
import itertools
import numpy as np

def main(args):

    dataset = make_dataset(args.scene)

    cache_path = f"./autotune_cache/{args.library}"

    os.makedirs(cache_path, exist_ok=True)
    
    with torch.no_grad():
        if args.library == "torchsparse":
            from models.torchsparse import SparseResUNet42, SparseResNet21D, SparseResNetLarge
            from torchsparse.nn.functional.conv.conv_config import get_default_conv_config, set_global_conv_config
            import torchsparse
            
            conv_config = get_default_conv_config()
            conv_config.downsample_mode = "minkowski"        # We follow the same downsampling definition as MinkowskiEngine               
            conv_config.kmap_mode = "hashmap"       # hashmap-on-the-fly mode cannot support minkowski downsample mode 
        
            set_global_conv_config(conv_config)

            if (args.model == "ResNet"):
                model = SparseResNet21D(in_channels=args.input_channels)

            elif (args.model == "ResNetLarge"):
                model = SparseResNetLarge(in_channels=args.input_channels)

            elif (args.model == "UNet"):
                model = SparseResUNet42(in_channels=args.input_channels)

            else:
                raise NotImplementedError(args.model)

            model = model.cuda().half()
            model.eval()
            dataloader = torchsparse_dataloader

            torchsparse.tune(
                model=model,
                data_loader=[next(dataloader(dataset, args.input_channels, sort_coordinates=True))],
                n_samples=1,
                enable_fp16=True,
                force_retune=True,
                save_dir=cache_path
            )

        elif args.library == "minuet":
            from models.minuet import SparseResUNet42, SparseResNet21D, SparseResNetLarge
            import minuet
            
            if (args.model == "ResNet"):
                model = SparseResNet21D(in_channels=args.input_channels)
            
            elif (args.model == "ResNetLarge"):
                model = SparseResNetLarge(in_channels=args.input_channels)

            elif (args.model == "UNet"):
                model = SparseResUNet42(in_channels=args.input_channels)
            else:
                raise NotImplementedError(args.model)
            
            model = model.cuda().half()
            model.eval()
            dataloader = minuet_dataloader

            model_cache = minuet.nn.KernelMapCache(ndim=3, dtype=torch.int32, device='cuda')
            minuet.set_kernel_map_cache(module=model, cache=model_cache)

            cache_path_file = os.path.join(
                cache_path,
                f"{args.scene}_{args.model}.json"
            )
            minuet.autotune(
                model,
                data=[next(dataloader(dataset, args.input_channels, sort_coordinates=True))],
                model_cache=model_cache,
                cache_path=cache_path_file
            )
        
        elif args.library == "spira":
            from models.spira import SparseResUNet42, SparseResNet21D, SparseResNetLarge
            import spira

            if (args.model == "ResNet"):
                model = SparseResNet21D(in_channels=args.input_channels)
                target_strides = [(2, 2, 2), (4, 4, 4), (8, 8, 8), (8, 8, 16)]
                layer_strides = [(1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1), (1, 1, 2)]
                kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (1, 1, 3)]
                source_strides = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2), (4, 4, 4), (4, 4, 4), (8, 8, 8), (8, 8, 8), (8, 8, 16)]

            elif (args.model == "ResNetLarge"):
                model = SparseResNetLarge(in_channels=args.input_channels)
                target_strides = [(2, 2, 2), (4, 4, 4), (8, 8, 8)]
                layer_strides = [(1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1), (2, 2, 2), (1, 1, 1)]
                kernel_sizes = [(5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5), (5, 5, 5)]
                source_strides = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2), (4, 4, 4), (4, 4, 4), (8, 8, 8), (8, 8, 8)] 

            elif (args.model == "UNet"):
                model = SparseResUNet42(in_channels=args.input_channels)
                target_strides = [(2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16)]
                layer_strides = [(1,1,1), (2,2,2), (1,1,1), (2,2,2), (1,1,1), (2,2,2), (1,1,1), (2,2,2), (1,1,1)]
                kernel_sizes = [(3, 3, 3), (2, 2, 2), (3, 3, 3), (2, 2, 2), (3, 3, 3), (2, 2, 2), (3, 3, 3), (2, 2, 2), (3, 3, 3)]
                source_strides = [(1, 1, 1), (1, 1, 1), (2, 2, 2), (2, 2, 2), (4, 4, 4), (4, 4, 4), (8, 8, 8), (8, 8, 8), (16, 16, 16), (16, 16, 16)] 
            else:
                raise NotImplementedError(args.model)

            model = model.cuda().half()
            model.eval()
            dataloader = spira_dataloader

            model_cache = spira.nn.KernelMapCache()
            spira.set_kernel_map_cache(module=model, cache=model_cache)

            tensor = next(dataloader(dataset, args.input_channels, sort_coordinates=True))

            model, chosen_dataflows = spira.tune(
                model=model,
                data_loader=[tensor, tensor, tensor, tensor, tensor],
                kernel_map_cache=model_cache,
                test_force_os=True,             
                n_samples=5,                   
                force_retune=True,
                save_dir = cache_path,              
                verbose=False,
                return_dataflows=True
            )

            group_lookup = {}

            for group_id, map_df in chosen_dataflows.items():
                effective_stride, kernel_size, stride = group_id
                group_lookup[(effective_stride, kernel_size, stride)] = map_df
            dataflows = []
            for i in range(len(kernel_sizes)):
                key = (source_strides[i], kernel_sizes[i], layer_strides[i])
                if key not in group_lookup:
                    raise KeyError(
                        f"No tuned group found for kernel_size={kernel_sizes[i]}, "
                        f"input_stride={source_strides[i]}"
                    )
                dataflows.append(group_lookup[key])

        else:
            raise NotImplementedError(args.library)

        timings = []

        for r in range(7):
            tensor = next(dataloader(dataset, args.input_channels, sort_coordinates=False))

            if args.library == "minuet":
                model_cache.reset()
                minuet.nn.functional.cuda_free_buffers()

                from minuet import SparseTensor
                
                torch.cuda.synchronize()
                event1 = torch.cuda.Event(enable_timing=True)
                event2 = torch.cuda.Event(enable_timing=True)

                event1.record()
                index = minuet.nn.functional.build_sorted_index(tensor.C)
                tensor = SparseTensor(coordinates=tensor.C[index],
                                    features=tensor.F[index])
                _ = model(tensor)
                event2.record()
                event2.synchronize()


            elif args.library == "spira":
                model_cache.reset()
                torch.cuda.synchronize()
                event1 = torch.cuda.Event(enable_timing=True)
                event2 = torch.cuda.Event(enable_timing=True)
                event1.record()
                tensor.order()
                model_cache.add_all_kernel_maps(
                    tensor._coordinates,
                    kernel_sizes,
                    source_strides,
                    target_strides,
                    (1, 1, 1),
                    dataflows)
                _ = model(tensor)
                event2.record()
                event2.synchronize()

            elif args.library == "torchsparse":
                torch.cuda.synchronize()
                event1 = torch.cuda.Event(enable_timing=True)
                event2 = torch.cuda.Event(enable_timing=True)

                event1.record()
                _ = model(tensor)
                event2.record()
                event2.synchronize()

            else:
                raise NotImplementedError(args.library)

            if r >= 3:
                elapsed = event1.elapsed_time(event2)
                timings.append((r, elapsed))
                

        elapsed_times = [t for r, t in timings]

        print(f"scene: {args.scene}")
        print(f"model: {args.model}")
        print()
        print(f"library: {args.library}")
        print("round_times_ms: [", end="")
        print(", ".join(f"{x:.4f}" for x in elapsed_times), end="")
        print("]")
        print(f"overall_avg_ms: {np.mean(elapsed_times):.4f}")
        print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, choices=["minuet", "torchsparse", "spira"],
                        required=True, help="Which library to benchmark")
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="dataset path")
    parser.add_argument("--model",
                        type=str,
                        choices=["ResNet", "ResNetLarge", "UNet"],
                        required=True,
                        help="which model to benchmark")
    parser.add_argument("--input_channels",
                        type=int,
                        default=4,
                        help="number of input channels in model")
    main(parser.parse_args())