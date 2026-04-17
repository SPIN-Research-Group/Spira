import argparse
from loaders import make_dataset, torchsparse_dataloader, minuet_dataloader, spira_dataloader, spira_64_dataloader
import os
import torch
import itertools
import numpy as np

def main(args):

    dataset = make_dataset(args.dataset_path)

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
                data_loader=dataloader(dataset, args.input_channels),
                n_samples=args.autotuning_samples,
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

            data = dataloader(dataset, args.input_channels)
            data_to_tune = itertools.islice(data, args.autotuning_samples)

            cache_path_file = os.path.join(
                cache_path,
                f"{args.dataset}_{args.model}.json"
            )
            minuet.autotune(
                model,
                data=data_to_tune,
                model_cache=model_cache,
                cache_path=cache_path_file
            )
        
        elif args.library == "spira" or args.library == "spira_64":
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
            if args.library == "spira":
                dataloader = spira_dataloader
            else:
                dataloader = spira_64_dataloader

            model_cache = spira.nn.KernelMapCache()
            spira.set_kernel_map_cache(module=model, cache=model_cache)

            model, chosen_dataflows = spira.tune(
                model=model,
                data_loader=dataloader(dataset, args.input_channels),
                kernel_map_cache=model_cache,
                test_force_os=True,             
                n_samples=args.autotuning_samples,                   
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

        # Warm-up
        for i, inputs in enumerate(dataloader(dataset, args.input_channels)):
            if args.library == "minuet":
                model_cache.reset()
                minuet.nn.functional.cuda_free_buffers()
            elif args.library == "spira" or args.library == "spira_64":
                model_cache.reset()
            if i >= 3:
                break
            _ = model(inputs)

        for r in range(4):
            for s, inputs in enumerate(dataloader(dataset, args.input_channels, sort_coordinates=False)):
                
                if args.library == "minuet":
                    model_cache.reset()
                    minuet.nn.functional.cuda_free_buffers()

                    from minuet import SparseTensor
                    
                    torch.cuda.synchronize()
                    event1 = torch.cuda.Event(enable_timing=True)
                    event2 = torch.cuda.Event(enable_timing=True)

                    event1.record()
                    index = minuet.nn.functional.build_sorted_index(inputs.C)
                    inputs = SparseTensor(coordinates=inputs.C[index],
                                        features=inputs.F[index])
                    _ = model(inputs)
                    event2.record()
                    event2.synchronize()


                elif args.library == "spira" or args.library == "spira_64":
                    model_cache.reset()
                    torch.cuda.synchronize()
                    event1 = torch.cuda.Event(enable_timing=True)
                    event2 = torch.cuda.Event(enable_timing=True)
                    event1.record()
                    inputs.order()
                    model_cache.add_all_kernel_maps(
                        inputs._coordinates,
                        kernel_sizes,
                        source_strides,
                        target_strides,
                        (1, 1, 1),
                        dataflows)
                    _ = model(inputs)
                    event2.record()
                    event2.synchronize()

                elif args.library == "torchsparse":
                    torch.cuda.synchronize()
                    event1 = torch.cuda.Event(enable_timing=True)
                    event2 = torch.cuda.Event(enable_timing=True)

                    event1.record()
                    _ = model(inputs)
                    event2.record()
                    event2.synchronize()

                else:
                    raise NotImplementedError(args.library)

                elapsed = event1.elapsed_time(event2)
                timings.append((r, s, elapsed))
                
    
    timings = np.asarray(timings)  

    rounds = timings[:, 0].astype(int)
    samples = timings[:, 1].astype(int)

    num_samples = samples.max() + 1
    num_rounds = rounds.max() + 1

    timings_matrix = np.full((num_rounds, num_samples), np.nan)
    for r, s, t in timings:
        timings_matrix[int(r), int(s)] = t

    per_sample_avg = np.nanmean(timings_matrix, axis=0)

    overall_avg = np.nanmean(per_sample_avg)

    print(f"dataset: {args.dataset}")
    print(f"num_samples: {num_samples}")
    print(f"model: {args.model}")
    print()

    print(f"library: {args.library}")
    print("per_sample_time: [", end="")
    print(", ".join(f"{x:.4f}" for x in per_sample_avg), end="")
    print("]")
    print(f"overall_avg: {overall_avg:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, choices=["minuet", "torchsparse", "spira", "spira_64"],
                        required=True, help="Which library to benchmark")
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="the dataset for benchmarking")
    parser.add_argument("--dataset_path",
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
    parser.add_argument("--autotuning_samples",
                        default=5,
                        type=int,
                        help="samples for autotuning process")
    main(parser.parse_args())