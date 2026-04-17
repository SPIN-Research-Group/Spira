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
            from torchsparse.nn import Conv3d
            from torchsparse.nn.functional.conv.conv_config import get_default_conv_config, set_global_conv_config
            import torchsparse
            
            conv_config = get_default_conv_config()
            conv_config.downsample_mode = "minkowski"    # We follow the same downsampling definition as MinkowskiEngine               
            conv_config.kmap_mode = "hashmap"            # hashmap-on-the-fly mode cannot support minkowski downsample mode 
        
            set_global_conv_config(conv_config)

            model = Conv3d(in_channels = args.cin, out_channels = args.cout, kernel_size = args.k)

            model = model.cuda().half()
            model.eval()
            dataloader = torchsparse_dataloader

            torchsparse.tune(
                model=model,
                data_loader=dataloader(dataset, args.cin),
                n_samples=args.autotuning_samples,
                enable_fp16=True,
                force_retune=True,
                save_dir=cache_path
            )

        elif args.library == "minuet":
            from minuet.nn import SparseConv
            import minuet
            
            model = SparseConv(ndim=3, in_channels=args.cin, out_channels=args.cout, kernel_size=args.k)
            
            model = model.cuda().half()
            model.eval()
            dataloader = minuet_dataloader

            model_cache = minuet.nn.KernelMapCache(ndim=3, dtype=torch.int32, device='cuda')
            minuet.set_kernel_map_cache(module=model, cache=model_cache)

            data = dataloader(dataset, args.cin)
            data_to_tune = itertools.islice(data, args.autotuning_samples)

            cache_path_file = os.path.join(
                cache_path,
                f"{args.dataset}_{args.cin}_{args.cout}_{args.k}.json"
            )
            minuet.autotune(
                model,
                data=data_to_tune,
                model_cache=model_cache,
                cache_path=cache_path_file
            )
        
        elif args.library == "spira":
            from spira.nn import SparseConv
            import spira

            model = SparseConv(in_channels=args.cin, out_channels=args.cout, kernel_size=args.k)

            model = model.cuda().half()
            model.eval()
            dataloader = spira_dataloader

            model_cache = spira.nn.KernelMapCache()
            spira.set_kernel_map_cache(module=model, cache=model_cache)

            model = spira.tune(
                model=model,
                data_loader=dataloader(dataset, args.cin),
                kernel_map_cache=model_cache,
                test_force_os=True,             
                n_samples=args.autotuning_samples,                   
                force_retune=True,
                save_dir = cache_path,              
                verbose=False,
                return_dataflows=False
            )

        else:
            raise NotImplementedError(args.library)

        timings = []

        # Warm-up
        for i, inputs in enumerate(dataloader(dataset, args.cin)):
            if args.library == "minuet":
                model_cache = minuet.nn.KernelMapCache(ndim=3,
                                               dtype=torch.int32,
                                               device="cuda")
                minuet.set_kernel_map_cache(module=model, cache=model_cache)
            elif args.library == "spira":
                model_cache.reset()
            if i >= 3:
                break
            _ = model(inputs)

        for r in range(4):
            for s, inputs in enumerate(dataloader(dataset, args.cin)):
                
                if args.library == "minuet":
                    model_cache = minuet.nn.KernelMapCache(ndim=3,
                                               dtype=torch.int32,
                                               device="cuda")
                    minuet.set_kernel_map_cache(module=model, cache=model_cache)

                    torch.cuda.synchronize()
                    event1 = torch.cuda.Event(enable_timing=True)
                    event2 = torch.cuda.Event(enable_timing=True)

                    event1.record()
                    _ = model(inputs)
                    event2.record()
                    event2.synchronize()


                elif args.library == "spira":
                    model_cache.reset()
                    torch.cuda.synchronize()
                    event1 = torch.cuda.Event(enable_timing=True)
                    event2 = torch.cuda.Event(enable_timing=True)
                    event1.record()
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
    print(f"input channels: {args.cin}")
    print(f"output channels: {args.cout}")
    print(f"kernel size: {args.k}")
    print()

    print(f"library: {args.library}")
    print("per_sample_time: [", end="")
    print(", ".join(f"{x:.4f}" for x in per_sample_avg), end="")
    print("]")
    print(f"overall_avg: {overall_avg:.4f}")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=str, choices=["minuet", "torchsparse", "spira"],
                        required=True, help="Which library to benchmark")
    
    parser.add_argument("--dataset",
                        type=str,
                        required=True,
                        help="the dataset for benchmarking")
    
    parser.add_argument("--dataset_path",
                        type=str,
                        required=True,
                        help="dataset path")
    
    parser.add_argument("--cin", required=True, type=int, help="input channels")

    parser.add_argument("--cout", required=True, type=int, help="output channels")

    parser.add_argument("--k", required=True, type=int, help="kernel size")

    parser.add_argument("--autotuning_samples",
                        default=5,
                        type=int,
                        help="samples for autotuning process")
    main(parser.parse_args())
