# Spira: Exploiting Voxel Data Structural Properties  for Efficient Sparse Convolution in Point Cloud Networks

[<i>Spira</i>](https://arxiv.org/pdf/2511.20834) is the first voxel-property-aware Sparse Convolution engine to efficiently execute Point Cloud Networks on modern GPUs. 

Sparse Convolution (SpC) powers 3D point cloud networks widely used in autonomous driving and AR/VR. SpC builds a kernel map that stores mappings between input voxel coordinates, output coordinates, and weight offsets, then uses this map to compute feature vectors for output coordinates. Our work identifies three key properties of voxel coordinates: they are integer-valued, bounded within a limited spatial range, and geometrically continuous, i.e., neighboring voxels on the same object surface are highly likely to exist at small spatial offsets from each other. Prior SpC engines do not fully exploit these properties and suffer from high pre-processing and post-processing overheads during kernel map construction. 

Spira the first voxel-property-aware SpC engine for GPUs. Spira proposes: (i) a high-performance one-shot search algorithm that builds the kernel map with no preprocessing and high memory locality, (ii) an effective packed-native processing
scheme that accesses packed voxel coordinates at low cost, (iii) a flexible dual-dataflow execution mechanism that efficiently computes output feature vectors by adapting to layer characteristics, and (iv) a network-wide parallelization strategy that builds kernel maps for all SpC layers concurrently at network start. Spira provides significant performance benefits across various point cloud networks, real datasets, and GPU architectures.

## Cite Spira

Please use the following citations to cite Spira, if you find this repository useful:

Bibtex entries for citation:
```
@article{Adamopoulos2026Spira,
  author={Adamopoulos, Dionysios and Poulopoulou, Anastasia and Goumas, Georgios and Giannoula, Christina},
  title={Spira: Exploiting Voxel Data Structural Properties  for Efficient Sparse Convolution in Point Cloud Networks}, 
  journal={Proceedings of Machine Learning and Systems},
  volume={8},
  year={2026}
}
```

## Repository Overview
Todo

## Quick Start

### Hardware & Software Requirements 

The artifact should run on hardware platforms with: 

x86-64 CPU: 
* Memory: ≥ 64GB
* Storage: ≥ 256GB
  
NVIDIA GPU: 
* Memory: ≥ 16GB
* CUDA Compute Capability (SM): 7.5+

The artifact should be executed on a Linux-based operating system with an up-to-date NVIDIA driver that supports CUDA 12.4 or newer.

### Step 1. Building the Artifact 
We recommend using Docker Engine for building the artifact to fully control all software dependencies. Please follow the instructions to [Install Docker Engine](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) first. Note that if the current user is not in the docker user group, all following docker-related commands require root privilege (i.e. with sudo) to run. If you want to verify that the NVIDIA Container Toolkit is correctly installed, you can run the following command:

```shell
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 nvidia-smi
```

This will download the base docker image and if everything is set up correctly, you’ll see the output with GPU information.

To build the docker image for the artifact, you can use the following snippet:
```shell
nvidia-smi #Run this command to see the GPU IDs
export GPU_ID=0   #Assuming you want to use GPU 0 for evaluation
export CUDA_ARCHS="$(nvidia-smi -i $GPU_ID --query-gpu=compute_cap --format=csv | tail -n 1)" #This captures the compute capability of the GPU

docker build --build-arg CUDA_ARCHS=$CUDA_ARCHS -t spira .
```
After successfully built the docker image, execute the following command to 
launch the container:

```shell
docker run -it --rm --gpus "device=$GPU_ID" -v "$(pwd):/workspace/artifacts" spira
```

All subsequent commands must be executed inside the container.

### Step 2. Prepare the Datasets 

For our evaluation we use scenes from 3 real-world datasets that are licensed. For each of these datasets, we provide detailed instructions on how to obtain access:

* Waymo: Accept the terms in the official Waymo [website](https://waymo.com/open/terms/) to grant your Google account access to the dataset.

* ScanNet: Make sure you have a Hugging Face account and accept the conditions in this [repository ](https://huggingface.co/datasets/Pointcept/scannet-compressed). From Hugging Face go to Access Tokens -> Create New Token to generate a Read Token Type that will allow you to access your Hugging Face account from terminal. Then execute:

```shell
 export HF_TOKEN=hf_....
 ```

* KITTI: Go to SemanticKITTI [website](https://semantic-kitti.org/dataset.html) and click on "Download KITTI Odometry Benchmark Velodyne point clouds" to receive the download URL in your email. Then execute:
```shell
export URL=https://..../data_odometry_velodyne.zip
 ```
Then the following snippet will download and prepare all 3 datasets:

```shell
cd datasets/
bash download_scannet.sh
bash download_kitti.sh
bash download_waymo.sh
 ```
Note: The script that downloads the Waymo dataset requires web-based Google Authentication. A link will be printed for login via a browser and then a temporary token to authorize the download will be provided.

For our evaluation we also use synthetic randomly distributed voxel scenes. Execute the following command to generate them:

```shell
python3 random_voxel_generator.py
cd .. 
 ```
### Step 3. Run the Experiments

We have 5 scripts/experiments that evaluate Spira:

#### Run All Experiments

The following command will execute all experiments and generate all figures in the /figures subfolder:

```shell
bash automate/run_all.sh
 ```

For more detailed explanation of the experiments see below. Each script can be configured by modifying the experiment parameters defined within the corresponding *.sh file (e.g., scenes, networks, baselines).

#### (A) End-to-End Inference Performance 

The following command will evaluate end-to-end inference performance of all Sparse Convolution Engines using different datasets and networks (Figures 8-9 of paper):

```shell
bash automate/end_to_end.sh
 ```

#### (B) Layerwise Performance

The following command will evaluate layerwise performance of all Sparse Convolution Engines averaged across all datasets for different Sparse Convolution layer configurations (Figure 10 of paper):

```shell
bash automate/layerwise.sh
 ```

#### (C) Mapping Performance

The following command will evaluate the mapping performance in voxel indexing step of all Sparse Convolution Engines across varying input coordinate counts and layer kernel sizes (Figure 12 of paper):

```shell
bash automate/mapping.sh
 ```

#### (D) Scene Density Ablation Study

The following command will evaluate end-to-end inference performance of all Sparse Convolution Engines averaged across all networks for synthetic scenes of varying sparsity (Figure 16 of paper):

```shell
bash automate/ablation.sh
 ```

#### (E) Correctness

The following command will compare the output of Sparse Convolution Engines (coordinates and features) across different layer configurations (For Spira we verify all different threshold selections):

```shell
bash automate/correctness.sh
 ```


## Contact

For any suggestions for improvement, any issues related to Spira source code or for reporting bugs, please contact Dionysios Adamopoulos at dionadam2013@gmail.com.

