# Spira Artifact 

This is the artifact for the paper **Spira: Exploiting Voxel Data Structural Properties for Efficient Sparse Convolution in Point Cloud Networks (MLSys 2026)**. Excluding build time and datasets downloading, it is expected to take about **90 minutes** to finish all evaluations in the artifact. 

## Hardware & Software Requirements 

The artifact should run on hardware platforms with: 

x86-64 CPU: 
* Memory: ≥ 64GB
* Storage: ≥ 256GB
  
NVIDIA GPU: 
* Memory: ≥ 16GB
* CUDA Compute Capability (SM): 7.5+

The artifact should be executed on a Linux-based operating system with an up-to-date NVIDIA driver that supports CUDA 12.4 or newer.

## Step 0. Accessing Machines in our Lab (for MLSys 2026 AE)

Please send your SSH public key to the authors to be provided with access to a compute node of the authors' institution equipped with 2 × NVIDIA A100 GPUs, which will be the environment that this artifact will be evaluated. After receiving the key, a user account for each reviewer will be created and you will be provided with connection details.

Please coordinate with the other reviewers before running experiments and ensure that only one reviewer uses the machine at a time. Τhis coordination among reviewers is important to eliminate interference and correctly reproduce this artifact.

## Step 1. Downloading & Building the Artifact

The source code of our artifact can be found at Zenodo: [Spira_Artifact](https://zenodo.org/records/18879475). Please download the .zip file and copy it to the machine.

We recommend using Docker Engine for building the artifact to fully control all software dependencies.

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

## Step 2. Prepare the Datasets 

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
## Step 3. Run the Experiments

We have 5 scripts/experiments that evaluate Spira:

### Run All Experiments

The following command will execute all experiments and generate all figures in the /figures subfolder:

```shell
bash automate/run_all.sh
 ```

For more detailed explanation of the experiments see below. Each script can be configured by modifying the experiment parameters defined within the corresponding *.sh file (e.g., scenes, networks, baselines).

### (A) End-to-End Inference Performance 

The following command will evaluate end-to-end inference performance of all Sparse Convolution Engines using different datasets and networks (Figures 8-9 of paper):

```shell
bash automate/end_to_end.sh
 ```

### (B) Layerwise Performance

The following command will evaluate layerwise performance of all Sparse Convolution Engines averaged across all datasets for different Sparse Convolution layer configurations (Figure 10 of paper):

```shell
bash automate/layerwise.sh
 ```

### (C) Mapping Performance

The following command will evaluate the mapping performance in voxel indexing step of all Sparse Convolution Engines across varying input coordinate counts and layer kernel sizes (Figure 12 of paper):

```shell
bash automate/mapping.sh
 ```

### (D) Scene Density Ablation Study

The following command will evaluate end-to-end inference performance of all Sparse Convolution Engines averaged across all networks for synthetic scenes of varying sparsity (Figure 16 of paper):

```shell
bash automate/ablation.sh
 ```

### (E) Correctness

The following command will compare the output of Sparse Convolution Engines (coordinates and features) across different layer configurations (For Spira we verify all different threshold selections):

```shell
bash automate/correctness.sh
 ```
