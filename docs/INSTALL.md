## Follow these steps to install the environment
- **STEP 1: Create environment**
    ```
    ## python3.8 should be strictly followed.
    conda create -n m3cad python=3.8
    conda activate m3cad
    ```
- **STEP 2: Install cudatoolkit**

    If CUDA is already installed, update environment variables:
    ```sh
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CUDNN_PATH=$CUDA_HOME/lib64
    ```
    If CUDA is not installed, install it via Conda:
    ```sh
    conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    ```
- **STEP 3: Install torch**
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
- **STEP 4: Set environment variables**
    ```
    # GCC 9.4 is strongly recommended. Otherwise, it might encounter errors.
    export PATH=YOUR_GCC_PATH/bin:$PATH
    ```
- **STEP 5: Install ninja and packaging**
    ```
    pip install ninja packaging
    ```
- **STEP 6: Install our repo**
    ```
    pip install -v -e .
    ```
- **STEP 7: Install other dependencies (including customized nuscenes-devkit and OpenCOOD)**
    ```
    git submodule update --init --recursive &&
    cd submodules/nuscenes-devkit/setup && pip install -e .
    cd submodules/OpenCOOD/setup && pip install -e . 
    ```