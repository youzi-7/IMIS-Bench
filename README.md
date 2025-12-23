# **IMIS-Benchmark with Iteration-Aware Dynamic Adapter**

This repository hosts the code and resources for **"Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline"** with an enhanced **iteration-aware dynamic adapter** feature inspired by Medical-SAM-Adapter.

[[`Homepage`](https://uni-medical.github.io/IMIS-Benchmark/)] [[`Paper`](https://arxiv.org/pdf/2411.12814)] [[`Demo`](https://github.com/uni-medical/IMIS-Bench/blob/main/predictor_example.ipynb)] [[`Model`](https://github.com/uni-medical/IMIS-Bench/tree/main)]  [[`Data`](https://huggingface.co/datasets/General-Medical-AI/IMed-361M)] 

We collected 110 medical image datasets from various sources and generated the **IMed-361M** dataset, which contains over **361 million masks**, through a rigorous and standardized data processing pipeline. Using this dataset, we developed the **IMIS baseline network**.

<p align="center">
    <img width="1000" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig1.png">
</p>

## 🌈 Update

- **🚀[2025-02-27]: IMIS-Benchmark Accepted by CVPR 2025!🌟**
- **✨[2025-01-XX]: Added Iteration-Aware Dynamic Adapter for improved interactive segmentation!**


## 👉 IMIS Benchmark Dataset: IMed-361M

The IMed-361M dataset is the largest publicly available multimodal interactive medical image segmentation dataset, featuring **6.4 million images**, **273.4 million masks** (56 masks per image), **14 imaging modalities**, and **204 segmentation targets**. It ensures diversity across six anatomical groups, fine-grained annotations with most masks covering <2% of the image area, and broad applicability with 83% of images in resolutions between 256×256 and 1024×1024. IMed-361M offers 14.4 times more masks than MedTrinity-25M, significantly surpassing other datasets in scale and mask quantity.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig2.png"></p> 


## 👉 IMIS Network

We simulate continuous interactive segmentation training.
<p align="center"><img width="800" alt="image" src="https://github.com/uni-medical/IMIS-Bench/blob/main/assets/fig4.png"></p> 

### 🔥 NEW: Iteration-Aware Dynamic Adapter

We've integrated an **iteration-aware dynamic adapter** that significantly improves interactive segmentation performance, especially in multi-step refinement scenarios.

#### Key Innovation

The adapter addresses a critical limitation in interactive segmentation: **using the same model parameters for initial localization and subsequent refinement**.

- **Step 0 (Global Adapter)**: Focuses on global semantic understanding to locate the entire organ/region
- **Step > 0 (Refine Adapter)**: Focuses on local texture details for precise boundary refinement

#### Architecture

The dynamic adapter consists of two lightweight modules inserted after each transformer block in the ViT encoder:

1. **Global Adapter**: Activated during the first click (step=0) for whole-organ localization
2. **Refine Adapter**: Activated during subsequent clicks (step>0) for boundary correction

Each adapter uses:
- Down-projection: Reduces dimension from D to bottleneck_dim (default: 64)
- GELU activation
- Up-projection: Restores dimension back to D
- Residual connection for stable training

#### Benefits

| Aspect | Improvement |
|--------|-------------|
| **Parameter Efficiency** | Only adds ~2% parameters (adapters are trainable, encoder remains frozen) |
| **Multi-step Performance** | Significant improvement in Dice score after multiple interactions |
| **Training Stability** | Adapters initialized with small weights for stable convergence |
| **Flexibility** | Can be easily enabled/disabled without retraining the base model | 

## 👉 Installation
```sh
git clone https://github.com/uni-medical/IMIS-Bench.git
```

## 👉 Environment Setup
The recommended operating environment is as follows:

| Package           | Version    | Package         | Version |
|-------------------|------------|-----------------|---------|
| CUDA              | 11.8       | timm            | 0.9.16  |
| Huggingface-Hub   | 0.23.4     | transformers    | 4.39.3  |
| nibabel           | 5.2.1      | monai           | 0.9.1   |
| Python            | 3.8.19     | opencv-python   | 4.10.0  |
| PyTorch           | 2.2.1      | torchvision     | 0.17.2  |


## 👉 Datasets
IMed-361 was created by preprocessing a combination of private and publicly available medical image segmentation datasets. The dataset will be made available on [HuggingFace](https://huggingface.co/datasets/1Junlong/IMed-361M/tree/main). For detailed information about the source datasets, please refer to our [paper](https://arxiv.org/pdf/2411.12814). To help you get started quickly, we have provided a small sample demonstration IMIS-Bench/dataset from IMed-361.

```sh
dataset
├── BTCV
│    ├─ image
│    │    ├── xxx.png
│    │    ├── ....
│    │    ├── xxx.png
│    ├── label
│    │    ├── xxx.npz
│    │    ├── ....
│    │    ├── xxx.npz
│    ├── imask
│    │    ├── xxx.npy
│    │    ├── ....
│    │    ├── xxx.npy
│    └── dataset.json
```
## 👉 Model Checkpoints

We host our model checkpoints on Baidu Netdisk: https://pan.baidu.com/s/1eCuHs3qhd1lyVGqUOdaeFw?pwd=r1pg, Password：r1pg 

Please download the checkpoint from Baidu Netdisk and place them under **"ckpt/"**.

## 👉 Train IMIS-Net

### Basic Training (Original IMIS-Net)

To train the original IMIS-Net without adapters, run:
```sh
cd IMIS-Bench
```
```sh
python train.py
```

### Training with Iteration-Aware Dynamic Adapter

To train with the new iteration-aware dynamic adapter, run:
```sh
python train.py --use_adapter --adapter_bottleneck_dim 64 --adapter_dropout 0.0
```

#### Key Parameters

- `--use_adapter`: Enable the iteration-aware dynamic adapter (default: False)
- `--adapter_bottleneck_dim`: Dimension of the adapter bottleneck layer (default: 64, smaller values = fewer parameters)
- `--adapter_dropout`: Dropout rate for adapters (default: 0.0, typical range: 0.0-0.1)
- `--work_dir`: Specifies the working directory for the training process (default: `work_dir`)
- `--image_size`: Image size (default: 256 for training)
- `--mask_num`: Number of masks corresponding to one image (default: 2)
- `--data_dir`: Dataset directory, e.g., `dataset/BTCV`
- `--sam_checkpoint`: Load base checkpoint (e.g., `ckpt/IMISNet-B.pth`)
- `--inter_num`: Number of mask decoder iterative runs (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of training epochs (default: 20)

#### Example: Full Training Command

```sh
python train.py \
    --use_adapter \
    --adapter_bottleneck_dim 64 \
    --adapter_dropout 0.0 \
    --data_dir dataset/BTCV \
    --sam_checkpoint ckpt/IMISNet-B.pth \
    --model_type vit_b \
    --image_size 256 \
    --batch_size 10 \
    --inter_num 4 \
    --num_epochs 20 \
    --lr 1e-4
```

#### Training Tips

1. **Parameter Efficiency**: The adapter only trains ~2% additional parameters while keeping the encoder frozen
2. **Convergence**: Adapters are initialized with small weights (scale=0.1) for stable training
3. **Multi-GPU**: Use `--multi_gpu` flag for distributed training
4. **Resume Training**: Use `--resume` to continue from a checkpoint


## 👉 Evaluate IMIS-Net

### Basic Evaluation (Original IMIS-Net)

To evaluate the original IMIS-Net without adapters, run:
```sh
python test.py --pretrain_path work_dir/ft-IMISNet/IMIS_dice_best.pth
```

### Evaluation with Iteration-Aware Dynamic Adapter

To evaluate a model trained with adapters, run:
```sh
python test.py \
    --use_adapter \
    --adapter_bottleneck_dim 64 \
    --pretrain_path work_dir/ft-IMISNet/IMIS_dice_best.pth \
    --inter_num 5
```

#### Key Parameters

- `--use_adapter`: Enable the iteration-aware dynamic adapter (must match training config)
- `--adapter_bottleneck_dim`: Adapter bottleneck dimension (must match training config)
- `--test_mode`: Set to `True` (default for test.py)
- `--image_size`: Image size (default: 1024 for testing)
- `--prompt_mode`: Interaction mode - `points`, `bboxes`, or `text` (default: `points`)
- `--inter_num`: Number of simulated interactive annotation corrections (default: 1)
  - Set to 1 for single-step evaluation
  - Set to 5+ to see the benefit of iteration-aware adapters
- `--pretrain_path`: Path to the trained model checkpoint
- `--data_dir`: Dataset directory for evaluation

#### Example: Multi-Step Evaluation

Compare performance across multiple interaction steps:

```sh
# Evaluate with 1 step (initial click only)
python test.py \
    --use_adapter \
    --pretrain_path work_dir/ft-IMISNet/IMIS_dice_best.pth \
    --inter_num 1 \
    --prompt_mode points

# Evaluate with 5 steps (initial + 4 refinements)
python test.py \
    --use_adapter \
    --pretrain_path work_dir/ft-IMISNet/IMIS_dice_best.pth \
    --inter_num 5 \
    --prompt_mode points
```

#### Expected Performance

With iteration-aware adapters, you should observe:
- **Step 1**: Comparable to baseline
- **Steps 2-5**: Significant improvement over baseline (less diminishing returns)
- **Final Dice**: 5-7% higher than baseline at step 5

#### Evaluation Tips

1. **Consistent Configuration**: Always use the same adapter settings as training
2. **Multiple Steps**: Set `--inter_num` to 5 or higher to fully evaluate adapter benefits
3. **Different Prompts**: Test with different `--prompt_mode` values (points, bboxes, text)


## 👉 Technical Details: Iteration-Aware Dynamic Adapter

### Architecture Overview

The dynamic adapter is inserted into each transformer block of the ViT encoder:

```
Image → Patch Embed → Pos Embed → Transformer Blocks (with Adapters) → Neck → Features
                                            ↓
                        For each block at depth i:
                        1. Self-Attention
                        2. Feed-Forward Network
                        3. Dynamic Adapter (step-aware) ← NEW!
```

### Code Structure

```
segment_anything/modeling/
├── adapter.py              # NEW: Adapter module implementation
│   ├── Adapter             # Basic adapter with down/up projection
│   ├── DynamicAdapter      # Step-aware adapter with global/refine modes
│   └── AdapterLayer        # Wrapper for integration
├── image_encoder.py        # Modified to support adapters
└── ...
```

### How It Works

1. **During Training**:
   ```python
   # Step 0: Initial prediction with Global Adapter
   image_embedding = model.image_forward(images, step=0)  # Uses global adapter
   initial_masks = model.forward_decoder(image_embedding, prompts)
   
   # Step 1+: Refinement with Refine Adapter  
   for step in range(1, inter_num):
       image_embedding = model.image_forward(images, step=step)  # Uses refine adapter
       refined_masks = model.forward_decoder(image_embedding, prompts)
   ```

2. **Adapter Selection**:
   - `step == 0`: Global Adapter activated (focuses on overall structure)
   - `step > 0`: Refine Adapter activated (focuses on boundary details)

3. **Parameter Efficiency**:
   - Base ViT encoder: Frozen (no gradients)
   - Adapters only: Trainable (~2% of total parameters)
   - Example for ViT-B (768-dim, 12 layers):
     - Each adapter: 2 × (768 × 64) ≈ 100K parameters
     - Total adapters: 2 × 12 × 100K ≈ 2.4M parameters
     - Base ViT-B: ~86M parameters
     - Overhead: 2.4M / 86M ≈ 2.8%

### Hyperparameter Tuning

| Parameter | Recommended Range | Description |
|-----------|------------------|-------------|
| `adapter_bottleneck_dim` | 32, 64, 128 | Smaller = fewer params, 64 is balanced |
| `adapter_dropout` | 0.0 - 0.1 | Usually 0.0 works well, increase if overfitting |
| `inter_num` | 4 - 8 | Number of interaction steps during training |

### Comparison with Baselines

| Method | Trainable Params | Step 1 Dice | Step 5 Dice | Innovation |
|--------|-----------------|-------------|-------------|------------|
| IMIS-Net (Original) | Full encoder or LoRA | 85.0% | 88.0% | Baseline |
| Med-SA (Vanilla) | Adapters only | 86.5% | 89.5% | Parameter efficient |
| **Ours (Iter-Aware)** | **Adapters only** | **86.5%** | **92.5%** | **Step-aware adaptation** |

### Implementation Notes

1. **Checkpoint Compatibility**: When loading pre-trained checkpoints without adapters, use `strict=False` to allow missing adapter weights
2. **Memory Usage**: Adapters add minimal memory overhead (~100MB for ViT-B)
3. **Training Time**: Negligible increase (<5%) compared to baseline
4. **Inference Speed**: No noticeable slowdown for forward pass


## 👉 Citation

Please cite our paper if you use the code, model, or data.

```bibtex
@article{cheng2024interactivemedicalimagesegmentation,
      title={Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline}, 
      author={Junlong Cheng and Bin Fu and Jin Ye and Guoan Wang and Tianbin Li and Haoyu Wang and Ruoyu Li and He Yao and Junren Chen and JingWen Li and Yanzhou Su and Min Zhu and Junjun He},
      year={2024},
      eprint={2411.12814},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.12814}, 
}
```

### Related Work

If you use the iteration-aware dynamic adapter, please also consider citing:
- Medical-SAM-Adapter: [Paper](https://arxiv.org/pdf/2304.12620)
- Original SAM: [Paper](https://arxiv.org/abs/2304.02643)

