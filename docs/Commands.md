Commands for running experiments.
===========
# Training
### Deep Sets
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type deepset 
```

### Attention MIL
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode path --model_type amil 
```

### Cluster MI-FCN
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode cluster --model_type mifcn 
```

### DeepGraphConv
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode graph --model_type dgc 
```

### Patch-GCN
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec --mode graph --model_type patchgcn 
```