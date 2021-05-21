Commands for running experiments.
===========
# Training
### Deep Sets
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode path --model_type deepset 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode path --model_type deepset 
```

### Attention MIL
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode path --model_type amil 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode path --model_type amil 
```

### Cluster MI-FCN
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode cluster --model_type mifcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode cluster --model_type mifcn 
```

### DeepGraphConv
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode graph --model_type dgc 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode graph --model_type dgc 
```

### Patch-GCN
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_blca_100 --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_brca_100 --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_gbmlgg_100 --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_luad_100 --mode graph --model_type patchgcn 
CUDA_VISIBLE_DEVICES=0 python main.py --which_splits 5foldcv --split_dir tcga_ucec_100 --mode graph --model_type patchgcn 
```