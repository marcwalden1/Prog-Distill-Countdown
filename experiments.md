# Experiments

This file tracks completed training runs.

---

## Qwen2.5-0.5B / balanced-grpo-seed1

**Date:** Wed Apr  1 00:47:27 EDT 2026
**Duration:** 2h 55m
**SLURM Job ID:** 2929935

### Setup
| Field | Value |
|---|---|
| Base model | Qwen2.5-0.5B |
| Data source | `data/balanced/` |
| Algorithm | GRPO |
| Train batch size | 256 |
| Rollout n | 4 |
| Val n | 4 |
| Learning rate | 1e-6 |
| Max response length | 1024 |
| KL loss coef | 0.001 |
| KL loss type | low_var_kl |
| Entropy coeff | 0 |
| Loss aggregation | token-mean |
| Advantage normalization | False |
| KL in reward | False |
| Save/test freq | every 50 steps |
| Total epochs | 1 |
| vLLM GPU mem util | 0.6 |
| GPUs | 4 × nvidia_h100_80gb_hbm3 |
| Node | holygpu8a17604.rc.fas.harvard.edu |
| Total steps | 1635 |

### Training metrics (step 1635)
| Metric | Value |
|---|---|
| Val reward mean@4 | 0.4128 |
| Actor entropy | 0.0058 |
| KL loss | 0.030464 |
| Grad norm | 0.3706 |
| Response length (mean) | 21.8 |
| Response clip ratio | 0.0000 |

### Eval metrics (step 1600, n=32, temp=0.6, dataset=balanced)
| Dataset | Accuracy |
|---|---|
| balanced (n=3,4) | 0.2149 |

### Training curve
![Training curve](figures/Qwen2.5-0.5B/balanced-grpo-seed1/training_curve.png)

### Eval accuracy curve
![Eval accuracy](figures/Qwen2.5-0.5B/balanced-grpo-seed1/eval_accuracy.png)

### Replication
```bash
MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-grpo-seed1 DATA_SOURCE=balanced sbatch scripts/train_grpo.sh
MODEL_NAME=Qwen2.5-0.5B EXP_NAME=balanced-grpo-seed1 CHECKPOINT_DIR=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/rl-checkpoints sbatch --array=1-32 scripts/eval.sh
```

---

## Qwen2.5-1.5B / balanced-grpo-seed1

**Date:** Wed Apr  1 03:43:51 EDT 2026
**Duration:** 7h 15m
**SLURM Job ID:** 2929936

### Setup
| Field | Value |
|---|---|
| Base model | Qwen2.5-1.5B |
| Data source | `data/balanced/` |
| Algorithm | GRPO |
| Train batch size | 256 |
| Rollout n | 4 |
| Val n | 4 |
| Learning rate | 1e-6 |
| Max response length | 1024 |
| KL loss coef | 0.001 |
| KL loss type | low_var_kl |
| Entropy coeff | 0 |
| Loss aggregation | token-mean |
| Advantage normalization | False |
| KL in reward | False |
| Save/test freq | every 50 steps |
| Total epochs | 1 |
| vLLM GPU mem util | 0.6 |
| GPUs | 4 × nvidia_h100_80gb_hbm3 |
| Node | holygpu8a17604.rc.fas.harvard.edu |
| Total steps | 1635 |

### Training metrics (step 1635)
| Metric | Value |
|---|---|
| Val reward mean@4 | 0.6845 |
| Actor entropy | 0.0938 |
| KL loss | 0.006382 |
| Grad norm | 0.0269 |
| Response length (mean) | 493.7 |
| Response clip ratio | 0.2930 |

### Eval metrics (step 1600, n=32, temp=0.6, dataset=balanced)
| Dataset | Accuracy |
|---|---|
| balanced (n=3,4) | 0.6087 |

### Training curve
![Training curve](figures/Qwen2.5-1.5B/balanced-grpo-seed1/training_curve.png)

### Eval accuracy curve
![Eval accuracy](figures/Qwen2.5-1.5B/balanced-grpo-seed1/eval_accuracy.png)

### Replication
```bash
MODEL_NAME=Qwen2.5-1.5B EXP_NAME=balanced-grpo-seed1 DATA_SOURCE=balanced sbatch scripts/train_grpo.sh
MODEL_NAME=Qwen2.5-1.5B EXP_NAME=balanced-grpo-seed1 CHECKPOINT_DIR=/n/holylabs/LABS/kdbrantley_lab/Lab/mwalden/rl-checkpoints sbatch --array=1-32 scripts/eval.sh
```
