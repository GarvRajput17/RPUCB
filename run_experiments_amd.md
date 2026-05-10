# Running Experiments (AMD GPU Guide)

If you are using an AMD GPU, PyTorch relies on **ROCm** (Radeon Open Compute) for hardware acceleration. The commands below are set to use `--device auto`, which means PyTorch will automatically try to detect and use your GPU if everything is installed correctly. 
*(Note: If it can't find a compatible GPU setup, it will reliably fall back to using the CPU without crashing).*

These commands are also set to `--runs 1` to give you a single, fast training pass per experiment.

---

## Part 1: ML-1M Dataset Experiments

**1. Baseline DeepCF**
```bash
python main.py --model deepcf --dataset ml-1m --config configs/ml-1m.yaml --device auto --runs 1
```

**2. Static Mask (Baseline)**
```bash
python main.py --model static_mask --dataset ml-1m --config configs/ml-1m.yaml --device auto --runs 1
```

**3. Static Mask with Attention (Baseline)**
```bash
python main.py --model static_mask_attn --dataset ml-1m --config configs/ml-1m.yaml --device cuda:0 --runs 1
```

**4. RP-UCB (Our Method)**
```bash
python main.py --model rpucb --dataset ml-1m --config configs/ml-1m.yaml --device cuda:0 --runs 1
```

**5. RP-UCB with Attention (Our Method)**
```bash
python main.py --model rpucb_attn --dataset ml-1m --config configs/ml-1m.yaml --device cuda:0 --runs 1
```

---

## Part 2: Amazon Music (AMusic) Dataset Experiments

**6. Baseline DeepCF**
```bash
python main.py --model deepcf --dataset AMusic --config configs/AMusic.yaml --device cuda:0 --runs 1
```

**7. Static Mask (Baseline)**
```bash
python main.py --model static_mask --dataset AMusic --config configs/AMusic.yaml --device cuda:0 --runs 1
```

**8. Static Mask with Attention (Baseline)**
```bash
python main.py --model static_mask_attn --dataset AMusic --config configs/AMusic.yaml --device cuda:0 --runs 1
```

**9. RP-UCB (Our Method)**
```bash
python main.py --model rpucb --dataset AMusic --config configs/AMusic.yaml --device cuda:0 --runs 1
```

**10. RP-UCB with Attention (Our Method)**
```bash
python main.py --model rpucb_attn --dataset AMusic --config configs/AMusic.yaml --device cuda:0 --runs 1
```

---

## Part 3: CiteULike Dataset Experiments

**11. Baseline DeepCF**
```bash
python main.py --model deepcf --dataset citeulike --config configs/citeulike.yaml --device cuda:0 --runs 1
```

**12. Static Mask (Baseline)**
```bash
python main.py --model static_mask --dataset citeulike --config configs/citeulike.yaml --device cuda:0 --runs 1
```

**13. Static Mask with Attention (Baseline)**
```bash
python main.py --model static_mask_attn --dataset citeulike --config configs/citeulike.yaml --device cuda:0 --runs 1
```

**14. RP-UCB (Our Method)**
```bash
python main.py --model rpucb --dataset citeulike --config configs/citeulike.yaml --device cuda:0 --runs 1
```

**15. RP-UCB with Attention (Our Method)**
```bash
python main.py --model rpucb_attn --dataset citeulike --config configs/citeulike.yaml --device cuda:0 --runs 1
```

---

## Part 4: Display the Final Results

After running the experiments, run this final command to print out the aggregated summary table:

**16. Print Results Table**
```bash
python -c "from src.utils import print_results_table; print_results_table()"
```
