#!/usr/bin/env python3
"""
Proof of concept: run deepcf and rpucb_attn on all 3 datasets with reduced epochs.
This script runs all 6 experiments sequentially and prints the final results table.
Works on Windows, macOS, and Linux.
"""

import subprocess
import sys
from datetime import datetime
import torch

# Colors for output (works on Windows 10+)
class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*40}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{'='*40}{Colors.END}\n")

def print_run_header(run_num, model, dataset):
    print(f"{Colors.BLUE}{'-'*40}{Colors.END}")
    print(f"{Colors.YELLOW}[Run {run_num}/6]{Colors.END} Starting: {Colors.GREEN}{model}{Colors.END} on {Colors.GREEN}{dataset}{Colors.END}")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{Colors.BLUE}{'-'*40}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}\n")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}\n")

def main():
    print_header("RP-UCB PoC Experiment Suite")
    
    print("Running 2 models (deepcf, rpucb_attn) × 3 datasets (ml-1m, AMusic, citeulike)")
    print("Single run (--runs 1) with reduced epochs for fast feedback")
    print("Estimated time: ~3h45m on CPU\n")
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"{Colors.GREEN}GPU detected, using CUDA{Colors.END}\n")
    else:
        device = "cpu"
        print(f"{Colors.YELLOW}No GPU found, using CPU (this will be slower){Colors.END}\n")
    
    print(f"{Colors.YELLOW}Start time: {datetime.now()}{Colors.END}\n")
    
    # Define experiments
    datasets = ["ml-1m", "AMusic", "citeulike"]
    models = ["deepcf", "rpucb_attn"]
    
    # Track results
    total_runs = 0
    completed_runs = 0
    failed_runs = []
    
    # Run all experiments
    for dataset in datasets:
        for model in models:
            total_runs += 1
            
            print_run_header(total_runs, model, dataset)
            
            cmd = [
                sys.executable,
                "main.py",
                "--model", model,
                "--dataset", dataset,
                "--config", f"configs/{dataset}.yaml",
                "--device", device,
                "--runs", "1"
            ]
            
            try:
                result = subprocess.run(cmd, check=True)
                completed_runs += 1
                print_success(f"Completed: {model} on {dataset}")
            except subprocess.CalledProcessError as e:
                failed_runs.append(f"{model} on {dataset}")
                print_error(f"Failed: {model} on {dataset} (exit code: {e.returncode})")
            except Exception as e:
                failed_runs.append(f"{model} on {dataset}")
                print_error(f"Error running {model} on {dataset}: {e}")
    
    # Print summary
    print_header("All Experiments Complete")
    
    print("Summary:")
    print(f"  Total runs: {total_runs}")
    print(f"  {Colors.GREEN}Completed: {completed_runs}{Colors.END}")
    if failed_runs:
        print(f"  {Colors.RED}Failed: {len(failed_runs)}{Colors.END}")
        for failed in failed_runs:
            print(f"    - {failed}")
    
    print(f"\n{Colors.YELLOW}End time: {datetime.now()}{Colors.END}\n")
    
    # Print results table
    print_header("Results Table")
    
    try:
        from src.utils import print_results_table
        print_results_table()
    except Exception as e:
        print(f"{Colors.RED}Error printing results table: {e}{Colors.END}")
    
    print(f"\n{Colors.GREEN}Done! Check the results above.{Colors.END}\n")

if __name__ == "__main__":
    main()