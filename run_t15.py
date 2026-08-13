"""Definitive T=15 battery: committed model, horizon extension only."""
import sys
sys.path.insert(0, '.')
import src.experiments.exp3_emergency as e3
e3.T = 15
e3.run_experiment(save_dir='results_t15')
