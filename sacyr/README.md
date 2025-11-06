# Running Sweeps with Weights & Biases

Follow these steps to launch and run a sweep in wandb:

0. **Dependencies**

Install pytorch first, then add the next modules:
```python
pip install wandb scikit-learn tqdm tensorboard rtdl_num_embeddings omegaconf holidays matplotlib
```

# Link python to python3 on cluster

```bash
ln -s /usr/bin/python3 /usr/bin/python
```

1. **Create the sweep:** activate the conda environment and navigate to the root of the repository


```bash
wandb sweep configs/sweep_periodical_mlp.yaml
```
This command will return a sweep_id

2. **Ejecutar el agente para el sweep:**

It returns the command to run sweeps, something like wandb agent inaki/my-template/874plwyb

```bash
wandb agent my_entity/my_project/SWEEP_ID --count 5
```

This will execute up to 5 runs of the specified sweep. Each run will search a set of parameters

3. **Script to Run multiple agents in parallel:**

It automatically creates N agents that split the given executions in parallel. In the following example, 100 executions of the search space are established, parallelized across 5 agents, for MLP + periodic embedding.
```python
python run_sweep.py --config configs/cnn_lstm.yaml --total-runs 100 --processes 1
```

```python
python run_sweep.py --config configs/sweep_periodical_mlp.yaml --total-runs 100 --processes 1
```