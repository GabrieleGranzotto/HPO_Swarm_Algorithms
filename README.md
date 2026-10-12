# Swarm Intelligence-Based Hyperparameter Optimization (HPO) Framework for CNNs

## Project Overview

This project implements a modular, scalable, and hardware-aware framework for the hyperparameter optimization (HPO) of Convolutional Neural Networks (CNNs) in PyTorch. Moving away from traditional, computationally expensive techniques like Grid Search or Random Search, the system leverages meta-heuristic algorithms based on Swarm Intelligence to efficiently and targetedly explore the search space.

Designed with a strong emphasis on performance and hardware safety, the framework orchestrates the parallel training of neural architectures on GPUs/CPUs, maximizing computational throughput without running into Out-Of-Memory (OOM) errors.

---

## Architecture & Core Features

The framework has been engineered following the principles of modularity (Separation of Concerns) and abstraction, isolating the optimization logic from the neural training pipeline.

* **Dynamic Hardware-Aware Parallelization:** Swarm execution is not purely sequential. The system dynamically calculates available GPU VRAM (minus a defined `safe_vram_margin_mb`) and instantiates an asynchronous thread pool (`ThreadPoolExecutor`). This ensures the simultaneous training of the maximum number of neural networks permitted by the hardware, drastically reducing execution times.
* **Automatic Mixed Precision (AMP):** Native integration of `torch.autocast` and `GradScaler` to execute computationally intensive matrix operations in `float16`/`bfloat16` formats. This halves the memory footprint of each model, allowing for double the amount of parallel workers and accelerating forward/backward pass times.
* **Divergence Early Stopping:** A surgical safety mechanism that monitors gradient explosion (loss returning `NaN`). If a particle explores a toxic hyperparameter combination, training is instantly aborted, saving clock cycles and assigning a penalizing fitness score to guide the swarm away from that region.
* **Centralized Configuration (Decoupling):** The entire environment (search space bounds, architectural hyperparameters, dataset ratios) is driven by a `config.yml` file. No *magic numbers* are hardcoded into the system.
* **Asynchronous Telemetry & Logging:** A centralized `Orchestrator` injects measurement hooks directly into the swarm's lifecycle, extracting generation times (throughput) and fitness metrics without altering the algorithm's underlying mathematics. Data is batch-dumped to JSON files to prevent I/O bottlenecks.

---

## Swarm Intelligence Algorithms

The core of the search space exploration relies on two bio-inspired algorithms, designed to dynamically balance global exploration and local exploitation.

### 1. Particle Swarm Optimization (PSO)

A classic, robust algorithm inspired by the flocking behavior of birds.

* **Logic:** Each solution (particle) moves through the n-dimensional hyperparameter space by updating its velocity vector.
* **Dynamics:** Movement is driven by three forces: inertia (the particle's previous direction), the *cognitive* component (memory of the best position found by the individual particle), and the *social* component (attraction toward the Global Best found by the entire swarm).
* **Use Case:** Ideal for rapid and stable convergence toward global minima/maxima.

### 2. Dung Beetle Optimizer (DBO)

An innovative and sophisticated meta-heuristic algorithm modeling the survival behaviors of dung beetles, offering superior diversification during exploration.

* **Logic:** The swarm is divided into four sub-populations, each with a specific task and dedicated update mathematics:
  * **Rolling:** Beetles rolling the dung ball (broad exploration influenced by wind/light and the global worst position to avoid local minima).
  * **Brooding:** Beetles laying eggs in a dynamic safe area around the Global Best (high-precision local exploitation).
  * **Foraging:** Young beetles exploring a wider area around the nest (searching for promising variants).
  * **Stealing:** Beetles stealing resources from others, moving toward the Global Best in a stochastic and unpredictable manner, ensuring escape from local minima.

---

## System Components (Modules)

1. `network.py`: Contains the `NN_Environment`, the evaluation proxy acting as the oracle for the swarm. It implements a custom `LightweightCNN` and manages training pipelines, data loaders, spatial transformations, AMP, and denormalization functions.
2. `swarm_algorithms.py`: The mathematical core of the project. It provides the base `SwarmAlgorithm` infrastructure for unified history management and ThreadPool-based parallelization, alongside specialized `PSO` and `DBO` subclasses.
3. `orchestrator.py`: The conductor. Initializes the hardware environment, links the YAML configuration, manages high-precision timers via hooks, and saves experiment history to ensure total reproducibility.

---

## Results & Computational Analysis

*(Pending data collection)*
*Performance analysis, algorithmic convergence metrics, and parallel throughput results (Wall-Clock time and factorial speedup) will be integrated upon completion of the execution cycles.*
