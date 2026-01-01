# Federated Learning Simulation Platform

## Overview

This project simulates a federated learning setup using TensorFlow with multiple client nodes training on private synthetic data. It implements federated averaging manually and compares it with centralized training.

### What is Federated Learning?

Federated learning allows multiple participants (clients) to collaboratively train a shared model without exchanging their raw data. Instead of collecting all data in a central location, the model is trained locally on each client's device, and only model updates (gradients or weights) are shared with a central server. This approach addresses key challenges in machine learning:

- **Privacy Preservation**: Sensitive data never leaves the client's device
- **Data Sovereignty**: Organizations maintain control over their data
- **Reduced Communication Costs**: Only model updates are transmitted, not raw data
- **Scalability**: Can leverage distributed computing resources

### Federated Averaging Algorithm

The core algorithm implemented in this simulation is Federated Averaging (FedAvg), proposed by McMahan et al. The process works as follows:

1. **Initialization**: A global model is initialized and distributed to participating clients
2. **Local Training**: Each selected client trains the model on their local data for one or more epochs
3. **Model Aggregation**: The central server collects model updates from clients and computes a weighted average
4. **Global Update**: The averaged model becomes the new global model for the next round
5. **Iteration**: Steps 2-4 repeat for multiple rounds until convergence

### Comparison with Centralized Training

This simulation compares federated learning with traditional centralized training:

- **Centralized Training**: All data is collected at a central server, model is trained on the aggregated dataset
- **Federated Training**: Model training is distributed across clients, only model updates are shared

The comparison helps understand the trade-offs between privacy/communication efficiency and potential performance differences.

## Features

- Parameterizable client count and data distribution
- Manual implementation of federated averaging algorithm
- Side-by-side comparison with centralized training
- Synthetic dataset generation with configurable statistical properties
- Comprehensive training logs and performance visualization
- Jupyter notebook for interactive exploration

## Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download this repository.

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate synthetic data:
   ```bash
   python src/generate_data.py
   ```

### Running the Simulation

#### Option 1: Command Line
Run the main simulation script:
```bash
python src/simulation.py
```

#### Option 2: Jupyter Notebook
Launch the interactive notebook:
```bash
jupyter notebook notebooks/federated_learning_simulation.ipynb
```

Follow the cells in order to execute the simulation step-by-step.

## Project Structure

```
federated-learning-assignment/
├── src/
│   ├── generate_data.py      # Synthetic data generation
│   └── simulation.py         # Main simulation script
├── notebooks/
│   └── federated_learning_simulation.ipynb  # Jupyter notebook
├── data/                     # Generated client datasets
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore               # Git ignore rules
```

## Configuration

You can modify the simulation parameters in the scripts:

- `num_clients`: Number of client datasets (default: 10)
- `samples_per_client`: Data samples per client (default: 1000)
- `num_rounds`: Federated learning communication rounds (default: 20)
- `num_clients_per_round`: Clients participating per round (default: 5)
- `num_epochs`: Epochs for centralized training comparison (default: 20)
- `learning_rate`: Learning rate for model training (default: 0.01)

## Simulation Details

### Data Generation
The synthetic data generator creates heterogeneous datasets across clients:
- Each client gets a unique dataset with configurable size
- Features are generated with client-specific statistical properties
- Labels are created using a linear relationship with added noise
- This simulates real-world scenarios where data distributions vary across participants

### Model Architecture
A simple neural network is used for binary classification:
- Input layer: 10 features
- Hidden layer: 64 neurons with ReLU activation
- Hidden layer: 32 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Loss function: Binary cross-entropy
- Optimizer: Adam

### Training Process
- **Federated Training**: Clients perform local training for 1 epoch per round
- **Centralized Training**: Single model trained on aggregated data for multiple epochs
- Both approaches use the same model architecture and hyperparameters for fair comparison

## Output and Results

The simulation generates:
- **Console logs**: Real-time training progress showing loss and accuracy per round/epoch
- **Performance chart** (`federated_vs_centralized_comparison.png`): Visual comparison of training curves

### Interpreting Results
- **Convergence**: Federated learning typically requires more communication rounds than centralized epochs
- **Performance Gap**: May observe slight performance differences due to distributed nature
- **Stability**: Federated averaging can be more robust to heterogeneous data distributions
- **Communication Efficiency**: Demonstrates how federated learning reduces data transmission needs

## Requirements

- TensorFlow (for model training and operations)
- NumPy (for numerical computations)
- Matplotlib (for visualization)
- Jupyter (for interactive notebook)

## Further Reading

- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629) - Original FedAvg paper
- [Towards Federated Learning at Scale: System Design](https://arxiv.org/abs/1902.01046)
- [Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977)