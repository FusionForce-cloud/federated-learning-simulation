import numpy as np
import tensorflow as tf
import os

def generate_synthetic_data(num_clients=10, samples_per_client=1000, num_features=10, num_classes=2):
    """
    Generate synthetic datasets for federated learning clients.
    
    Args:
        num_clients: Number of client datasets to generate
        samples_per_client: Number of samples per client
        num_features: Number of input features
        num_classes: Number of output classes
    
    Returns:
        List of tuples (x, y) for each client
    """
    np.random.seed(42)
    client_data = []
    
    for client_id in range(num_clients):
        # Generate features with some client-specific bias
        x = np.random.randn(samples_per_client, num_features) + client_id * 0.1
        
        # Generate labels based on a simple linear relationship with noise
        weights = np.random.randn(num_features)
        logits = x @ weights + client_id * 0.5
        probabilities = 1 / (1 + np.exp(-logits))  # Sigmoid for binary classification
        
        if num_classes == 2:
            y = (probabilities > 0.5).astype(int)
        else:
            # For multi-class, use softmax
            y = np.argmax(probabilities, axis=1) if num_classes > 2 else y
        
        client_data.append((x.astype(np.float32), y.astype(np.int32)))
    
    return client_data

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs('../data', exist_ok=True)
    
    # Generate data
    client_data = generate_synthetic_data(num_clients=10, samples_per_client=1000)
    
    # Save data
    for i, (x, y) in enumerate(client_data):
        np.savez(f'../data/client_{i}_data.npz', x=x, y=y)
    
    print("Synthetic data generated and saved to data/ directory")