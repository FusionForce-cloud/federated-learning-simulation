import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from generate_data import generate_synthetic_data

def create_keras_model():
    """Create a simple neural network model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def federated_training(client_data, num_rounds=10, num_clients_per_round=5, learning_rate=0.01):
    """Perform federated training"""
    # Initialize global model
    global_model = create_keras_model()
    global_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='binary_crossentropy', metrics=['accuracy'])
    
    training_losses = []
    training_accuracies = []
    
    for round_num in range(1, num_rounds + 1):
        print(f'Round {round_num}')
        
        # Select random clients for this round
        selected_indices = np.random.choice(len(client_data), num_clients_per_round, replace=False)
        selected_clients = [client_data[i] for i in selected_indices]
        
        # Train local models
        local_weights = []
        local_losses = []
        local_accuracies = []
        
        for client_x, client_y in selected_clients:
            # Create local model with global weights
            local_model = create_keras_model()
            local_model.set_weights(global_model.get_weights())
            local_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                              loss='binary_crossentropy', metrics=['accuracy'])
            
            # Train locally
            history = local_model.fit(client_x, client_y, epochs=1, batch_size=32, verbose=0)
            
            local_weights.append(local_model.get_weights())
            local_losses.append(history.history['loss'][0])
            local_accuracies.append(history.history['accuracy'][0])
        
        # Average the weights
        new_weights = []
        for layer_idx in range(len(local_weights[0])):
            layer_weights = [w[layer_idx] for w in local_weights]
            avg_weight = np.mean(layer_weights, axis=0)
            new_weights.append(avg_weight)
        
        # Update global model
        global_model.set_weights(new_weights)
        
        # Record metrics
        avg_loss = np.mean(local_losses)
        avg_accuracy = np.mean(local_accuracies)
        training_losses.append(avg_loss)
        training_accuracies.append(avg_accuracy)
        
        print(f'  Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')
    
    return global_model, training_losses, training_accuracies

def centralized_training(client_data, num_epochs=10, learning_rate=0.01):
    """Perform centralized training for comparison"""
    # Combine all client data
    all_x = np.concatenate([x for x, y in client_data])
    all_y = np.concatenate([y for x, y in client_data])
    
    model = create_keras_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                 loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(all_x, all_y, epochs=num_epochs, batch_size=32, verbose=1)
    
    return model, history.history['loss'], history.history['accuracy']

def plot_comparison(fed_losses, fed_accuracies, cent_losses, cent_accuracies):
    """Plot performance comparison"""
    rounds = range(1, len(fed_losses) + 1)
    epochs = range(1, len(cent_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss comparison
    ax1.plot(rounds, fed_losses, label='Federated', marker='o')
    ax1.plot(epochs, cent_losses, label='Centralized', marker='s')
    ax1.set_xlabel('Training Rounds/Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy comparison
    ax2.plot(rounds, fed_accuracies, label='Federated', marker='o')
    ax2.plot(epochs, cent_accuracies, label='Centralized', marker='s')
    ax2.set_xlabel('Training Rounds/Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('federated_vs_centralized_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Parameters
    num_clients = 10
    samples_per_client = 1000
    num_rounds = 20
    num_clients_per_round = 5
    num_epochs = 20
    
    # Generate or load data
    if not os.path.exists('data/client_0_data.npz'):
        print("Generating synthetic data...")
        client_data = generate_synthetic_data(num_clients, samples_per_client)
    else:
        print("Loading existing data...")
        client_data = []
        for i in range(num_clients):
            data = np.load(f'data/client_{i}_data.npz')
            client_data.append((data['x'], data['y']))
    
    print(f"Data loaded for {len(client_data)} clients")
    
    # Federated training
    print("\nStarting federated training...")
    fed_model, fed_losses, fed_accuracies = federated_training(
        client_data, num_rounds, num_clients_per_round
    )
    
    # Centralized training
    print("\nStarting centralized training...")
    cent_model, cent_losses, cent_accuracies = centralized_training(
        client_data, num_epochs
    )
    
    # Plot comparison
    print("\nPlotting comparison...")
    plot_comparison(fed_losses, fed_accuracies, cent_losses, cent_accuracies)
    
    print("Simulation complete! Check federated_vs_centralized_comparison.png for results.")