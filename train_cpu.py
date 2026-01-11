import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder

def relu(Z): return np.maximum(0, Z)
def relu_deriv(Z): return Z > 0
def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True)

def run_cpu_training(epochs=50, hidden_size=256, lr=0.01):
    print("\n--- ENTRAÎNEMENT CPU (NumPy) ---")
    
    # Chargement
    X_train = np.load("data_ready/X_train.npy")
    y_train = np.load("data_ready/y_train.npy")
    X_test = np.load("data_ready/X_test.npy")
    y_test = np.load("data_ready/y_test.npy")

    # On multiplie la taille du dataset par 50 !
    X_train = np.tile(X_train, (10, 1)) 
    y_train = np.tile(y_train, (10,))
        
    # One-Hot
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
    
    input_size = X_train.shape[1]
    output_size = y_train_onehot.shape[1]
    m = X_train.shape[0]

    # Init Poids
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
    b2 = np.zeros((1, output_size))

    # Historique complet (Dictionnaire)
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'epochs': [],
        'time_per_epoch': []
    }

    start_total = time.time()

    for epoch in range(epochs + 1):
        t0 = time.time()
        
        # --- Forward Train ---
        Z1 = np.dot(X_train, W1) + b1
        A1 = relu(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = softmax(Z2)
        
        # --- Backward Train ---
        dZ2 = A2 - y_train_onehot
        dW2 = (1/m) * np.dot(A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dZ1 = np.dot(dZ2, W2.T) * relu_deriv(Z1)
        dW1 = (1/m) * np.dot(X_train.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # --- Update ---
        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2
        
        duration = time.time() - t0
        history['time_per_epoch'].append(duration)

        # --- Metrics & Test (tous les 10 epochs) ---
        if epoch % 10 == 0:
            # Train Metrics
            loss_train = -np.mean(np.sum(y_train_onehot * np.log(A2 + 1e-8), axis=1))
            pred_train = np.argmax(A2, axis=1)
            acc_train = np.mean(pred_train == y_train)

            # Test Forward (Inférence seule)
            Z1_test = np.dot(X_test, W1) + b1
            A1_test = relu(Z1_test)
            Z2_test = np.dot(A1_test, W2) + b2
            A2_test = softmax(Z2_test)

            # Test Metrics
            loss_test = -np.mean(np.sum(y_test_onehot * np.log(A2_test + 1e-8), axis=1))
            pred_test = np.argmax(A2_test, axis=1)
            acc_test = np.mean(pred_test == y_test)

            history['epochs'].append(epoch)
            history['train_loss'].append(loss_train)
            history['test_loss'].append(loss_test)
            history['train_acc'].append(acc_train)
            history['test_acc'].append(acc_test)

            print(f"CPU Epoch {epoch} | Tr Loss: {loss_train:.3f} Acc: {acc_train:.1%} | Te Loss: {loss_test:.3f} Acc: {acc_test:.1%}")

    total_time = time.time() - start_total
    print(f"CPU Terminé. Temps total: {total_time:.2f}s")
    
    # IMPORTANT : On ne renvoie que 2 valeurs ici pour correspondre au main.py
    return history, total_time