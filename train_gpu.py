import numpy as np
from numba import cuda, float32
import time
from sklearn.preprocessing import OneHotEncoder

TPB = 16

# --- KERNELS CUDA (Identiques) ---
@cuda.jit
def matmul_tiled(A, B, C):
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    tx, ty = cuda.threadIdx.x, cuda.threadIdx.y
    row = cuda.blockIdx.x * TPB + tx
    col = cuda.blockIdx.y * TPB + ty
    tmp = 0.0
    num_tiles = (A.shape[1] + TPB - 1) // TPB
    for t in range(num_tiles):
        if row < A.shape[0] and (t * TPB + ty) < A.shape[1]:
            sA[tx, ty] = A[row, t * TPB + ty]
        else: sA[tx, ty] = 0.0
        if col < B.shape[1] and (t * TPB + tx) < B.shape[0]:
            sB[tx, ty] = B[t * TPB + tx, col]
        else: sB[tx, ty] = 0.0
        cuda.syncthreads()
        for k in range(TPB): tmp += sA[tx, k] * sB[k, ty]
        cuda.syncthreads()
    if row < C.shape[0] and col < C.shape[1]: C[row, col] = tmp

@cuda.jit
def transpose_tiled(A, A_T):
    tile = cuda.shared.array((TPB, TPB), dtype=float32)
    x = cuda.blockIdx.x * TPB + cuda.threadIdx.x
    y = cuda.blockIdx.y * TPB + cuda.threadIdx.y
    if y < A.shape[0] and x < A.shape[1]:
        tile[cuda.threadIdx.y, cuda.threadIdx.x] = A[y, x]
    cuda.syncthreads()
    x_out = cuda.blockIdx.y * TPB + cuda.threadIdx.x
    y_out = cuda.blockIdx.x * TPB + cuda.threadIdx.y
    if y_out < A_T.shape[0] and x_out < A_T.shape[1]:
        A_T[y_out, x_out] = tile[cuda.threadIdx.x, cuda.threadIdx.y]

@cuda.jit
def relu_forward(Z):
    r, c = cuda.grid(2)
    if r < Z.shape[0] and c < Z.shape[1]:
        if Z[r, c] < 0: Z[r, c] = 0

@cuda.jit
def relu_backward(dZ, Z):
    r, c = cuda.grid(2)
    if r < dZ.shape[0] and c < dZ.shape[1]:
        if Z[r, c] <= 0: dZ[r, c] = 0

@cuda.jit
def update_params(W, dW, lr):
    r, c = cuda.grid(2)
    if r < W.shape[0] and c < W.shape[1]:
        W[r, c] = W[r, c] - lr * dW[r, c]

# --- HELPER ---
def get_grid(rows, cols):
    bx = (rows + TPB - 1) // TPB
    by = (cols + TPB - 1) // TPB
    return (bx, by), (TPB, TPB)

def softmax_cpu(Z_host):
    e_Z = np.exp(Z_host - np.max(Z_host, axis=1, keepdims=True))
    return e_Z / np.sum(e_Z, axis=1, keepdims=True)

# --- TRAIN FUNCTION ---
def run_gpu_training(epochs=50, hidden_size=256, lr=0.01):
    print("\n--- ENTRAÎNEMENT GPU (Numba Custom Kernels) ---")
    
    # 1. Load Data
    X_train = np.load("data_ready/X_train.npy")
    y_train = np.load("data_ready/y_train.npy")
    X_test = np.load("data_ready/X_test.npy")
    y_test = np.load("data_ready/y_test.npy")

    # On multiplie la taille du dataset par 50 !
    X_train = np.tile(X_train, (10, 1))
    y_train = np.tile(y_train, (10,))
    
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1)).astype(np.float32)
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1)).astype(np.float32)
    
    N, input_size = X_train.shape
    N_test = X_test.shape[0]
    output_size = y_train_onehot.shape[1]
    
    # 2. Init & Alloc
    W1 = np.random.randn(input_size, hidden_size).astype(np.float32) * np.sqrt(2/input_size)
    W2 = np.random.randn(hidden_size, output_size).astype(np.float32) * np.sqrt(2/hidden_size)

    d_X = cuda.to_device(X_train)
    d_X_test = cuda.to_device(X_test) # Alloc Test
    d_W1 = cuda.to_device(W1)
    d_W2 = cuda.to_device(W2)
    
    # Train Buffers
    d_Z1 = cuda.device_array((N, hidden_size), dtype=np.float32)
    d_Z2 = cuda.device_array((N, output_size), dtype=np.float32)
    d_dZ2 = cuda.device_array((N, output_size), dtype=np.float32)
    d_dZ1 = cuda.device_array((N, hidden_size), dtype=np.float32)
    d_dW2 = cuda.device_array((hidden_size, output_size), dtype=np.float32)
    d_dW1 = cuda.device_array((input_size, hidden_size), dtype=np.float32)
    
    # Transpose Buffers
    d_X_T = cuda.device_array((input_size, N), dtype=np.float32)
    d_Z1_T = cuda.device_array((hidden_size, N), dtype=np.float32)
    d_W2_T = cuda.device_array((output_size, hidden_size), dtype=np.float32)

    # Test Buffers
    d_Z1_test = cuda.device_array((N_test, hidden_size), dtype=np.float32)
    d_Z2_test = cuda.device_array((N_test, output_size), dtype=np.float32)

    # Pre-calc Transpose X
    g_XT, b_XT = get_grid(X_train.shape[1], X_train.shape[0])
    transpose_tiled[g_XT, b_XT](d_X, d_X_T)
    cuda.synchronize()

    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'epochs': [],
        'time_per_epoch': []
    }
    
    start_total = time.time()

    for epoch in range(epochs + 1):
        t0 = time.time()
        
        # --- A. FORWARD TRAIN ---
        g1, b1 = get_grid(N, hidden_size)
        matmul_tiled[g1, b1](d_X, d_W1, d_Z1)
        cuda.synchronize()
        relu_forward[g1, b1](d_Z1)
        cuda.synchronize()
        g2, b2 = get_grid(N, output_size)
        matmul_tiled[g2, b2](d_Z1, d_W2, d_Z2)
        cuda.synchronize()

        # --- B. LOSS TRAIN (Hybrid) ---
        Z2_host = d_Z2.copy_to_host()
        A2_host = softmax_cpu(Z2_host)
        dZ2_host = (A2_host - y_train_onehot) / N
        d_dZ2 = cuda.to_device(dZ2_host)
        
        # --- C. BACKWARD (Full GPU) ---
        g_Z1T, b_Z1T = get_grid(hidden_size, N)
        transpose_tiled[g_Z1T, b_Z1T](d_Z1, d_Z1_T)
        cuda.synchronize()
        g_dW2, b_dW2 = get_grid(hidden_size, output_size)
        matmul_tiled[g_dW2, b_dW2](d_Z1_T, d_dZ2, d_dW2)
        cuda.synchronize()
        
        g_W2T, b_W2T = get_grid(output_size, hidden_size)
        transpose_tiled[g_W2T, b_W2T](d_W2, d_W2_T)
        cuda.synchronize()
        matmul_tiled[g1, b1](d_dZ2, d_W2_T, d_dZ1)
        cuda.synchronize()
        relu_backward[g1, b1](d_dZ1, d_Z1)
        cuda.synchronize()
        
        g_dW1, b_dW1 = get_grid(input_size, hidden_size)
        matmul_tiled[g_dW1, b_dW1](d_X_T, d_dZ1, d_dW1)
        cuda.synchronize()
        
        # --- D. UPDATE ---
        update_params[g_dW1, b_dW1](d_W1, d_dW1, lr)
        update_params[g_dW2, b_dW2](d_W2, d_dW2, lr)
        cuda.synchronize()
        
        duration = time.time() - t0
        history['time_per_epoch'].append(duration)
        
        # --- METRICS & TEST ---
        if epoch % 10 == 0:
            # Train Stats
            loss_train = -np.mean(np.sum(y_train_onehot * np.log(A2_host + 1e-8), axis=1))
            acc_train = np.mean(np.argmax(A2_host, axis=1) == y_train)
            
            # Test Forward (GPU)
            g1_t, b1_t = get_grid(N_test, hidden_size)
            matmul_tiled[g1_t, b1_t](d_X_test, d_W1, d_Z1_test)
            cuda.synchronize()
            relu_forward[g1_t, b1_t](d_Z1_test)
            cuda.synchronize()
            g2_t, b2_t = get_grid(N_test, output_size)
            matmul_tiled[g2_t, b2_t](d_Z1_test, d_W2, d_Z2_test)
            cuda.synchronize()
            
            # Test Stats (CPU calculation)
            Z2_test_host = d_Z2_test.copy_to_host()
            A2_test_host = softmax_cpu(Z2_test_host)
            loss_test = -np.mean(np.sum(y_test_onehot * np.log(A2_test_host + 1e-8), axis=1))
            acc_test = np.mean(np.argmax(A2_test_host, axis=1) == y_test)

            history['epochs'].append(epoch)
            history['train_loss'].append(loss_train)
            history['test_loss'].append(loss_test)
            history['train_acc'].append(acc_train)
            history['test_acc'].append(acc_test)
            
            print(f"GPU Epoch {epoch} | Tr Loss: {loss_train:.3f} Acc: {acc_train:.1%} | Te Loss: {loss_test:.3f} Acc: {acc_test:.1%}")
            
    total_time = time.time() - start_total
    print(f"GPU Terminé. Temps total: {total_time:.2f}s")
    
    return history, total_time