import os
import preprocessing
import train_cpu
import train_gpu
import comparison

# Paramètres
EPOCHS = 50
HIDDEN_SIZE = 256
LEARNING_RATE = 0.01

def main():
    # 1. Préparation (si nécessaire)
    if not os.path.exists("data_ready/X_train.npy"):
        print("Préparation des données...")
        success = preprocessing.prepare_dataset()
        if not success: return
    else:
        print("Données trouvées.")

    # 2. Train CPU
    cpu_hist, cpu_time = train_cpu.run_cpu_training(
        epochs=EPOCHS, 
        hidden_size=HIDDEN_SIZE, 
        lr=LEARNING_RATE
    )

    # 3. Train GPU
    gpu_hist, gpu_time = train_gpu.run_gpu_training(
        epochs=EPOCHS, 
        hidden_size=HIDDEN_SIZE, 
        lr=LEARNING_RATE
    )

    # 4. Comparaison
    comparison.compare_results(cpu_hist, cpu_time, gpu_hist, gpu_time)

if __name__ == "__main__":
    main()