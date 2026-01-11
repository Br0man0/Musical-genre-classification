import matplotlib.pyplot as plt
import numpy as np

def compare_results(cpu_hist, cpu_time, gpu_hist, gpu_time):
    # --- CALCULS STATISTIQUES ---
    # On ignore souvent la première époque pour la moyenne car elle inclut le temps de compilation JIT
    cpu_avg_epoch = np.mean(cpu_hist['time_per_epoch'][1:]) 
    gpu_avg_epoch = np.mean(gpu_hist['time_per_epoch'][1:])
    
    speedup_total = cpu_time / gpu_time
    speedup_epoch = cpu_avg_epoch / gpu_avg_epoch

    print("\n=================================================")
    print("           RAPPORT D'ANALYSE TEMPORELLE          ")
    print("=================================================")
    
    print(f"\n1. TEMPS TOTAL D'ENTRAÎNEMENT (Train + Test)")
    print(f"   - CPU (NumPy) : {cpu_time:.2f} secondes")
    print(f"   - GPU (CUDA)  : {gpu_time:.2f} secondes")
    print(f"   => ACCÉLÉRATION GLOBALE : x{speedup_total:.2f}")

    print(f"\n2. TEMPS MOYEN PAR EPOCH (hors compilation)")
    print(f"   - CPU (NumPy) : {cpu_avg_epoch:.4f} s/epoch")
    print(f"   - GPU (CUDA)  : {gpu_avg_epoch:.4f} s/epoch")
    print(f"   => ACCÉLÉRATION PAR EPOCH : x{speedup_epoch:.2f}")
    
    print("\n=================================================")

    epochs = cpu_hist['epochs']
    
    # --- FIGURE 1 : METRIQUES D'APPRENTISSAGE ---
    fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig1.suptitle('Performances Modèle: CPU vs GPU', fontsize=16)

    # 1. Loss CPU
    axs[0, 0].plot(epochs, cpu_hist['train_loss'], label='Train Loss', color='blue')
    axs[0, 0].plot(epochs, cpu_hist['test_loss'], label='Test Loss', color='cyan', linestyle='--')
    axs[0, 0].set_title("Loss CPU")
    axs[0, 0].set_ylabel("Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Loss GPU
    axs[0, 1].plot(epochs, gpu_hist['train_loss'], label='Train Loss', color='red')
    axs[0, 1].plot(epochs, gpu_hist['test_loss'], label='Test Loss', color='orange', linestyle='--')
    axs[0, 1].set_title("Loss GPU")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # 3. Accuracy CPU
    axs[1, 0].plot(epochs, cpu_hist['train_acc'], label='Train Acc', color='blue')
    axs[1, 0].plot(epochs, cpu_hist['test_acc'], label='Test Acc', color='cyan', linestyle='--')
    axs[1, 0].set_title("Accuracy CPU")
    axs[1, 0].set_ylabel("Accuracy")
    axs[1, 0].set_xlabel("Epoques")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # 4. Accuracy GPU
    axs[1, 1].plot(epochs, gpu_hist['train_acc'], label='Train Acc', color='red')
    axs[1, 1].plot(epochs, gpu_hist['test_acc'], label='Test Acc', color='orange', linestyle='--')
    axs[1, 1].set_title("Accuracy GPU")
    axs[1, 1].set_xlabel("Epoques")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("metrics_comparison.png")
    print("Graphique métriques sauvegardé : 'metrics_comparison.png'")

    # --- FIGURE 2 : ANALYSE TEMPORELLE ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Analyse de Performance Temporelle', fontsize=16)
    
    labels = ['CPU', 'GPU']
    colors = ['blue', 'red']

    # Graphique Temps Total
    times_total = [cpu_time, gpu_time]
    bars1 = ax1.bar(labels, times_total, color=colors, alpha=0.7)
    ax1.set_title('Temps Total Entraînement (s)')
    ax1.set_ylabel('Secondes (moins c\'est mieux)')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Ajout des valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    # Graphique Temps par Epoch
    times_epoch = [cpu_avg_epoch, gpu_avg_epoch]
    bars2 = ax2.bar(labels, times_epoch, color=colors, alpha=0.7)
    ax2.set_title('Temps Moyen par Epoch (s)')
    ax2.set_ylabel('Secondes (moins c\'est mieux)')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajout des valeurs sur les barres
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("time_comparison.png")
    print("Graphique temps sauvegardé : 'time_comparison.png'")
    
    plt.show()

if __name__ == "__main__":
    compare_results()