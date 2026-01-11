import os
import librosa
import numpy as np
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATASET_PATH = "genres_original"
IMG_SIZE = 64
INPUT_VECTOR_SIZE = IMG_SIZE * IMG_SIZE 
SR = 22050
DURATION = 30

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
        melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=IMG_SIZE)
        melspec_db = librosa.power_to_db(melspec, ref=np.max)
        melspec_resized = resize(melspec_db, (IMG_SIZE, IMG_SIZE), anti_aliasing=True)
        norm_spec = (melspec_resized - melspec_resized.min()) / (melspec_resized.max() - melspec_resized.min())
        return norm_spec.flatten()
    except Exception as e:
        print(f"Erreur: {e}")
        return None

def prepare_dataset():
    print("--- Démarrage du preprocessing ---")
    if not os.path.exists(DATASET_PATH):
        print(f"ERREUR: Le dossier '{DATASET_PATH}' n'existe pas.")
        return False

    inputs, labels = [], []
    genres = os.listdir(DATASET_PATH)
    
    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        if os.path.isdir(genre_path):
            print(f"Genre: {genre}...")
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):
                    data = extract_features(os.path.join(genre_path, filename))
                    if data is not None:
                        inputs.append(data)
                        labels.append(genre)

    X = np.array(inputs, dtype=np.float32)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    
    # MODIFICATION ICI : test_size=0.3 pour faire 70/30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    os.makedirs("data_ready", exist_ok=True)
    np.save("data_ready/X_train.npy", X_train)
    np.save("data_ready/X_test.npy", X_test)
    np.save("data_ready/y_train.npy", y_train)
    np.save("data_ready/y_test.npy", y_test)
    np.save("data_ready/classes.npy", encoder.classes_)
    
    print(f"Données sauvegardées (Train: {X_train.shape[0]}, Test: {X_test.shape[0]})")
    return True

if __name__ == "__main__":
    prepare_dataset()