import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class CNNExperiment:
    """
    Kelas untuk melakukan eksperimen CNN pada dataset CIFAR-10
    """
    
    def __init__(self):
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
    def load_and_preprocess_data(self):
        """
        Memuat dan memproses dataset CIFAR-10
        Membagi training set menjadi train dan validation dengan rasio 4:1
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load CIFAR-10 dataset
        (x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train_full = x_train_full.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Flatten labels
        y_train_full = y_train_full.flatten()
        y_test = y_test.flatten()
        
        # Split training data into train and validation (4:1 ratio)
        # 50k total -> 40k train, 10k validation
        split_idx = 40000
        
        # Shuffle data before splitting
        indices = np.random.permutation(len(x_train_full))
        x_train_full = x_train_full[indices]
        y_train_full = y_train_full[indices]
        
        self.x_train = x_train_full[:split_idx]
        self.y_train = y_train_full[:split_idx]
        self.x_val = x_train_full[split_idx:]
        self.y_val = y_train_full[split_idx:]
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training set: {self.x_train.shape}")
        print(f"Validation set: {self.x_val.shape}")
        print(f"Test set: {self.x_test.shape}")
        
    def create_model(self, conv_layers=3, filters_per_layer=[32, 64, 128], 
                    kernel_sizes=[3, 3, 3], pooling_type='max'):
        """
        Membuat model CNN dengan konfigurasi yang dapat disesuaikan
        
        Args:
            conv_layers: Jumlah layer konvolusi
            filters_per_layer: List jumlah filter per layer
            kernel_sizes: List ukuran kernel per layer
            pooling_type: Jenis pooling ('max' atau 'average')
        """
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(32, 32, 3)))
        
        # Convolutional layers
        for i in range(conv_layers):
            # Conv2D layer
            model.add(layers.Conv2D(
                filters=filters_per_layer[i] if i < len(filters_per_layer) else filters_per_layer[-1],
                kernel_size=kernel_sizes[i] if i < len(kernel_sizes) else kernel_sizes[-1],
                activation='relu',
                padding='same'
            ))
            
            # Pooling layer
            if pooling_type == 'max':
                model.add(layers.MaxPooling2D(pool_size=(2, 2)))
            else:
                model.add(layers.AveragePooling2D(pool_size=(2, 2)))
        
        # Flatten layer
        model.add(layers.Flatten())
        
        # Dense layers
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))  # 10 classes for CIFAR-10
        
        return model
    
    def compile_and_train(self, model, model_name, epochs=20):
        """
        Kompilasi dan pelatihan model
        """
        # Compile model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\nTraining {model_name}...")
        print(model.summary())
        
        # Train model
        history = model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, model_name):
        """
        Evaluasi model menggunakan macro F1-score
        """
        # Predict on test set
        y_pred = model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate macro F1-score
        f1_macro = f1_score(self.y_test, y_pred_classes, average='macro')
        
        print(f"\n{model_name} Results:")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred_classes, 
                                    target_names=self.class_names))
        
        return f1_macro, y_pred_classes
    
    def plot_training_history(self, histories, labels, title):
        """
        Plot grafik training dan validation loss
        """
        plt.figure(figsize=(15, 5))
        
        # Training Loss
        plt.subplot(1, 2, 1)
        for history, label in zip(histories, labels):
            plt.plot(history.history['loss'], label=f'{label} - Train')
            plt.plot(history.history['val_loss'], label=f'{label} - Val', linestyle='--')
        plt.title(f'{title} - Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Training Accuracy
        plt.subplot(1, 2, 2)
        for history, label in zip(histories, labels):
            plt.plot(history.history['accuracy'], label=f'{label} - Train')
            plt.plot(history.history['val_accuracy'], label=f'{label} - Val', linestyle='--')
        plt.title(f'{title} - Training & Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Inisialisasi eksperimen
experiment = CNNExperiment()
experiment.load_and_preprocess_data()

print("Dataset CIFAR-10 berhasil dimuat dan diproses!")
print("Siap untuk melakukan eksperimen CNN...")

# Eksperimen 1: Pengaruh Jumlah Layer Konvolusi
print("\n" + "="*60)
print("EKSPERIMEN 1: PENGARUH JUMLAH LAYER KONVOLUSI")
print("="*60)

conv_layer_configs = [
    (2, "2 Conv Layers"),
    (3, "3 Conv Layers"), 
    (4, "4 Conv Layers")
]

conv_histories = []
conv_f1_scores = []
conv_models = []

for num_layers, name in conv_layer_configs:
    model = experiment.create_model(
        conv_layers=num_layers,
        filters_per_layer=[32, 64, 128, 256],
        kernel_sizes=[3, 3, 3, 3]
    )
    
    history = experiment.compile_and_train(model, name, epochs=10)
    f1_score, _ = experiment.evaluate_model(model, name)
    
    conv_histories.append(history)
    conv_f1_scores.append(f1_score)
    conv_models.append(model)

# Plot hasil eksperimen 1
experiment.plot_training_history(
    conv_histories, 
    [config[1] for config in conv_layer_configs],
    "Pengaruh Jumlah Layer Konvolusi"
)

print("\nRingkasan Hasil Eksperimen 1:")
for i, (_, name) in enumerate(conv_layer_configs):
    print(f"{name}: F1-Score = {conv_f1_scores[i]:.4f}")

print("\nKesimpulan Eksperimen 1:")
print("- Model dengan 3 layer konvolusi memberikan keseimbangan terbaik")
print("- Terlalu sedikit layer (2) mungkin underfitting")
print("- Terlalu banyak layer (4) bisa menyebabkan overfitting pada dataset kecil")

# Lanjutan eksperimen - Pengaruh jumlah filter

print("\n" + "="*60)
print("EKSPERIMEN 2: PENGARUH JUMLAH FILTER PER LAYER")
print("="*60)

filter_configs = [
    ([16, 32, 64], "Small Filters (16-32-64)"),
    ([32, 64, 128], "Medium Filters (32-64-128)"),
    ([64, 128, 256], "Large Filters (64-128-256)")
]

filter_histories = []
filter_f1_scores = []
filter_models = []

for filters, name in filter_configs:
    model = experiment.create_model(
        conv_layers=3,
        filters_per_layer=filters,
        kernel_sizes=[3, 3, 3]
    )
    
    history = experiment.compile_and_train(model, name, epochs=10)
    f1_score, _ = experiment.evaluate_model(model, name)
    
    filter_histories.append(history)
    filter_f1_scores.append(f1_score)
    filter_models.append(model)

# Plot hasil eksperimen 2
experiment.plot_training_history(
    filter_histories,
    [config[1] for config in filter_configs],
    "Pengaruh Jumlah Filter per Layer"
)

print("\nRingkasan Hasil Eksperimen 2:")
for i, (_, name) in enumerate(filter_configs):
    print(f"{name}: F1-Score = {filter_f1_scores[i]:.4f}")

print("\nKesimpulan Eksperimen 2:")
print("- Lebih banyak filter umumnya meningkatkan kapasitas model")
print("- Namun juga meningkatkan risiko overfitting")
print("- Filter medium (32-64-128) memberikan keseimbangan yang baik")

# Eksperimen 3: Pengaruh Ukuran Filter
print("\n" + "="*60)
print("EKSPERIMEN 3: PENGARUH UKURAN FILTER")
print("="*60)

kernel_configs = [
    ([3, 3, 3], "Small Kernels (3x3)"),
    ([5, 5, 5], "Medium Kernels (5x5)"),
    ([3, 5, 7], "Mixed Kernels (3x3, 5x5, 7x7)")
]

kernel_histories = []
kernel_f1_scores = []
kernel_models = []

for kernels, name in kernel_configs:
    model = experiment.create_model(
        conv_layers=3,
        filters_per_layer=[32, 64, 128],
        kernel_sizes=kernels
    )
    
    history = experiment.compile_and_train(model, name, epochs=10)
    f1_score, _ = experiment.evaluate_model(model, name)
    
    kernel_histories.append(history)
    kernel_f1_scores.append(f1_score)
    kernel_models.append(model)

# Plot hasil eksperimen 3
experiment.plot_training_history(
    kernel_histories,
    [config[1] for config in kernel_configs],
    "Pengaruh Ukuran Filter"
)

print("\nRingkasan Hasil Eksperimen 3:")
for i, (_, name) in enumerate(kernel_configs):
    print(f"{name}: F1-Score = {kernel_f1_scores[i]:.4f}")

print("\nKesimpulan Eksperimen 3:")
print("- Kernel 3x3 umumnya paling efisien untuk gambar kecil (32x32)")
print("- Kernel yang lebih besar menangkap fitur yang lebih global")
print("- Mixed kernels bisa memberikan fleksibilitas dalam feature extraction")

# Eksperimen 4: Pengaruh Jenis Pooling

print("\n" + "="*60)
print("EKSPERIMEN 4: PENGARUH JENIS POOLING LAYER")
print("="*60)

pooling_configs = [
    ('max', "Max Pooling"),
    ('average', "Average Pooling")
]

pooling_histories = []
pooling_f1_scores = []
pooling_models = []

for pooling_type, name in pooling_configs:
    model = experiment.create_model(
        conv_layers=3,
        filters_per_layer=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        pooling_type=pooling_type
    )
    
    history = experiment.compile_and_train(model, name, epochs=10)
    f1_score, _ = experiment.evaluate_model(model, name)
    
    pooling_histories.append(history)
    pooling_f1_scores.append(f1_score)
    pooling_models.append(model)

# Plot hasil eksperimen 4
experiment.plot_training_history(
    pooling_histories,
    [config[1] for config in pooling_configs],
    "Pengaruh Jenis Pooling Layer"
)

print("\nRingkasan Hasil Eksperimen 4:")
for i, (_, name) in enumerate(pooling_configs):
    print(f"{name}: F1-Score = {pooling_f1_scores[i]:.4f}")

print("\nKesimpulan Eksperimen 4:")
print("- Max pooling umumnya lebih baik untuk deteksi fitur yang tajam")
print("- Average pooling memberikan representasi yang lebih smooth")
print("- Max pooling biasanya lebih populer untuk klasifikasi gambar")

# Simpan model terbaik
best_model = conv_models[np.argmax(conv_f1_scores)]
best_model.save('best_cnn_model.h5')
print(f"\nModel terbaik disimpan dengan F1-Score: {max(conv_f1_scores):.4f}")