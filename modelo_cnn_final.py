
# Importar librerías básicas para manejar arrays y gráficos
import numpy as np
import matplotlib.pyplot as plt
import torch                      # Motor principal de PyTorch
import torch.nn as nn             # Para definir redes neuronales
import torch.optim as optim       # Optimizadores
from torchvision import datasets, transforms  # Para MNIST y transformaciones
from torch.utils.data import DataLoader       # Carga eficiente de datos
from torch.utils.data import random_split #Divide de manera aleatoria el dataset de PyTorch, usando datos que no se han visto en el entrenamiento
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch.nn as nn
import torch.nn.functional as F

# Transformaciones: convertir a tensor y normalizar a [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # media y std para MNIST
])

# Cargar conjuntos de entrenamiento y prueba
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# DataLoaders para manejar por lotes
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Capa convolucional: 1 canal entrada, 32 filtros, tamaño 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        # Capa de pooling 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Capa densa intermedia: flatten primero, luego 128 neuronas
        self.fc1 = nn.Linear(32 * 13 * 13, 128)  # 28x28 -> 26x26 (conv) -> 13x13 (pool)
        # Capa de salida: 10 clases
        self.fc2 = nn.Linear(128, 10)
        # Activación ReLU
        self.relu = nn.ReLU()

    def forward(self, x):
        # Paso hacia adelante
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 32 * 13 * 13)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
  
  # BÚSQUEDA DE HIPERPARÁMETROS (Grid Search)

# 1. Volver a cargar el dataset completo con transformaciones (sin TestLoader)
from torchvision import datasets
full_dataset = datasets.MNIST(root='./data',
                             train=True,
                             download=True,
                             transform=transform)

# 2. Dividir en entrenamiento (90%) y validación (10%)
train_size = int(0.9 * len(full_dataset))  # 90% de muestras
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 3. Definir malla de hiperparámetros a probar
learning_rates = [0.001, 0.005, 0.01]  # posibles tasas de aprendizaje
batch_sizes    = [32, 64]             # posibles tamaños de lote
epoch_list     = [5, 10]              # posibles números de épocas

# 4. Variables para guardar el mejor resultado
best_val_acc = 0.0        # Mejor precisión de validación
best_params  = {}         # Parámetros que la produjeron
results      = []         # Lista con todos los resultados

# 5. Grid Search: probar cada combinación
for lr in learning_rates:                # iterar sobre tasas de aprendizaje
    for bs in batch_sizes:               # iterar sobre batch sizes
        for num_epochs in epoch_list:    # iterar sobre números de épocas

            # 5.1 Crear DataLoaders con el batch size actual
            from torch.utils.data import DataLoader
            train_loader_gs = DataLoader(train_dataset,
                                         batch_size=bs,
                                         shuffle=True)
            val_loader_gs   = DataLoader(val_dataset,
                                         batch_size=bs,
                                         shuffle=False)

            # 5.2 Instanciar un modelo fresco y moverlo a GPU/CPU
            model_gs = CNN().to(device)         # CNN definida
            criterion = nn.CrossEntropyLoss()   # Función de pérdida
            optimizer = optim.Adam(model_gs.parameters(),
                                   lr=lr)        # Optimizer con lr actual

            # 5.3 Entrenamiento rápido con num_epochs
            for epoch in range(num_epochs):
                model_gs.train()               # Modo entrenamiento
                for images, labels in train_loader_gs:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()     # Borrar gradientes previos
                    outputs = model_gs(images) # Forward pass
                    loss = criterion(outputs, labels)  # Calcular pérdida
                    loss.backward()           # Backpropagation
                    optimizer.step()          # Actualizar pesos

            # 5.4 Evaluación en validación
            model_gs.eval()                  # Modo evaluación
            correct, total = 0, 0
            with torch.no_grad():           # Sin calcular gradientes
                for images, labels in val_loader_gs:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model_gs(images)
                    _, preds = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (preds == labels).sum().item()
            val_acc = 100 * correct / total  # Porcentaje de aciertos

            # 5.5 Guardar resultado y, si es el mejor, actualizar best_*
            results.append({
                'lr': lr,
                'batch_size': bs,
                'epochs': num_epochs,
                'val_acc': val_acc
            })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params  = {'lr': lr, 'batch_size': bs, 'epochs': num_epochs}

# 6. Mostrar por pantalla la mejor combinación
print(f"Mejor precisión en validación: {best_val_acc:.2f}%")
print("Parámetros óptimos:", best_params)

transform = transforms.Compose([
    transforms.ToTensor(),                  # Convierte PIL → Tensor con valores [0,1]
    transforms.Normalize((0.5,), (0.5,))    # Normaliza a media 0.5 y std 0.5
])

# Descarga y carga de MNIST
train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               download=True,
                               transform=transform)
test_dataset  = datasets.MNIST(root='./data',
                               train=False,
                               download=True,
                               transform=transform)

# DataLoader para iterar en batches de 64
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Capa convolucional 1: 1→32 filtros, kernel 3x3, padding 1 para mantener tamaño espacial
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Capa convolucional 2: 32→64 filtros, kernel 3x3, padding 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Capa de pooling 2x2 para reducir tamaño a la mitad
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout tras el pooling (25% de nodos inactivos)
        self.dropout1 = nn.Dropout(0.25)
        # Capa totalmente conectada (fully connected):
        # After 2 conv + 1 pool: from 28x28 → 14x14 → canales=64
        # Así que tamaño de flatten = 64 * 14 * 14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        # Segundo dropout antes de la salida (50% de nodos inactivos)
        self.dropout2 = nn.Dropout(0.5)
        # Capa de salida: 128→10 clases
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Primer bloque: Conv → ReLU
        x = F.relu(self.conv1(x))
        # Segundo bloque: Conv → ReLU
        x = F.relu(self.conv2(x))
        # Pooling para reducir dimensión espacial a la mitad
        x = self.pool(x)
        # Dropout tras pooling
        x = self.dropout1(x)
        # Aplanar tensor para conectar con capa densa
        x = x.view(-1, 64 * 14 * 14)
        # Capa densa + ReLU
        x = F.relu(self.fc1(x))
        # Segundo Dropout
        x = self.dropout2(x)
        # Capa de salida (logits)
        x = self.fc2(x)
        return x

# 4. Configuración de dispositivo, modelo, pérdida y optimizador

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)                             # Mover modelo a GPU/CPU
criterion = nn.CrossEntropyLoss()                    # Pérdida para multiclase
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam con lr óptimo

# 5. Entrenamiento final (10 épocas)

train_losses = []      # Lista para pérdida por época
train_accuracies = []  # Lista para precisión por época

for epoch in range(10):      # Usamos epochs = 10
    model.train()            # Modo entrenamiento
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()            # Resetear gradientes
        outputs = model(images)          # Forward pass
        loss = criterion(outputs, labels)# Calcular pérdida
        loss.backward()                  # Backpropagation
        optimizer.step()                 # Actualizar pesos

        running_loss += loss.item()      # Acumular pérdida
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calcular métricas de la época
    epoch_loss = running_loss / len(train_loader)
    epoch_acc  = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"Época {epoch+1}: pérdida={epoch_loss:.4f}, precisión={epoch_acc:.2f}%")

# 6. Evaluación en test
model.eval()   # Modo evaluación
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en test: {100 * correct / total:.2f}%")

# 7. Graficar pérdida y precisión por época

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Pérdida', color = 'yellow')
plt.title('Pérdida de entrenamiento por época')
plt.xlabel('Época'); plt.ylabel('Pérdida')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Precisión (%)', color='purple')
plt.title('Precisión de entrenamiento por época')
plt.xlabel('Época'); plt.ylabel('Precisión (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Exportar modelo entrenado
torch.save(model.state_dict(), 'modelo_cnn_final.pth')
