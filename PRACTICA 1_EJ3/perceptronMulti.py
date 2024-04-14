import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def signo(x):
    return np.where(x > 0, 1, -1)

def funcion_activacion(x):
    return 1 / (1 + np.exp(-x))

class PerceptronMulticapa:
    def __init__(self, capas):
        self.pesos = []
        self.bias = []
        for i in range(len(capas) - 1):
            self.pesos.append(np.random.rand(capas[i], capas[i + 1]))
            self.bias.append(np.random.rand(capas[i + 1]))

    def alimentar_adelante(self, x):
        for i in range(len(self.pesos)):
            x = np.dot(x, self.pesos[i]) + self.bias[i]
            x = funcion_activacion(x)
        return x

    def retropropagacion(self, x, d, tasa_aprendizaje):
        for i in range(len(self.pesos) - 1, -1, -1):
            delta = d * funcion_activacion(x[i]) * (1 - funcion_activacion(x[i]))
            dw = np.dot(x[i].T, delta)
            self.pesos[i] -= tasa_aprendizaje * dw
            self.bias[i] -= tasa_aprendizaje * delta

            d = np.dot(delta, self.pesos[i].T)

    def entrenar(self, X, d, tasa_aprendizaje, epocas):
        for _ in range(epocas):
            for i in range(len(X)):
                x = X[i]
                y = self.alimentar_adelante(x)
                
                error = y - d[i]
                self.retropropagacion(x, error, tasa_aprendizaje)

    def clasificar(self, X):
        y = []
        for i in range(len(X)):
            x = X[i]
            y.append(np.argmax(self.alimentar_adelante(x)))
        return y

# Lectura y preprocesamiento del dataset `concentlite.csv`
data = pd.read_csv('concentlite.csv')
X = data[['x1', 'x2']].to_numpy()
y = data['y'].to_numpy()

# Normalización de las características
X = (X - X.min()) / (X.max() - X.min())

# Entrenamiento del perceptrón multicapa
capas = [2, 5, 2]  # Ejemplo: 2 neuronas en la entrada, 5 en la capa oculta y 2 en la salida
red = PerceptronMulticapa(capas)
tasa_aprendizaje = 0.01
epocas = 1000
red.entrenar(X, y, tasa_aprendizaje, epocas)

# Clasificación de nuevos datos
X_nuevo = np.array([[0.1, 0.2], [0.8, 0.9]])
y_nuevo = red.clasificar(X_nuevo)
print("Clases de los nuevos datos:", y_nuevo)

# Visualización gráfica
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')  # Colores para cada clase
plt.scatter(X_nuevo[:, 0], X_nuevo[:, 1], marker='^', c=y_nuevo, s=100)  # Marcadores para nuevos datos
plt.title('Clasificación con perceptrón multicapa')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()
