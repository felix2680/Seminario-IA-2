import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Función para calcular todas las métricas de evaluación
def calcular_metricas(y_verdadero, y_predicho):
    accuracy = accuracy_score(y_verdadero, y_predicho)
    precision = precision_score(y_verdadero, y_predicho, average='macro', zero_division=1)
    recall = recall_score(y_verdadero, y_predicho, average='macro', zero_division=1)
    f1 = f1_score(y_verdadero, y_predicho, average='macro')
    return accuracy, precision, recall, f1

# Función para entrenar y evaluar los modelos
def entrenar_y_evaluar_modelos(X_entrenamiento, X_prueba, y_entrenamiento, y_prueba):
    resultados = {}

    # Regresión Logística
    modelo_logistico = LogisticRegression(max_iter=1000)
    modelo_logistico.fit(X_entrenamiento, y_entrenamiento)
    predicciones_logistico = modelo_logistico.predict(X_prueba)
    resultados['logistico'] = calcular_metricas(y_prueba, predicciones_logistico)

    # K-Vecinos Cercanos
    modelo_knn = KNeighborsClassifier(n_neighbors=5)
    modelo_knn.fit(X_entrenamiento, y_entrenamiento)
    predicciones_knn = modelo_knn.predict(X_prueba)
    resultados['knn'] = calcular_metricas(y_prueba, predicciones_knn)

    # SVM
    modelo_svm = SVC(kernel='linear')
    modelo_svm.fit(X_entrenamiento, y_entrenamiento)
    predicciones_svm = modelo_svm.predict(X_prueba)
    resultados['svm'] = calcular_metricas(y_prueba, predicciones_svm)

    # Naive Bayes
    modelo_nb = GaussianNB()
    modelo_nb.fit(X_entrenamiento, y_entrenamiento)
    predicciones_nb = modelo_nb.predict(X_prueba)
    resultados['nb'] = calcular_metricas(y_prueba, predicciones_nb)

    return resultados

# Función para cargar datos, codificar, dividir y entrenar
def procesar_dataset(ruta_archivo):
    datos = pd.read_csv(ruta_archivo)
    X = datos.drop(columns=['class_type'])
    y = datos['class_type']
    X_codificado = pd.get_dummies(X)
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X_codificado, y, test_size=0.2, random_state=42)
    return X_entrenamiento, X_prueba, y_entrenamiento, y_prueba, X_codificado, y

# Función para realizar PCA y graficar
def graficar_pca(X_codificado, y, titulo):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_codificado)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
    plt.title(titulo)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Tipo de Clase')
    plt.show()

# Procesar y evaluar zoo.csv
X_entrenamiento_zoo, X_prueba_zoo, y_entrenamiento_zoo, y_prueba_zoo, X_codificado_zoo, y_zoo = procesar_dataset('zoo.csv')
resultados_zoo = entrenar_y_evaluar_modelos(X_entrenamiento_zoo, X_prueba_zoo, y_entrenamiento_zoo, y_prueba_zoo)

# Procesar y evaluar zoo2.csv
X_entrenamiento_zoo2, X_prueba_zoo2, y_entrenamiento_zoo2, y_prueba_zoo2, X_codificado_zoo2, y_zoo2 = procesar_dataset('zoo2.csv')
resultados_zoo2 = entrenar_y_evaluar_modelos(X_entrenamiento_zoo2, X_prueba_zoo2, y_entrenamiento_zoo2, y_prueba_zoo2)

# Procesar y evaluar zoo3.csv
X_entrenamiento_zoo3, X_prueba_zoo3, y_entrenamiento_zoo3, y_prueba_zoo3, X_codificado_zoo3, y_zoo3 = procesar_dataset('zoo3.csv')
resultados_zoo3 = entrenar_y_evaluar_modelos(X_entrenamiento_zoo3, X_prueba_zoo3, y_entrenamiento_zoo3, y_prueba_zoo3)

# Imprimir resultados
print("Resultados para zoo:")
print("Métricas para Regresión Logística:")
print("Accuracy:", resultados_zoo['logistico'][0])
print("Precision:", resultados_zoo['logistico'][1])
print("Recall:", resultados_zoo['logistico'][2])
print("F1 Score:", resultados_zoo['logistico'][3])

print("\nMétricas para K-Vecinos Cercanos:")
print("Accuracy:", resultados_zoo['knn'][0])
print("Precision:", resultados_zoo['knn'][1])
print("Recall:", resultados_zoo['knn'][2])
print("F1 Score:", resultados_zoo['knn'][3])

print("\nMétricas para SVM:")
print("Accuracy:", resultados_zoo['svm'][0])
print("Precision:", resultados_zoo['svm'][1])
print("Recall:", resultados_zoo['svm'][2])
print("F1 Score:", resultados_zoo['svm'][3])

print("\nMétricas para Naive Bayes:")
print("Accuracy:", resultados_zoo['nb'][0])
print("Precision:", resultados_zoo['nb'][1])
print("Recall:", resultados_zoo['nb'][2])
print("F1 Score:", resultados_zoo['nb'][3])

print("\nResultados para zoo2:")
print("Métricas para Regresión Logística:")
print("Accuracy:", resultados_zoo2['logistico'][0])
print("Precision:", resultados_zoo2['logistico'][1])
print("Recall:", resultados_zoo2['logistico'][2])
print("F1 Score:", resultados_zoo2['logistico'][3])

print("\nMétricas para K-Vecinos Cercanos:")
print("Accuracy:", resultados_zoo2['knn'][0])
print("Precision:", resultados_zoo2['knn'][1])
print("Recall:", resultados_zoo2['knn'][2])
print("F1 Score:", resultados_zoo2['knn'][3])

print("\nMétricas para SVM:")
print("Accuracy:", resultados_zoo2['svm'][0])
print("Precision:", resultados_zoo2['svm'][1])
print("Recall:", resultados_zoo2['svm'][2])
print("F1 Score:", resultados_zoo2['svm'][3])

print("\nMétricas para Naive Bayes:")
print("Accuracy:", resultados_zoo2['nb'][0])
print("Precision:", resultados_zoo2['nb'][1])
print("Recall:", resultados_zoo2['nb'][2])
print("F1 Score:", resultados_zoo2['nb'][3])

print("\nResultados para zoo3:")
print("Métricas para Regresión Logística:")
print("Accuracy:", resultados_zoo3['logistico'][0])
print("Precision:", resultados_zoo3['logistico'][1])
print("Recall:", resultados_zoo3['logistico'][2])
print("F1 Score:", resultados_zoo3['logistico'][3])

print("\nMétricas para K-Vecinos Cercanos:")
print("Accuracy:", resultados_zoo3['knn'][0])
print("Precision:", resultados_zoo3['knn'][1])
print("Recall:", resultados_zoo3['knn'][2])
print("F1 Score:", resultados_zoo3['knn'][3])

print("\nMétricas para SVM:")
print("Accuracy:", resultados_zoo3['svm'][0])
print("Precision:", resultados_zoo3['svm'][1])
print("Recall:", resultados_zoo3['svm'][2])
print("F1 Score:", resultados_zoo3['svm'][3])

print("\nMétricas para Naive Bayes:")
print("Accuracy:", resultados_zoo3['nb'][0])
print("Precision:", resultados_zoo3['nb'][1])
print("Recall:", resultados_zoo3['nb'][2])
print("F1 Score:", resultados_zoo3['nb'][3])

# Graficar PCA para cada dataset
graficar_pca(X_codificado_zoo, y_zoo, 'PCA Zoo Data')
graficar_pca(X_codificado_zoo2, y_zoo2, 'PCA Zoo2 Data')
graficar_pca(X_codificado_zoo3, y_zoo3, 'PCA Zoo3 Data')