import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Cargar el dataset zoo2
data_zoo2 = pd.read_csv("zoo2.csv")

# Separar características (features) y etiquetas (labels)
X_zoo2 = data_zoo2.drop(columns=['class_type'])  # características
y_zoo2 = data_zoo2['class_type']  # etiquetas

# Codificar las características categóricas utilizando one-hot encoding para zoo2
X_zoo2_encoded = pd.get_dummies(X_zoo2)

# Dividir los datos de zoo2 en conjuntos de entrenamiento y prueba
X_train_zoo2, X_test_zoo2, y_train_zoo2, y_test_zoo2 = train_test_split(X_zoo2_encoded, y_zoo2, test_size=0.2, random_state=42)

# 1. Regresión Logística para zoo2
logistic_model_zoo2 = LogisticRegression(max_iter=1000)
logistic_model_zoo2.fit(X_train_zoo2, y_train_zoo2)

# Predecir en el conjunto de prueba para zoo2
logistic_preds_zoo2 = logistic_model_zoo2.predict(X_test_zoo2)

# 2. K-Vecinos Cercanos para zoo2
knn_model_zoo2 = KNeighborsClassifier(n_neighbors=5)
knn_model_zoo2.fit(X_train_zoo2, y_train_zoo2)

# Predecir en el conjunto de prueba para zoo2
knn_preds_zoo2 = knn_model_zoo2.predict(X_test_zoo2)

# Función para calcular todas las métricas de evaluación
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_true, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Calcular métricas para Regresión Logística en zoo2
logistic_accuracy_zoo2, logistic_precision_zoo2, logistic_recall_zoo2, logistic_f1_zoo2 = calculate_metrics(y_test_zoo2, logistic_preds_zoo2)

# Calcular métricas para K-Vecinos Cercanos en zoo2
knn_accuracy_zoo2, knn_precision_zoo2, knn_recall_zoo2, knn_f1_zoo2 = calculate_metrics(y_test_zoo2, knn_preds_zoo2)

# Imprimir los resultados para zoo2
print("Resultados para zoo2:")
print("Métricas para Regresión Logística:")
print("Accuracy:", logistic_accuracy_zoo2)
print("Precision:", logistic_precision_zoo2)
print("Recall:", logistic_recall_zoo2)
print("F1 Score:", logistic_f1_zoo2)

print("\nMétricas para K-Vecinos Cercanos:")
print("Accuracy:", knn_accuracy_zoo2)
print("Precision:", knn_precision_zoo2)
print("Recall:", knn_recall_zoo2)
print("F1 Score:", knn_f1_zoo2)

# Cargar el dataset zoo3
data_zoo3 = pd.read_csv("zoo3.csv")

# Separar características (features) y etiquetas (labels)
X_zoo3 = data_zoo3.drop(columns=['class_type'])  # características
y_zoo3 = data_zoo3['class_type']  # etiquetas

# Codificar las características categóricas utilizando one-hot encoding para zoo3
X_zoo3_encoded = pd.get_dummies(X_zoo3)

# Dividir los datos de zoo3 en conjuntos de entrenamiento y prueba
X_train_zoo3, X_test_zoo3, y_train_zoo3, y_test_zoo3 = train_test_split(X_zoo3_encoded, y_zoo3, test_size=0.2, random_state=42)

# 1. Regresión Logística para zoo3
logistic_model_zoo3 = LogisticRegression(max_iter=1000)
logistic_model_zoo3.fit(X_train_zoo3, y_train_zoo3)

# Predecir en el conjunto de prueba para zoo3
logistic_preds_zoo3 = logistic_model_zoo3.predict(X_test_zoo3)

# 2. K-Vecinos Cercanos para zoo3
knn_model_zoo3 = KNeighborsClassifier(n_neighbors=5)
knn_model_zoo3.fit(X_train_zoo3, y_train_zoo3)

# Predecir en el conjunto de prueba para zoo3
knn_preds_zoo3 = knn_model_zoo3.predict(X_test_zoo3)

# Calcular métricas para Regresión Logística en zoo3
logistic_accuracy_zoo3, logistic_precision_zoo3, logistic_recall_zoo3, logistic_f1_zoo3 = calculate_metrics(y_test_zoo3, logistic_preds_zoo3)

# Calcular métricas para K-Vecinos Cercanos en zoo3
knn_accuracy_zoo3, knn_precision_zoo3, knn_recall_zoo3, knn_f1_zoo3 = calculate_metrics(y_test_zoo3, knn_preds_zoo3)

# Imprimir los resultados para zoo3
print("\nResultados para zoo3:")
print("Métricas para Regresión Logística:")
print("Accuracy:", logistic_accuracy_zoo3)
print("Precision:", logistic_precision_zoo3)
print("Recall:", logistic_recall_zoo3)
print("F1 Score:", logistic_f1_zoo3)

print("\nMétricas para K-Vecinos Cercanos:")
print("Accuracy:", knn_accuracy_zoo3)
print("Precision:", knn_precision_zoo3)
print("Recall:", knn_recall_zoo3)
print("F1 Score:", knn_f1_zoo3)

# Cargar el conjunto de datos Zoo
zoo_data = pd.read_csv('zoo3.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = zoo_data.drop(columns=['animal_name', 'class_type'])
y = zoo_data['class_type']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reducir la dimensionalidad de las características X a 2 dimensiones usando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Métricas para SVM
svm_model = SVC(kernel='linear')  # Selecciona el kernel deseado, por ejemplo 'linear'
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Confusion matrix para SVM
svm_conf_matrix = confusion_matrix(y_test, svm_predictions)

# Calculando Sensitivity y Specificity desde la matriz de confusión para SVM
svm_true_negatives = svm_conf_matrix[0,0]
svm_false_positives = svm_conf_matrix[0,1]
svm_false_negatives = svm_conf_matrix[1,0]
svm_true_positives = svm_conf_matrix[1,1]

svm_sensitivity = svm_true_positives / (svm_true_positives + svm_false_negatives)
svm_specificity = svm_true_negatives / (svm_true_negatives + svm_false_positives)

print("Metricas para SVM:")
print("Precision:", svm_precision)
print("Sensitivity:", svm_sensitivity)
print("Specificity:", svm_specificity)
print("F1 Score:", svm_f1)

#Métricas para Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions, average='weighted')
nb_recall = recall_score(y_test, nb_predictions, average='weighted')
nb_f1 = f1_score(y_test, nb_predictions, average='weighted')

# Confusion matrix para Naive Bayes
nb_conf_matrix = confusion_matrix(y_test, nb_predictions)

# Calculando Sensitivity y Specificity desde la matriz de confusión para Naive Bayes
nb_true_negatives = nb_conf_matrix[0,0]
nb_false_positives = nb_conf_matrix[0,1]
nb_false_negatives = nb_conf_matrix[1,0]
nb_true_positives = nb_conf_matrix[1,1]

nb_sensitivity = nb_true_positives / (nb_true_positives + nb_false_negatives)
nb_specificity = nb_true_negatives / (nb_true_negatives + nb_false_positives)

print("\nMetrics for Naive Bayes:")
print("Precision:", nb_precision)
print("Sensitivity:", nb_sensitivity)
print("Specificity:", nb_specificity)
print("F1 Score:", nb_f1)
print("Accuracy:", nb_accuracy)

# Graficar los datos después de reducir la dimensionalidad
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.title('PCA Zoo Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Class Type')
plt.show()