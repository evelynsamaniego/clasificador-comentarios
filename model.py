import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk
from nltk.corpus import stopwords

# Descargar stopwords
nltk.download('stopwords')
spanish_stopwords = stopwords.words('spanish')

# Cargar dataset
df = pd.read_csv("comentarios_universidad.csv", quotechar='"')
df['etiqueta'] = df['etiqueta'].astype(int)

# Función de limpieza
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"@\w+", "", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

df["texto"] = df["texto"].apply(limpiar_texto)

# Vectorización
vectorizador = TfidfVectorizer(max_features=3000, stop_words=None, ngram_range=(1, 2))
X = vectorizador.fit_transform(df["texto"])
y = df["etiqueta"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Evaluación
y_pred = modelo.predict(X_test)
reporte = classification_report(y_test, y_pred, output_dict=True)
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
matriz = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", matriz)

# --- GRÁFICA MATRIZ DE CONFUSIÓN ---
plt.figure(figsize=(6,5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativo", "Positivo"], yticklabels=["Negativo", "Positivo"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión del Modelo")
plt.tight_layout()
plt.show()

# --- GRÁFICA DE MÉTRICAS POR CLASE ---
metricas = ['precision', 'recall', 'f1-score']
valores_clase_0 = [reporte['0'][m] for m in metricas]
valores_clase_1 = [reporte['1'][m] for m in metricas]

x = range(len(metricas))
plt.figure(figsize=(8, 5))
plt.bar(x, valores_clase_0, width=0.4, label='Negativo', align='center')
plt.bar([i + 0.4 for i in x], valores_clase_1, width=0.4, label='Positivo', align='center')
plt.xticks([i + 0.2 for i in x], metricas)
plt.ylim(0, 1)
plt.ylabel("Valor")
plt.title("Precisión, Recall y F1-score por Clase")
plt.legend()
plt.tight_layout()
plt.show()

# Guardar modelo y vectorizador
pickle.dump(modelo, open("modelo_sentimientos.pkl", "wb"))
pickle.dump(vectorizador, open("vectorizador.pkl", "wb"))
