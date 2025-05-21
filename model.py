import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import nltk
from nltk.corpus import stopwords

# Descargar stopwords si no están descargadas
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

# Vectorización con stopwords en español
# Vectorización con unigramas y bigramas
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
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

# Guardar modelo y vectorizador
pickle.dump(modelo, open("modelo_sentimientos.pkl", "wb"))
pickle.dump(vectorizador, open("vectorizador.pkl", "wb"))
