from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

# Cargar modelo y vectorizador
modelo = pickle.load(open("modelo_sentimientos.pkl", "rb"))
vectorizador = pickle.load(open("vectorizador.pkl", "rb"))

# App Flask
app = Flask(__name__)
CORS(app)  # <--- Aquí habilitas CORS para toda la app

@app.route("/")
def home():
    return "API Clasificador de Comentarios Activa"

@app.route("/api/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    comentario = data.get("comentario", "")
    
    if not comentario:
        return jsonify({"error": "Comentario vacío"}), 400

    comentario_vector = vectorizador.transform([comentario])
    prediccion = modelo.predict(comentario_vector)[0]
    resultado = "positivo" if prediccion == 1 else "negativo"

    return jsonify({"resultado": resultado})

if __name__ == "__main__":
    app.run(debug=True)
