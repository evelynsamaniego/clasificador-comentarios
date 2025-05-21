import pickle
from flask import Flask, render_template, request

# Cargar modelo y vectorizador
modelo = pickle.load(open("modelo_sentimientos.pkl", "rb"))
vectorizador = pickle.load(open("vectorizador.pkl", "rb"))

# App Flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    resultado = None
    clase = None

    if request.method == "POST":
        comentario = request.form["comentario"]
        comentario_vector = vectorizador.transform([comentario])
        prediccion = modelo.predict(comentario_vector)[0]

        if prediccion == 1:
            resultado = "Comentario positivo 😊"
            clase = "positivo"
        else:
            resultado = "Comentario negativo 😞"
            clase = "negativo"

    return render_template("index.html", resultado=resultado, clase=clase)

if __name__ == "__main__":
    app.run(debug=True)
