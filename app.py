from datetime import date
from pathlib import Path
import math

import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

# Carreguei a Pipeline uma vez, ao iniciar a aplicação.
app = Flask(__name__)
app.json.ensure_ascii = False
modelo = joblib.load(Path(__file__).resolve().parent / "modelo.pkl")
atributos = list(modelo.feature_names_in_)
categoricos = ["type_of_meal_plan", "room_type_reserved", "market_segment_type"]
numericos = [campo for campo in atributos if campo not in categoricos]
rotulos = {0: "Not_Canceled", 1: "Canceled"}


# Criei uma página com formulário para testar a API pelo navegador.
@app.get("/")
def inicio():
    return render_template("index.html")


# Conferi os campos, tipos e regras básicas antes de enviar a reserva ao modelo.
def validar_reserva(reserva):
    if not isinstance(reserva, dict):
        raise ValueError("Envie um objeto JSON com os atributos de uma reserva.")
    faltantes = sorted(set(atributos) - set(reserva))
    extras = sorted(set(reserva) - set(atributos))
    if faltantes:
        raise ValueError("Campos ausentes: " + ", ".join(faltantes))
    if extras:
        raise ValueError("Campos não esperados: " + ", ".join(extras))

    dados = reserva.copy()
    for campo in numericos:
        valor = dados[campo]
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            raise ValueError(f"{campo} deve ser um número.")
        try:
            finito = math.isfinite(valor)
        except OverflowError:
            finito = False
        if not finito or valor < 0:
            raise ValueError(f"{campo} deve ser um número finito e não negativo.")
        if campo != "avg_price_per_room":
            if valor != int(valor):
                raise ValueError(f"{campo} deve ser inteiro.")
            dados[campo] = int(valor)

    for campo in categoricos:
        if not isinstance(dados[campo], str) or not dados[campo].strip():
            raise ValueError(f"{campo} deve ser um texto não vazio.")
        dados[campo] = dados[campo].strip()
    for campo in ["required_car_parking_space", "repeated_guest"]:
        if dados[campo] not in (0, 1):
            raise ValueError(f"{campo} deve ser 0 ou 1.")
    try:
        date(dados["arrival_year"], dados["arrival_month"], dados["arrival_date"])
    except (ValueError, OverflowError):
        raise ValueError("A data de chegada é inválida.") from None
    if dados["no_of_adults"] + dados["no_of_children"] == 0:
        raise ValueError("A reserva deve ter pelo menos um hóspede.")
    if dados["no_of_week_nights"] + dados["no_of_weekend_nights"] == 0:
        raise ValueError("A reserva deve ter pelo menos uma noite.")
    return dados


# Recebi o JSON e usei a Pipeline exportada, sem treinar novamente.
@app.post("/predict")
def predict():
    if not request.is_json:
        return jsonify(erro="Use Content-Type: application/json."), 415
    try:
        reserva = validar_reserva(request.get_json(silent=True))
    except ValueError as erro:
        return jsonify(erro=str(erro)), 400

    entrada = pd.DataFrame([reserva], columns=atributos)
    classe = int(modelo.predict(entrada)[0])
    indice_cancelada = list(modelo.classes_).index(1)
    probabilidade = float(modelo.predict_proba(entrada)[0, indice_cancelada])
    return jsonify(
        classe=classe,
        booking_status=rotulos[classe],
        probabilidade_cancelamento=round(probabilidade, 4),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
