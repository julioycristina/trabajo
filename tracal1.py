# TRABAJO DE CALCULO CON LAS MISMAS HERRAMIENTAS QUE RIESGO FINANCIERO

import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Título
st.title("Predicción del Desempeño en Problemas de Cálculo")

# Cargar datos
@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_calculo_problemas.csv")

df = cargar_datos()
st.subheader("Vista previa del dataset")
st.dataframe(df.head())

# Codificación de variables categóricas
df_encoded = df.copy()
le_dict = {}

for col in df_encoded.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    le_dict[col] = le  # guardar para usar en predicción

# Separar X e y
X = df_encoded.drop("Resolucion_Correcta", axis=1)
y = df_encoded["Resolucion_Correcta"]

# Dividir en entrenamiento y test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(x_train, y_train)
score = modelo.score(x_test, y_test)

# Mostrar precisión
st.subheader(f"Precisión del modelo: {score:.2f}")

# Matriz de confusión
y_pred = modelo.predict(x_test)
matriz = confusion_matrix(y_test, y_pred)

st.subheader("Matriz de Confusión")
fig, ax = plt.subplots()
sns.heatmap(matriz, annot=True, fmt='d', cmap='Greens', ax=ax)
st.pyplot(fig)

# Importancia de características
importancias = modelo.feature_importances_
importancia_df = pd.DataFrame({
    "Característica": X.columns,
    "Importancia": importancias
}).sort_values(by="Importancia", ascending=False)

st.subheader("Importancia de las Características")
st.bar_chart(importancia_df.set_index("Característica"))

# Formulario de predicción
st.subheader("Formulario de Predicción")

with st.form("formulario_prediccion"):
    entrada = []
    for col in X.columns:
        if col in le_dict:  # es una variable categórica
            opciones = le_dict[col].classes_
            valor = st.selectbox(f"{col}", opciones)
            valor_cod = le_dict[col].transform([valor])[0]
            entrada.append(valor_cod)
        else:  # numérica
            valor = st.number_input(f"{col}", min_value=0.0, max_value=100.0)
            entrada.append(valor)
    
    submit = st.form_submit_button("Predecir")

    if submit:
        datos_pred = pd.DataFrame([entrada], columns=X.columns)
        pred = modelo.predict(datos_pred)[0]
        resultado = "Correcta" if pred == 1 else "Incorrecta"
        st.success(f"Predicción: la resolución será {resultado}")

