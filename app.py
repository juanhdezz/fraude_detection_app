import streamlit as st
import joblib
import pandas as pd

# Cargar el modelo y el scaler
model = joblib.load('modelo_prediccion_fraude_credito.joblib')
scaler = joblib.load('scaler_fraude_credito.joblib')

# CSS personalizado con imagen de fondo y ajuste de colores
st.markdown(
    """
    <style>
    /* Imagen de fondo */
    .main {
        background-image: url('fondo.jpg');
        background-size: cover;
        background-position: center;
        color: white;
        padding: 20px;
    }
    
    /* Estilo de la cabecera */
    .header {
        background-color: rgba(0, 0, 0, 0.7); /* Fondo negro con opacidad */
        padding: 20px;
        text-align: center;
        color: white;
        border-radius: 10px;
    }
    
    h1 {
        margin: 0;
        color: #f1c40f; /* Amarillo dorado */
    }
    
    /* Estilo del formulario y botones */
    .form-container {
        background-color: rgba(0, 0, 0, 0.7); /* Fondo negro con opacidad */
        padding: 20px;
        border-radius: 10px;
    }
    
    .stButton>button {
        background-color: #e67e22; /* Naranja */
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        font-size: 16px;
        cursor: pointer;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #d35400; /* Naranja más oscuro */
    }
    
    /* Estilo del footer */
    .footer {
        background-color: rgba(0, 0, 0, 0.7); /* Fondo negro con opacidad */
        padding: 10px;
        text-align: center;
        color: white;
        border-radius: 10px;
        margin-top: 20px;
    }
    
    .footer p {
        margin: 0;
        font-size: 14px;
    }
    
    .footer a {
        color: #f1c40f; /* Amarillo dorado */
        margin: 0 10px;
        text-decoration: none;
        font-size: 20px;
    }
    
    .footer a:hover {
        color: #e67e22; /* Naranja */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Incluir la librería de iconos Font Awesome
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Cabecera
st.markdown(
    """
    <div class="header">
        <h1>Detector de Fraude con Tarjetas de Crédito</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Crear un formulario dentro de un contenedor con fondo semitransparente
with st.form(key='my_form'):
    st.markdown(
        """
        <div class="form-container">
        """,
        unsafe_allow_html=True
    )
    
    distance_from_home = st.number_input("Distancia desde el hogar")
    distance_from_last_transaction = st.number_input("Distancia desde la última transacción")
    ratio_to_median_purchase_price = st.number_input("Proporción de compra al precio medio")
    repeat_retailer = st.selectbox("¿Compra del mismo vendedor?", ["1", "0"])
    used_chip = st.selectbox("¿Transacción con chip?", ["1", "0"])
    used_pin_number = st.selectbox("¿Transacción con PIN?", ["1", "0"])
    online_order = st.selectbox("¿Compra online?", ["1", "0"])

    submitted = st.form_submit_button("Predecir")

    st.markdown(
        """
        </div>
        """,
        unsafe_allow_html=True
    )

# Si se presiona el botón
if submitted:
    new_data = pd.DataFrame([[distance_from_home, distance_from_last_transaction, ratio_to_median_purchase_price,
                             int(repeat_retailer), int(used_chip), int(used_pin_number), int(online_order)]],
                            columns=['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price',
                                     'repeat_retailer', 'used_chip', 'used_pin_number', 'online_order'])

    new_data_scaled = scaler.transform(new_data)
    probabilidad_fraude = model.predict_proba(new_data_scaled)[0][1]

    if probabilidad_fraude > 0.5:
        st.error("¡Alerta! La transacción podría ser fraudulenta.")
        st.write(f"Probabilidad de fraude: {probabilidad_fraude:.2f}")
    else:
        st.success("La transacción parece legítima.")
        st.write(f"Probabilidad de fraude: {probabilidad_fraude:.2f}")

# Footer con íconos de contacto
st.markdown(
    """
    <div class="footer">
        <p>&copy; 2024 Juan Hernández "Fraud detection machine". Todos los derechos reservados.</p>
        <a href="https://github.com/juanhdezz" target="_blank"><i class="fab fa-github"></i> GitHub</a>
        <a href="https://www.linkedin.com/in/juan-hernandez-sag/" target="_blank"><i class="fab fa-linkedin"></i> LinkedIn</a>
        <a href="mailto:jhernandezsanchezagesta@gmail.com"><i class="fas fa-envelope"></i> Contacto</a>
    </div>
    """,
    unsafe_allow_html=True
)
