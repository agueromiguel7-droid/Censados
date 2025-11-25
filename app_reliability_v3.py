import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

# Importar solo KaplanMeierFitter de lifelines, y las librerías de scipy para las distribuciones
from lifelines import KaplanMeierFitter
from scipy.stats import lognorm, weibull_min, expon, gamma, linregress

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Herramienta de Análisis de Confiabilidad - Gráficas de Probabilidad",
    layout="wide"
)

# --- Funciones de Transformación (Basado en la lógica de Excel/VBA) ---

def FWeibKM(St):
    """Transformación Weibull: ln(ln(1/S(t)))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(np.log(1 / St))

def FLogKM(St):
    """Transformación Lognormal: Normal Inversa de (1 - S(t))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    # Norm_S_Inv de Excel es norm.ppf en SciPy (usamos lognorm para obtener el punto Z transformado)
    return lognorm.ppf(1 - St, 1, 0)

def FExpKM(St):
    """Transformación Exponencial: ln(1/S(t))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(1 / St)

def calculate_r_squared(kmf_curve, fitted_sf):
    """Calcula el R^2 comparando la KM con la curva ajustada S(t) de SciPy."""
    
    # Asegurarse de que las series tengan el mismo índice o longitud
    if len(kmf_curve) != len(fitted_sf):
        # Intentar alinear los índices de S(t) de KM para la regresión
        common_index = kmf_curve.index.intersection(fitted_sf.index)
        if common_index.empty:
             return 0.0
        km_sf = kmf_curve.loc[common_index].iloc[:, 0]
        fitted_sf = fitted_sf.loc[common_index].iloc[:, 0]
    else:
        km_sf = kmf_curve.iloc[:, 0]

    if len(km_sf) == 0:
        return 0.0

    mean_km = km_sf.mean()
    stc = np.sum((km_sf - mean_km)**2)
    scr = np.sum((km_sf - fitted_sf)**2)

    if stc == 0:
        return 1.0
    
    return max(0, 1 - (scr / stc))

# Función para cargar y procesar los datos pegados
def process_pasted_data(data_fallos_str, data_censurados_str):
    """Procesa las cadenas de texto pegadas, las combina y escala los tiempos."""
    
    def parse_text_data(data_str, event_val):
        try:
            df = pd.read_csv(io.StringIO(data_str), header=None, names=['Tiempo'], skipinitialspace=True)
            df = df.dropna(subset=['Tiempo'])
            df['Tiempo'] = pd.to_numeric(df['Tiempo'], errors='coerce')
            df['Evento'] = event_val
            return df.dropna(subset=['Tiempo'])
        except Exception:
            return pd.DataFrame({'Tiempo': [], 'Evento': []})

    df_fallos = parse_text_data(data_fallos_str, 1)
    df_censurados = parse_text_data(data_censurados_str, 0)
    data_combined = pd.concat([df_fallos, df_censurados], ignore_index=True)
    
    # --- Validación y Escalamiento ---
    epsilon = 0.00001
    
    if (data_combined['Tiempo'] <= 0).any():
        st.warning(f"¡Advertencia! Se detectaron tiempos <= 0. Se han reemplazado con {epsilon} para permitir el ajuste.")
        data_combined.loc[data_combined['Tiempo'] <= 0, 'Tiempo'] = epsilon
    
    T_scaled = data_combined['Tiempo']
    
    # Escalar si los datos están en un rango grande para mejorar la estabilidad
    mean_T = T_scaled.mean()
    if T_scaled.max() > 1 and mean_T > 0:
        data_combined['Tiempo_Original'] = T_scaled # Guardamos el original para la salida
        data_combined['Tiempo'] = T_scaled / mean_T
        st.info(f"Los datos de tiempo fueron escalados por su media ({mean_T:.2f}) para mejorar la convergencia numérica. Las salidas de tiempo se reescalarán.")
    else:
        mean_T = 1.0 
        data_combined['Tiempo_Original'] = T_scaled 

    st.session_state['scaling_factor'] = mean_T

    return data_combined

# Función para manejar el reinicio
def reset_application():
    """Limpia los estados de sesión para el reinicio."""
    keys_to_delete = ['T', 'E', 'data_loaded', 'scaling_factor', 'Tiempo_Original']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # Limpiamos las áreas de texto forzando un rerun
    st.rerun()

# --- Encabezado de la Herramienta ---
col1, col2 = st.columns([1, 4])

LOGO_FILE = "grupo.reliarisk.png"
try:
    with open(LOGO_FILE, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    col1.markdown(
        f'<div style="display:flex; align-items:center;">'
        f'<img src="data:image/png;base64,{encoded_string}" width="100">'
        f'</div>',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    col1.error(f"¡Advertencia! No se encontró el archivo '{LOGO_FILE}'.")

col2.title("Herramienta de Análisis de Confiabilidad")
col2.header("Estimación por Gráficas de Probabilidad (Método Lineal)")

st.markdown("---")

## 📤 Carga de Datos y Control (En el Cuerpo Principal)

st.header("⚙️ 1. Carga de Datos")
st.info("Copia y pega los valores de tiempo en los campos de texto correspondientes (un valor por línea).")

col_fallas, col_censura = st.columns(2)

with col_fallas:
    st.markdown("### Datos de Tiempo de FALLA (Evento = 1)")
    # El estado de la sesión guarda el valor entre reruns
    data_fallos_str = st.text_area(
        "Pega aquí los tiempos de falla (Ctrl+V):",
        height=150,
        key="data_fallos_input"
    )

with col_censura:
    st.markdown("### Datos de Tiempo CENSURADO (Evento = 0)")
    data_censurados_str = st.text_area(
        "Pega aquí los tiempos censurados (Ctrl+V):",
        height=150,
        key="data_censurados_input"
    )

# --- Botones de Control ---
col_start, col_reset, col_spacer = st.columns([1, 1, 3])

if col_start.button("▶️ Iniciar Análisis", key="start_analysis"):
    if data_fallos_str.strip() or data_censurados_str.strip():
        try:
            # 1. Procesar datos y aplicar escalado/validación
            data = process_pasted_data(data_fallos_str, data_censurados_str)
            
            if data.empty:
                st.error("No se detectaron datos numéricos válidos en la entrada.")
            else:
                # 2. Guardar en estado de sesión (ESTO ES CRUCIAL)
                st.session_state['T'] = data['Tiempo'].astype(float)
                st.session_state['E'] = data['Evento'].astype(int)
                st.session_state['Tiempo_Original'] = data['Tiempo_Original'].astype(float)
                st.session_state['data_loaded'] = True
                st.success("Datos cargados y combinados correctamente.")
                
                # 3. Forzar la recarga para que el bloque de análisis se ejecute
                st.rerun() 
        except Exception as e:
            st.error(f"Error al procesar los datos. Asegúrate de que sean números. Error: {e}")
    else:
        st.warning("Por favor, pega datos en al menos uno de los campos para iniciar el análisis.")

col_reset.button("🗑️ Reiniciar Cálculo", on_click=reset_application, key="reset_button")


# --- Lógica de Procesamiento y Análisis ---

if st.session_state.get('data_loaded', False):
    T = st.session_state['T']
    E = st.session_state['E']
    scaling_factor = st.session_state['scaling_factor']
    
    st.success(f"Análisis activo: Total de {len(T)} puntos. {E.sum()} fallos, {len(T) - E.sum()} censurados. Factor de escalado: {scaling_factor:.2f}")

    # Mostrar la previsualización de datos combinados
    st.subheader("Datos Combinados (Primeras 10 Filas)")
    df_preview = pd.DataFrame({
        'Tiempo (Escalado)': T, 
        'Tiempo (Original)': st.session_state['Tiempo_Original'], 
        'Evento': E
    }).sort_values(by='Tiempo (Escalado)').head(10)
    st.dataframe(df_preview)
    st.markdown("---")


    ## 📊 2. Estimación Kaplan-Meier y Ajuste Lineal
    
    # --- 1. Estimador Kaplan-Meier ---
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    km_df = kmf.survival_function_.reset_index()
    T_unique = km_df['timeline'].values
    S_t = km_df.iloc[:, 1].values

    # Excluir los puntos donde S(t) es 0 o 1 para la regresión logarítmica
    valid_indices = (S_t > 0) & (S_t < 1)
    T_valid = T_unique[valid_indices]
    S_t_valid = S_t


