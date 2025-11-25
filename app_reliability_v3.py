import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines import WeibullFitter, LogNormalFitter, ExponentialFitter
import numpy as np
import base64
import io

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Herramienta de Análisis de Confiabilidad - Kaplan-Meier y Ajuste de Distribuciones",
    layout="wide"
)

# --- Funciones de Utilidad ---

# Función para calcular el R^2 (Pseudo)
def calculate_r_squared(kmf_curve, fitted_curve):
    """Calcula un pseudo R^2 comparando la KM con la curva ajustada."""
    common_index = kmf_curve.index.intersection(fitted_curve.index)
    if common_index.empty:
        return 0.0

    km_sf = kmf_curve.loc[common_index]
    fitted_sf = fitted_curve.loc[common_index]

    # Suma total de cuadrados (STC)
    mean_km = km_sf.mean()
    stc = np.sum((km_sf - mean_km)**2)

    # Suma de cuadrados de los residuos (SCR)
    scr = np.sum((km_sf - fitted_sf)**2)

    if stc == 0:
        return 1.0
    
    return max(0, 1 - (scr / stc))

# Función para cargar y procesar los datos pegados
def process_pasted_data(data_fallos_str, data_censurados_str):
    """Procesa las cadenas de texto pegadas y las combina en el formato (T, E)."""
    
    # Función auxiliar para convertir la cadena en DataFrame de una sola columna
    def parse_text_data(data_str, event_val):
        # Usamos io.StringIO para simular un archivo y pd.read_csv para manejar espacios/saltos
        try:
            df = pd.read_csv(io.StringIO(data_str), header=None, names=['Tiempo'], skipinitialspace=True)
            # Limpiamos filas vacías que pueden aparecer al pegar
            df = df.dropna(subset=['Tiempo'])
            df['Tiempo'] = pd.to_numeric(df['Tiempo'], errors='coerce')
            df['Evento'] = event_val
            return df.dropna(subset=['Tiempo']) # Eliminar filas donde la conversión falló
        except Exception:
            return pd.DataFrame({'Tiempo': [], 'Evento': []})

    # 1. Procesar Fallos (Tiempos de Evento = 1)
    df_fallos = parse_text_data(data_fallos_str, 1)
    
    # 2. Procesar Censurados (Tiempos de Evento = 0)
    df_censurados = parse_text_data(data_censurados_str, 0)
    
    # 3. Combinar los DataFrames
    data_combined = pd.concat([df_fallos, df_censurados], ignore_index=True)
    
    return data_combined

# Función para manejar el reinicio
def reset_application():
    """Limpia los estados de sesión para el reinicio."""
    keys_to_delete = ['T', 'E', 'data_loaded']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    # Forzar la recarga de la aplicación para limpiar los campos de texto
    st.experimental_rerun()

# --- Encabezado de la Herramienta ---
col1, col2 = st.columns([1, 4])

# Intento de cargar y mostrar el logo
LOGO_FILE = "mi_logo.png"
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
col2.header("Estimación Kaplan-Meier y Ajuste de Distribuciones de Probabilidad")

st.markdown("---")

## 📤 Carga de Datos y Control
st.sidebar.header("⚙️ Carga de Datos y Control")
st.sidebar.info("Copia y pega los valores de tiempo en los campos de texto correspondientes (un valor por línea).")

# Campos de texto para pegar datos
with st.sidebar:
    st.markdown("### Datos de Tiempo de FALLA (Evento = 1)")
    data_fallos_str = st.text_area(
        "Pega aquí los tiempos de falla (Ctrl+V):",
        height=150,
        key="data_fallos_input"
    )

    st.markdown("### Datos de Tiempo CENSURADO (Evento = 0)")
    data_censurados_str = st.text_area(
        "Pega aquí los tiempos censurados (Ctrl+V):",
        height=150,
        key="data_censurados_input"
    )

    # Botón de Procesar/Iniciar
    if st.button("▶️ Iniciar Análisis", key="start_analysis"):
        # Esto iniciará el procesamiento
        if data_fallos_str.strip() or data_censurados_str.strip():
            try:
                # Procesar y combinar datos
                data = process_pasted_data(data_fallos_str, data_censurados_str)
                
                if data.empty:
                    st.error("No se detectaron datos numéricos válidos en la entrada.")
                else:
                    # Guardar en estado de sesión para persistir
                    st.session_state['T'] = data['Tiempo'].astype(float)
                    st.session_state['E'] = data['Evento'].astype(int)
                    st.session_state['data_loaded'] = True
                    st.success("Datos cargados y combinados correctamente. Pulsa **'Reiniciar Cálculo'** para cargar nuevos datos.")
                    st.experimental_rerun() # Forzar rerun para mostrar los resultados
            except Exception as e:
                st.error(f"Error al procesar los datos. Asegúrate de que sean números. Error: {e}")
        else:
            st.warning("Por favor, pega datos en al menos uno de los campos para iniciar el análisis.")

    # Botón de Reinicio
    st.button("🗑️ Reiniciar Cálculo", on_click=reset_application, key="reset_button")


# --- Lógica de Procesamiento y Análisis ---
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False

if st.session_state['data_loaded']:
    T = st.session_state['T']
    E = st.session_state['E']
    
    st.success(f"Análisis activo: Total de {len(T)} puntos. {E.sum()} fallos, {len(T) - E.sum()} censurados.")

    # Mostrar la previsualización de datos combinados
    st.subheader("Datos Combinados (Formato [Tiempo, Evento])")
    df_preview = pd.DataFrame({'Tiempo': T, 'Evento': E}).sort_values(by='Tiempo').head(10)
    st.dataframe(df_preview)
    st.markdown("---")

    ## 📊 1. Estimación Kaplan-Meier y Ajuste de Distribuciones

    # --- 1. Estimador Kaplan-Meier ---
    kmf = KaplanMeierFitter()
    kmf.fit(T, E, label='Kaplan-Meier (No Paramétrico)')
    
    # --- 2. Ajuste a Distribuciones ---
    dist_fitters = {
        'Weibull': WeibullFitter().fit(T, E, label='Weibull'),
        'Lognormal': LogNormalFitter().fit(T, E, label='Lognormal'),
        'Exponencial': ExponentialFitter().fit(T, E, label='Exponencial'),
        
    }

    # --- 3. Gráfico Principal (Dispersión KM + Ajustes) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Graficar Kaplan-Meier con puntos (dispersión)
    kmf.plot_survival_function(ax=ax, linestyle='-', marker='o', markeredgecolor='b', color='b', markersize=4)
    
    # Graficar Ajustes de Distribución (línea de ajuste)
    for name, fitter in dist_fitters.items():
        fitter.plot_survival_function(ax=ax, color=fitter._label.lower()[0])

    ax.set_title('Curva de Confiabilidad (Kaplan-Meier vs. Distribuciones Ajustadas)')
    ax.set_xlabel('Tiempo')
    ax.set_ylabel('Confiabilidad (Survival Function, S(t))')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)
    
    st.markdown("---")

    # --- 4. Parámetros y Métricas ---
    st.subheader("Parámetros de Ajuste y Métricas")

    metric_data = []
    for name, fitter in dist_fitters.items():
        params = fitter.summary['coef'].T
        
        main_params = {}
        if name == 'Weibull':
            main_params['Pendiente (rho/forma)'] = f"{params.get('rho_', np.nan):.4f}"
            main_params['Intersección (lambda/escala)'] = f"{params.get('lambda_', np.nan):.4f}"
        elif name == 'Lognormal':
            main_params['Mu (locación)'] = f"{params.get('mu_', np.nan):.4f}"
            main_params['Sigma (escala)'] = f"{params.get('sigma_', np.nan):.4f}"
        elif name == 'Exponencial':
            main_params['Tasa (lambda)'] = f"{params.get('lambda_', np.nan):.4f}"
        

        # Calcular R^2
        r2 = calculate_r_squared(kmf.survival_function_.iloc[:, 0], fitter.survival_function_.iloc[:, 0])

        metric_data.append({
            'Distribución': name,
            'Parámetros Principales': main_params,
            '$R^2$ (Pseudo)': f"{r2:.4f}",
            'Log-Likelihood': f"{fitter.log_likelihood_:.2f}",
            'AIC': f"{fitter.AIC_:.2f}"
        })

    df_metrics = pd.DataFrame(metric_data)
    df_params_expanded = df_metrics['Parámetros Principales'].apply(pd.Series)
    df_final = pd.concat([df_metrics.drop('Parámetros Principales', axis=1), df_params_expanded], axis=1)
    
    st.dataframe(df_final)

    st.markdown("---")
    
    # --- 5. Análisis de Percentiles (Tiempo vs. Confiabilidad) ---
    st.subheader("Cálculo de Percentiles y Tiempos de Vida")

    selected_distribution_name = st.selectbox(
        "Selecciona la Distribución de Probabilidad para el cálculo de Percentiles:",
        list(dist_fitters.keys()),
        key="selector_dist_perc"
    )
    selected_fitter = dist_fitters[selected_distribution_name]

    col_perc, col_time = st.columns(2)

    with col_perc:
        st.markdown("#### ⏳ Calcular Tiempo ($t$) para un Percentil de Confiabilidad ($S(t)$)")
        
        percentil_confiabilidad = st.number_input(
            "Confiabilidad Requerida ($S(t)$ en %):",
            min_value=0.01, max_value=99.99, value=90.0, step=1.0, 
            key="perc_input_confiabilidad"
        ) / 100.0

        try:
            tiempo_percentil = selected_fitter.percentile_of_survival_function(percentil_confiabilidad)
            st.metric(
                label=f"Tiempo ($t$) al {percentil_confiabilidad*100:.1f}% de Confiabilidad",
                value=f"{tiempo_percentil.iloc[0]:.2f} unidades de tiempo"
            )
        except Exception:
            st.error("Asegúrate de que el valor esté dentro del rango lógico de la distribución.")

    with col_time:
        st.markdown("#### 🎯 Calcular Percentil de Confiabilidad ($S(t)$) para un Tiempo ($t$) dado")
        
        tiempo_dado = st.number_input(
            "Tiempo ($t$) de Operación:",
            min_value=T.min(), max_value=T.max() * 1.5, value=T.mean(), step=1.0, 
            key="time_input"
        )
        
        try:
            confiabilidad_st = selected_fitter.survival_function_at_times(tiempo_dado).iloc[0] * 100.0
            st.metric(
                label=f"Confiabilidad ($S(t)$) a $t={tiempo_dado:.2f}$",
                value=f"{confiabilidad_st:.2f}%"
            )
        except Exception:
            st.error("Asegúrate de que el tiempo esté dentro del rango de la función de supervivencia calculada.")

    st.markdown("---")

    # --- 6. Curvas de Distribución Secundarias ---
    st.header(f"2. Curvas de Distribución para {selected_distribution_name}")
    st.info("Estas curvas se basan en el ajuste paramétrico seleccionado.")

    fig_dist, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig_dist.suptitle(f'Funciones de Distribución para el Ajuste {selected_distribution_name}', fontsize=16)

    # 1. Densidad de Probabilidad (PDF)
    selected_fitter.plot_pdf(ax=axes[0, 0], loc=slice(None), show_censors=False, color='b')
    axes[0, 0].set_title('Densidad de Probabilidad (PDF)')
    axes[0, 0].set_xlabel('Tiempo')
    axes[0, 0].set_ylabel('$f(t)$')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Distribución Acumulada Directa (CDF)
    selected_fitter.plot_cumulative_density(ax=axes[0, 1], loc=slice(None), show_censors=False, color='g')
    axes[0, 1].set_title('Distribución Acumulada Directa (CDF)')
    axes[0, 1].set_xlabel('Tiempo')
    axes[0, 1].set_ylabel('$F(t)$')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Confiabilidad (Survival Function - Acumulada Inversa)
    selected_fitter.plot_survival_function(ax=axes[1, 0], loc=slice(None), show_censors=False, color='r')
    axes[1, 0].set_title('Función de Confiabilidad ($S(t)$)')
    axes[1, 0].set_xlabel('Tiempo')
    axes[1, 0].set_ylabel('$S(t)$')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Tasa de Falla (Hazard Function)
    selected_fitter.plot_hazard(ax=axes[1, 1], loc=slice(None), show_censors=False, color='k')
    axes[1, 1].set_title('Tasa de Falla (Hazard Function - $h(t)$)')
    axes[1, 1].set_xlabel('Tiempo')
    axes[1, 1].set_ylabel('$h(t)$')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_dist)

else:

    st.info("Para comenzar el análisis, pega los datos de **Tiempos de Falla** y **Tiempos Censurados** en las áreas de texto de la barra lateral y presiona **'Iniciar Análisis'**.")
