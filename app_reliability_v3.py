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

# =================================================================
# --- 1. FUNCIONES DE UTILIDAD Y CÁLCULO ---
# =================================================================

def FWeibKM(St):
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(np.log(1 / St))

def FLogKM(St):
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return lognorm.ppf(1 - St, 1, 0)

def FExpKM(St):
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(1 / St)

def calculate_r_squared(kmf_curve, fitted_sf):
    # Lógica de R^2 simplificada para estabilidad
    if len(kmf_curve) == 0: return 0.0
    
    km_sf = kmf_curve.iloc[:, 0]
    mean_km = km_sf.mean()
    stc = np.sum((km_sf - mean_km)**2)
    scr = np.sum((km_sf - fitted_sf.iloc[:, 0])**2)
    
    if stc == 0: return 1.0
    return max(0, 1 - (scr / stc))

def reset_application():
    """Limpia los estados de sesión y el input."""
    # Eliminamos solo las variables de estado, los text_area conservan su valor
    keys_to_delete = ['T', 'E', 'data_loaded', 'scaling_factor', 'Tiempo_Original']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


def process_and_fit_data(data_fallos_str, data_censurados_str):
    """
    Procesa, valida, escala los datos, y realiza todos los ajustes (Kaplan-Meier y Regresión).
    Devuelve los resultados del ajuste (dist_results) y el factor de escalado.
    """
    
    # 1. Parsing de datos
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
    
    if data_combined.empty:
        return None, None
    
    # --- Validación y Escalamiento ---
    epsilon = 0.00001
    if (data_combined['Tiempo'] <= 0).any():
        st.warning(f"¡Advertencia! Se detectaron tiempos <= 0. Se han reemplazado con {epsilon} para permitir el ajuste.")
        data_combined.loc[data_combined['Tiempo'] <= 0, 'Tiempo'] = epsilon
    
    T_scaled = data_combined['Tiempo']
    mean_T = T_scaled.mean()
    
    if T_scaled.max() > 1 and mean_T > 0:
        data_combined['Tiempo_Original'] = T_scaled 
        data_combined['Tiempo'] = T_scaled / mean_T
        st.info(f"Los datos de tiempo fueron escalados por su media ({mean_T:.2f}) para mejorar la convergencia numérica. Las salidas de tiempo se reescalarán.")
    else:
        mean_T = 1.0 
        data_combined['Tiempo_Original'] = T_scaled 

    T = data_combined['Tiempo']
    E = data_combined['Evento'].astype(int)

    # 2. Análisis Kaplan-Meier
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    km_df = kmf.survival_function_.reset_index()
    T_unique = km_df['timeline'].values
    S_t = km_df.iloc[:, 1].values

    # Excluir los puntos donde S(t) es 0 o 1 para la regresión
    valid_indices = (S_t > 0) & (S_t < 1)
    T_valid = T_unique[valid_indices]
    S_t_valid = S_t[valid_indices]
    Lnt_valid = np.log(T_valid)

    dist_results = {}
    
    # 3. Ajuste Lineal (Regresión Lineal)
    distribuciones_lineales = {
        'Weibull': {'transform_y': FWeibKM, 'transform_x': Lnt_valid},
        'Lognormal': {'transform_y': FLogKM, 'transform_x': Lnt_valid},
        'Exponencial': {'transform_y': FExpKM, 'transform_x': T_valid}
    }
    
    for name, params in distribuciones_lineales.items():
        try:
            Y_trans = params['transform_y'](S_t_valid)
            X_trans = params['transform_x']
            
            Pend, Inter, r_value, p_value, std_err = linregress(X_trans, Y_trans)
            R2 = r_value**2
            
            if name == 'Weibull':
                rho = Pend; eta = np.exp(-Inter / rho)
                dist_results['Weibull'] = {'dist': weibull_min(c=rho, scale=eta), 'R2': R2, 'params': {'rho': rho, 'eta': eta}}
            
            elif name == 'Lognormal':
                sigma = 1 / Pend; mu = -Inter / Pend
                dist_results['Lognormal'] = {'dist': lognorm(s=sigma, scale=np.exp(mu)), 'R2': R2, 'params': {'mu': mu, 'sigma': sigma}}

            elif name == 'Exponencial':
                landa = Pend
                dist_results['Exponencial'] = {'dist': expon(scale=1/landa), 'R2': R2, 'params': {'landa': landa}}

        except Exception:
            st.warning(f"⚠️ Error al ajustar la distribución {name}. Revise la consistencia de sus datos.")

    # 4. Estimación de Gamma por Momentos
    try:
        T_non_censored = T[E == 1]
        Xmedia = T_non_censored.mean()
        Xds = T_non_censored.std(ddof=1)
        
        escalaG_est = (Xds * Xds) / Xmedia
        formaG_est = Xmedia / escalaG_est

        gamma_dist = gamma(a=formaG_est, scale=escalaG_est)
        
        S_t_fitted_gamma = 1 - gamma_dist.cdf(T_unique)
        df_fitted_gamma = pd.DataFrame(S_t_fitted_gamma, index=T_unique)
        R2_gamma = calculate_r_squared(kmf.survival_function_, df_fitted_gamma)

        dist_results['Gamma'] = {'dist': gamma_dist, 'R2': R2_gamma, 'params': {'forma': formaG_est, 'escala': escalaG_est}}
        
    except Exception:
        st.warning("⚠️ Error al estimar la distribución Gamma.")
        
    # Devolvemos todos los datos y resultados necesarios
    return {
        'T': T, 
        'E': E, 
        'kmf': kmf, 
        'T_Original': data_combined['Tiempo_Original'],
        'dist_results': dist_results, 
        'scaling_factor': mean_T
    }, None

# =================================================================
# --- 2. INTERFAZ Y FLUJO PRINCIPAL ---
# =================================================================

# --- Encabezado ---
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

# El análisis se ejecutará automáticamente si hay datos, pero usamos el botón para forzar la acción
# Si el usuario hace clic en Iniciar Análisis, la página se recarga, y el análisis continúa debajo.
if col_start.button("▶️ Iniciar Análisis", key="start_analysis"):
    # Si hay datos, la recarga ocurrirá y el análisis comenzará automáticamente abajo.
    if not (data_fallos_str.strip() or data_censurados_str.strip()):
        st.warning("Por favor, pega datos en al menos uno de los campos para iniciar el análisis.")
    st.info("Procesando datos...")
    st.rerun() # Forzamos la recarga para que el análisis se ejecute con el estado actualizado

col_reset.button("🗑️ Reiniciar Cálculo", on_click=reset_application, key="reset_button")

st.markdown("---")

# =================================================================
# --- 3. INICIO DEL ANÁLISIS (AUTOMÁTICO SI HAY DATOS) ---
# =================================================================

# La aplicación intenta procesar los datos directamente del text_area
if data_fallos_str.strip() or data_censurados_str.strip():
    
    try:
        # Llamamos a la función de procesamiento y ajuste
        analysis_data, error = process_and_fit_data(data_fallos_str, data_censurados_str)
    except Exception as e:
        st.error(f"Error crítico al procesar y ajustar los datos. Causa: {e}")
        analysis_data = None

    if analysis_data:
        # Desempaquetar los resultados
        kmf = analysis_data['kmf']
        dist_results = analysis_data['dist_results']
        scaling_factor = analysis_data['scaling_factor']
        T = analysis_data['T']
        E = analysis_data['E']
        
        st.success(f"Análisis activo: Total de {len(T)} puntos. {E.sum()} fallos, {len(T) - E.sum()} censurados. Factor de escalado: {scaling_factor:.2f}")

        # Mostrar la previsualización de datos combinados
        st.subheader("Datos Combinados (Primeras 10 Filas)")
        df_preview = pd.DataFrame({
            'Tiempo (Escalado)': T, 
            'Tiempo (Original)': analysis_data['T_Original'], 
            'Evento': E
        }).sort_values(by='Tiempo (Escalado)').head(10)
        st.dataframe(df_preview)
        st.markdown("---")


        ## 📊 2. Curva de Confiabilidad y Ajustes
        
        st.header("📊 2. Curva de Confiabilidad y Ajustes")
        fig, ax = plt.subplots(figsize=(12, 6))

        # Graficar Kaplan-Meier
        kmf.plot_survival_function(ax=ax, linestyle='-', marker='o', markeredgecolor='b', color='b', markersize=4, label='Kaplan-Meier')

        color_map = {'Weibull': 'orange', 'Lognormal': 'green', 'Exponencial': 'red', 'Gamma': 'purple'}
        T_plot = kmf.survival_function_.index.values

        for name, result in dist_results.items():
            try:
                dist = result['dist']
                S_t_fitted = 1 - dist.cdf(T_plot)
                
                df_fitted = pd.DataFrame(S_t_fitted, index=T_plot, columns=[name])
                df_fitted.plot(ax=ax, color=color_map[name], linewidth=2, label=f'Ajuste {name}')
            except Exception:
                st.error(f"Error al graficar {name}. Parámetros inválidos.")

        ax.set_title('Curva de Confiabilidad (Kaplan-Meier vs. Distribuciones Ajustadas)')
        ax.set_xlabel(f'Tiempo (Escalado por {scaling_factor:.2f})')
        ax.set_ylabel('Confiabilidad (Survival Function, S(t))')
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)


        ## 📝 3. Parámetros y Métricas
        st.header("📝 3. Parámetros de Ajuste y Métricas")

        metric_data = []
        
        for name, result in dist_results.items():
            params = result['params']
            
            main_params = {}
            if name == 'Weibull':
                main_params['Pendiente (rho/forma)'] = f"{params['rho']:.4f}"
                main_params['Intersección (lambda/escala)'] = f"{params['eta'] * scaling_factor:.4f} (Original)"
            elif name == 'Lognormal':
                main_params['Mu (locación)'] = f"{params['mu'] + np.log(scaling_factor):.4f} (Original)"
                main_params['Sigma (escala)'] = f"{params['sigma']:.4f}"
            elif name == 'Exponencial':
                main_params['Tasa (lambda)'] = f"{params['landa'] / scaling_factor:.4f} (Original)"
            elif name == 'Gamma':
                main_params['Forma (a)'] = f"{params['forma']:.4f}"
                main_params['Escala (lambda)'] = f"{params['escala'] * scaling_factor:.4f} (Original)"

            metric_data.append({
                'Distribución': name,
                'Parámetros Principales': main_params,
                '$R^2$ (Regresión)': f"{result['R2']:.4f}",
                'Método': 'Gráfica Lineal' 
            })

        df_metrics = pd.DataFrame(metric_data)
        df_params_expanded = df_metrics['Parámetros Principales'].apply(pd.Series)
        df_final = pd.concat([df_metrics.drop('Parámetros Principales', axis=1), df_params_expanded], axis=1)
        
        st.dataframe(df_final)

        st.markdown("---")
        
        ## ⏳ 4. Cálculo de Percentiles y Tiempos de Vida
        st.header("⏳ 4. Cálculo de Percentiles y Tiempos de Vida")

        valid_dist_names = list(dist_results.keys())

        if valid_dist_names:
            selected_distribution_name = st.selectbox(
                "Selecciona la Distribución de Probabilidad para el cálculo de Percentiles:",
                valid_dist_names,
                key="selector_dist_perc"
            )
            result = dist_results[selected_distribution_name]
            selected_dist = result['dist']

            col_perc, col_time = st.columns(2)

            with col_perc:
                st.markdown("#### ⏳ Calcular Tiempo ($t$) para un Percentil de Confiabilidad ($S(t)$)")
                
                percentil_confiabilidad = st.number_input(
                    "Confiabilidad Requerida ($S(t)$ en %):",
                    min_value=0.01, max_value=99.99, value=90.0, step=1.0, 
                    key="perc_input_confiabilidad"
                ) / 100.0

                try:
                    q = 1.0 - percentil_confiabilidad
                    tiempo_percentil_scaled = selected_dist.ppf(q)
                    tiempo_percentil_original = tiempo_percentil_scaled * scaling_factor
                    
                    st.metric(
                        label=f"Tiempo ($t$) al {percentil_confiabilidad*100:.1f}% de Confiabilidad",
                        value=f"{tiempo_percentil_original:.2f} unidades de tiempo"
                    )
                except Exception:
                    st.error("Error al calcular el tiempo.")

            with col_time:
                st.markdown("#### 🎯 Calcular Percentil de Confiabilidad ($S(t)$) para un Tiempo ($t$) dado")
                
                tiempo_dado_original = st.number_input(
                    "Tiempo ($t$) de Operación:",
                    min_value=analysis_data['T_Original'].min(), 
                    max_value=analysis_data['T_Original'].max() * 1.5,
                    value=analysis_data['T_Original'].mean(), 
                    step=1.0, 
                    key="time_input"
                )
                
                tiempo_dado_scaled = tiempo_dado_original / scaling_factor
                
                try:
                    confiabilidad_st = (1.0 - selected_dist.cdf(tiempo_dado_scaled)) * 100.0
                    st.metric(
                        label=f"Confiabilidad ($S(t)$) a $t={tiempo_dado_original:.2f}$",
                        value=f"{confiabilidad_st:.2f}%"
                    )
                except Exception:
                    st.error("Error al calcular la confiabilidad.")

            st.markdown("---")

            ## 📈 5. Curvas de Distribución Secundarias
            st.header(f"📈 5. Curvas de Distribución para {selected_distribution_name}")
            
            # (Lógica de graficación de PDF, CDF, S(t), h(t) aquí...)
            # Este bloque es largo, pero asegúrate de que esté incluido completamente.
            
            fig_dist, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig
