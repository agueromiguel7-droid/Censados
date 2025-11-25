import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import base64
import io

# Importaciones necesarias para el método de Gráficas de Probabilidad
from lifelines import KaplanMeierFitter
from scipy.stats import lognorm, weibull_min, expon, gamma, linregress

# --- Configuración Inicial de la Página ---
st.set_page_config(
    page_title="Herramienta de Análisis de Confiabilidad",
    layout="wide"
)

# =================================================================
# --- 1. FUNCIONES DE UTILIDAD Y CÁLCULO ---
# =================================================================

# Funciones de Transformación (Basado en la lógica de Excel/VBA)
def FWeibKM(St):
    """Transformación Weibull: ln(ln(1/S(t)))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(np.log(1 / St))

def FLogKM(St):
    """Transformación Lognormal: Normal Inversa de (1 - S(t))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return lognorm.ppf(1 - St, 1, 0) # Usa lognorm para obtener el valor Z transformado

def FExpKM(St):
    """Transformación Exponencial: ln(1/S(t))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(1 / St)

def calculate_r_squared(kmf_curve, fitted_sf):
    """Calcula el R^2 comparando la KM con la curva ajustada S(t)."""
    if len(kmf_curve) == 0: return 0.0
    
    km_sf = kmf_curve.iloc[:, 0]
    fitted_sf_aligned = fitted_sf.iloc[:, 0]
    
    # Alinear las longitudes
    if len(kmf_curve.index) != len(fitted_sf.index):
        common_index = kmf_curve.index.intersection(fitted_sf.index)
        km_sf = kmf_curve.loc[common_index].iloc[:, 0]
        fitted_sf_aligned = fitted_sf.loc[common_index].iloc[:, 0]

    mean_km = km_sf.mean()
    stc = np.sum((km_sf - mean_km)**2)
    scr = np.sum((km_sf - fitted_sf_aligned)**2)
    
    if stc == 0: return 1.0
    return max(0, 1 - (scr / stc))

def reset_application():
    """Limpia los estados de sesión y el input."""
    keys_to_delete = ['T', 'E', 'data_loaded', 'scaling_factor', 'Tiempo_Original']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun() # Usamos la función estable

def process_and_fit_data(data_fallos_str, data_censurados_str):
    """Procesa, valida, escala los datos, y realiza todos los ajustes por regresión."""
    
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
        return None
    
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

    valid_indices = (S_t > 0) & (S_t < 1)
    T_valid = T_unique[valid_indices]
    S_t_valid = S_t[valid_indices]
    Lnt_valid = np.log(T_valid)

    dist_results = {}
    
    # 3. Ajuste Lineal
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
        
    return {
        'kmf': kmf, 
        'T': T,
        'E': E,
        'T_Original': data_combined['Tiempo_Original'],
        'dist_results': dist_results, 
        'scaling_factor': mean_T
    }


# =================================================================
# --- 2. INTERFAZ Y FLUJO PRINCIPAL ---
# =================================================================

# --- Encabezado ---
col1, col2 = st.columns([1, 4])

LOGO_FILE = "grupo.reliarisk.png"
try:
    with open(LOGO_FILE, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    # Logo al doble de tamaño: 200px
    col1.markdown(
        f'<div style="display:flex; align-items:center;">'
        f'<img src="data:image/png;base64,{encoded_string}" width="200">' 
        f'</div>',
        unsafe_allow_html=True
    )
except FileNotFoundError:
    col1.error(f"¡Advertencia! No se encontró el archivo '{LOGO_FILE}'.")

col2.title("Herramienta de Análisis de Confiabilidad")
col2.header("Estimación por Gráficas de Probabilidad (Método Lineal)")

st.markdown("---")

## 📤 1. Carga de Datos y Visualización (Fila Superior)

col_carga, col_preview = st.columns([1, 1.5])

with col_carga:
    st.subheader("1. Carga de Datos")
    st.info("Copia y pega los valores de tiempo en los campos de texto correspondientes.")
    
    col_fallas, col_censura = st.columns(2)
    with col_fallas:
        st.markdown("##### Tiempos de FALLA ($E=1$)")
        data_fallos_str = st.text_area(
            "Tiempos de Falla:", height=150, key="data_fallos_input", label_visibility="collapsed"
        )
    with col_censura:
        st.markdown("##### Tiempos CENSURADOS ($E=0$)")
        data_censurados_str = st.text_area(
            "Tiempos Censurados:", height=150, key="data_censurados_input", label_visibility="collapsed"
        )

    # Botones de Control
    col_start, col_reset, col_spacer = st.columns([1, 1, 1])

    if col_start.button("▶️ Iniciar Análisis", key="start_analysis"):
        if not (data_fallos_str.strip() or data_censurados_str.strip()):
            st.warning("Por favor, pega datos en al menos uno de los campos para iniciar el análisis.")
        st.info("Procesando datos...")
        st.rerun() 

    col_reset.button("🗑️ Reiniciar Cálculo", on_click=reset_application, key="reset_button")

# --- Lógica de Análisis y Despliegue de Resultados ---
if data_fallos_str.strip() or data_censurados_str.strip():
    try:
        # Llamamos a la función de procesamiento y ajuste
        analysis_data = process_and_fit_data(data_fallos_str, data_censurados_str)
    except Exception as e:
        st.error(f"Error crítico al procesar y ajustar los datos. Causa: {e}")
        analysis_data = None
else:
    analysis_data = None


# El análisis se ejecuta SOLAMENTE si hay datos procesados
if analysis_data:
    # Desempaquetar los resultados
    kmf = analysis_data['kmf']
    dist_results = analysis_data['dist_results']
    scaling_factor = analysis_data['scaling_factor']
    T = analysis_data['T']
    E = analysis_data['E']
    
    # 1.2 Mostrar Preview de Datos (Columna Derecha)
    with col_preview:
        st.subheader("Datos Combinados (Primeras 10 Filas)")
        df_preview = pd.DataFrame({
            'Tiempo (Escalado)': T, 
            'Tiempo (Original)': analysis_data['T_Original'], 
            'Evento': E
        }).sort_values(by='Tiempo (Escalado)').head(10)
        st.dataframe(df_preview, use_container_width=True)
    
    st.markdown("---")

    ## 📊 2. Curva de Confiabilidad (Fila 2 - Completa)
    st.subheader("2. Curva de Confiabilidad ($S(t)$) vs. Distribuciones Ajustadas")
    fig, ax = plt.subplots(figsize=(10, 6))

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
            pass 

    ax.set_title('Curva de Confiabilidad (Kaplan-Meier vs. Distribuciones Ajustadas)')
    ax.set_xlabel(f'Tiempo (Escalado por {scaling_factor:.2f})')
    ax.set_ylabel('Confiabilidad (Survival Function, S(t))')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

    st.markdown("---")
    
    ## 📝 3. Parámetros y Métricas & ⏳ 4. Percentiles (Fila 3 - Dividida)

    col_params, col_percentiles = st.columns([1, 1.2]) # Ajuste de columnas para la visualización

    with col_params:
        st.subheader("3. Parámetros de Ajuste y Métricas")
        metric_data = []
        
        for name, result in dist_results.items():
            params = result['params']
            
            main_params = {}
            if name == 'Weibull':
                main_params['Pendiente (rho/forma)'] = f"{params['rho']:.4f}"
                main_params['Escala (eta)'] = f"{params['eta'] * scaling_factor:.4f}"
            elif name == 'Lognormal':
                main_params['Mu (locación)'] = f"{params['mu'] + np.log(scaling_factor):.4f}"
                main_params['Sigma (escala)'] = f"{params['sigma']:.4f}"
            elif name == 'Exponencial':
                main_params['Tasa (lambda)'] = f"{params['landa'] / scaling_factor:.4f}"
            elif name == 'Gamma':
                main_params['Forma (a)'] = f"{params['forma']:.4f}"
                main_params['Escala (lambda)'] = f"{params['escala'] * scaling_factor:.4f}"

            metric_data.append({
                'Distribución': name,
                '$R^2$': f"{result['R2']:.4f}",
                'Parámetros': main_params,
            })

        df_metrics = pd.DataFrame(metric_data)
        df_params_expanded = df_metrics['Parámetros'].apply(pd.Series)
        df_final = pd.concat([df_metrics.drop('Parámetros', axis=1), df_params_expanded], axis=1)
        
        st.dataframe(df_final, use_container_width=True)


    with col_percentiles:
        st.subheader("4. Cálculo de Percentiles y Tiempos de Vida")

        valid_dist_names = list(dist_results.keys())

        if valid_dist_names:
            selected_distribution_name = st.selectbox(
                "Selecciona la Distribución de Probabilidad para el cálculo de Percentiles:",
                valid_dist_names,
                key="selector_dist_perc"
            )
            result = dist_results[selected_distribution_name]
            selected_dist = result['dist']

            col_perc_input, col_perc_output = st.columns(2)

            with col_perc_input:
                st.markdown("##### Calcular Tiempo ($t$) para un $S(t)$")
                percentil_confiabilidad = st.number_input(
                    "Confiabilidad Requerida ($S(t)$ en %):",
                    min_value=0.01, max_value=99.99, value=90.0, step=1.0, 
                    key="perc_input_confiabilidad"
                ) / 100.0
                try:
                    q = 1.0 - percentil_confiabilidad
                    tiempo_percentil_scaled = selected_dist.ppf(q)
                    tiempo_percentil_original = tiempo_percentil_scaled * scaling_factor
                    st.metric(label=f"Tiempo ($t$) al {percentil_confiabilidad*100:.1f}% de Confiabilidad", value=f"{tiempo_percentil_original:.2f} unidades de tiempo")
                except Exception:
                    st.error("Error en cálculo de tiempo.")

            with col_perc_output:
                st.markdown("##### Calcular $S(t)$ para un Tiempo ($t$) dado")
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
                    st.metric(label=f"Confiabilidad ($S(t)$) a $t={tiempo_dado_original:.2f}$", value=f"{confiabilidad_st:.2f}%")
                except Exception:
                    st.error("Error en cálculo de confiabilidad.")


    st.markdown("---")

    ## 📈 5. Curvas de Distribución Secundarias (Fila 4 - Completa)

    st.header(f"📈 5. Curvas de Distribución para {selected_distribution_name}")
    st.info("Curvas de: Densidad, Acumulada Directa, Confiabilidad y Tasa de Falla.")
    
    fig_dist, axes = plt.subplots(2, 2, figsize=(15, 12))

    T_max = analysis_data['T_Original'].max() * 1.2
    T_plot_original = np.linspace(0.001, T_max, 100)
    T_plot_scaled = T_plot_original / scaling_factor

    try: # 1. Densidad de Probabilidad (PDF)
        pdf_values = selected_dist.pdf(T_plot_scaled) / scaling_factor 
        axes[0, 0].plot(T_plot_original, pdf_values, color='b')
        axes[0, 0].set_title('Curva de Densidad (PDF)')
        axes[0, 0].set_xlabel('Tiempo')
        axes[0, 0].set_ylabel('$f(t)$')
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    except Exception: axes[0, 0].text(0.5, 0.5, f"Error al generar PDF.", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[0, 0].transAxes)
    
    try: # 2. Distribución Acumulada Directa (CDF)
        cdf_values = selected_dist.cdf(T_plot_scaled)
        axes[0, 1].plot(T_plot_original, cdf_values, color='g')
        axes[0, 1].set_title('Curva Acumulada Directa (CDF)')
        axes[0, 1].set_xlabel('Tiempo')
        axes[0, 1].set_ylabel('$F(t)$')
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    except Exception: axes[0, 1].text(0.5, 0.5, f"Error al generar CDF.", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[0, 1].transAxes)

    try: # 3. Confiabilidad (S(t))
        sf_values = 1 - selected_dist.cdf(T_plot_scaled)
        axes[1, 0].plot(T_plot_original, sf_values, color='r')
        axes[1, 0].set_title('Curva Acumulada Inversa (Confiabilidad S(t))')
        axes[1, 0].set_xlabel('Tiempo')
        axes[1, 0].set_ylabel('$S(t)$')
        axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    except Exception: axes[1, 0].text(0.5, 0.5, f"Error al generar S(t).", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[1, 0].transAxes)

    try: # 4. Tasa de Falla (Hazard Function - h(t))
        pdf_values = selected_dist.pdf(T_plot_scaled) / scaling_factor
        sf_values = 1 - selected_dist.cdf(T_plot_scaled)
        hazard_values = np.divide(pdf_values, sf_values, out=np.zeros_like(pdf_values), where=sf_values!=0)
        
        axes[1, 1].plot(T_plot_original, hazard_values, color='k')
        axes[1, 1].set_title('Tasa de Falla (Hazard Function - $h(t)$)')
        axes[1, 1].set_xlabel('Tiempo')
        axes[1, 1].set_ylabel('$h(t)$')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    except Exception: axes[1, 1].text(0.5, 0.5, f"Error al generar h(t).", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[1, 1].transAxes)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    st.pyplot(fig_dist)

else:
    # Mensaje de espera si no hay datos en las cajas
    st.info("Pegue sus datos de tiempos de falla y censurados arriba, y presione 'Iniciar Análisis' para ver las gráficas y los resultados.")
