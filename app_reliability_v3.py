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
    # Norm_S_Inv de Excel es norm.ppf en SciPy
    return lognorm.ppf(1 - St, 1, 0)

def FExpKM(St):
    """Transformación Exponencial: ln(1/S(t))"""
    St = np.clip(St, 1e-6, 1 - 1e-6)
    return np.log(1 / St)

def calculate_r_squared(kmf_curve, fitted_sf):
    """Calcula el R^2 comparando la KM con la curva ajustada S(t) de SciPy."""
    
    # Asegurarse de que las series tengan el mismo índice o longitud
    if len(kmf_curve) != len(fitted_sf):
        # Alinear los índices de S(t) de KM para la regresión
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
        # Si no se escala, el factor de escalado es 1
        mean_T = 1.0 
        data_combined['Tiempo_Original'] = T_scaled 

    st.session_state['scaling_factor'] = mean_T

    return data_combined

# Función para manejar el reinicio
def reset_application():
    """Limpia los estados de sesión para el reinicio."""
    keys_to_delete = ['T', 'E', 'data_loaded', 'scaling_factor']
    for key in keys_to_delete:
        if key in st.session_state:
            del st.session_state[key]
    st.experimental_rerun()

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

## 📤 Carga de Datos y Control
# (Controles de Streamlit y lógica de carga omitidos por brevedad, pero existen en el código)
# ... Lógica de carga y botones de Reiniciar y Analizar

# --- Lógica de Procesamiento y Análisis ---
# ... (Bloques de inicio de sesión)

if st.session_state.get('data_loaded', False):
    T = st.session_state['T']
    E = st.session_state['E']
    scaling_factor = st.session_state['scaling_factor']
    
    st.success(f"Análisis activo: Total de {len(T)} puntos. {E.sum()} fallos, {len(T) - E.sum()} censurados. Factor de escalado: {scaling_factor:.2f}")

    # Mostrar la previsualización de datos combinados
    st.subheader("Datos Combinados (Formato [Tiempo, Evento])")
    df_preview = pd.DataFrame({'Tiempo (Escalado)': T, 'Tiempo (Original)': st.session_state['Tiempo_Original'], 'Evento': E}).sort_values(by='Tiempo (Escalado)').head(10)
    st.dataframe(df_preview)
    st.markdown("---")


    ## 📊 1. Estimación Kaplan-Meier y Ajuste Lineal
    
    # --- 1. Estimador Kaplan-Meier ---
    kmf = KaplanMeierFitter()
    kmf.fit(T, E)
    km_df = kmf.survival_function_.reset_index()
    T_unique = km_df['timeline'].values
    S_t = km_df.iloc[:, 1].values

    # Excluir los puntos donde S(t) es 0 o 1 para la regresión logarítmica
    valid_indices = (S_t > 0) & (S_t < 1)
    T_valid = T_unique[valid_indices]
    S_t_valid = S_t[valid_indices]
    Lnt_valid = np.log(T_valid)

    # Inicializar el diccionario de resultados de ajuste
    dist_results = {}
    distribuciones_a_probar = {
        'Weibull': {'transform_y': FWeibKM, 'transform_x': Lnt_valid},
        'Lognormal': {'transform_y': FLogKM, 'transform_x': Lnt_valid},
        'Exponencial': {'transform_y': FExpKM, 'transform_x': T_valid}, # Exponencial usa T sin log
        'Gamma': {'transform_y': None, 'transform_x': None} # Gamma se calcula por momentos, no regresión lineal
    }
    
    # --- 2. Ajuste a Distribuciones por Regresión Lineal ---
    
    # 2.1 Ajuste Lineal para Weibull, Lognormal, Exponencial
    for name, params in distribuciones_a_probar.items():
        if name in ['Weibull', 'Lognormal']:
            try:
                Y_trans = params['transform_y'](S_t_valid)
                X_trans = params['transform_x']
                
                # Regresión Lineal (Slope, Intercept)
                Pend, Inter, r_value, p_value, std_err = linregress(X_trans, Y_trans)
                
                if name == 'Weibull':
                    rho = Pend        # forma
                    eta = np.exp(-Inter / rho) # escala
                    dist_results['Weibull'] = {'dist': weibull_min(c=rho, scale=eta), 'R2': r_value**2, 'params': {'rho': rho, 'eta': eta}}
                
                elif name == 'Lognormal':
                    sigma = 1 / Pend  # escala
                    mu = -Inter / Pend # locación
                    dist_results['Lognormal'] = {'dist': lognorm(s=sigma, scale=np.exp(mu)), 'R2': r_value**2, 'params': {'mu': mu, 'sigma': sigma}}

            except Exception as e:
                st.warning(f"⚠️ Error al ajustar la distribución {name}. Causa: Falla de regresión o datos no lineales.")

        elif name == 'Exponencial':
             try:
                Y_trans = params['transform_y'](S_t_valid)
                X_trans = params['transform_x'] # Tiempos (t)
                
                # Regresión Lineal (Slope, Intercept)
                PendE, InterE, r_value, p_value, std_err = linregress(X_trans, Y_trans)
                
                landa = PendE # tasa
                dist_results['Exponencial'] = {'dist': expon(scale=1/landa), 'R2': r_value**2, 'params': {'landa': landa}}

             except Exception as e:
                 st.warning(f"⚠️ Error al ajustar la distribución {name}. Causa: Falla de regresión o datos no lineales.")

    # 2.2 Estimación de Gamma por Método de Momentos (Más estable que MLE)
    try:
        T_non_censored = T[E == 1]
        Xmedia = T_non_censored.mean()
        Xds = T_non_censored.std(ddof=1)
        
        # Parámetros del Método de Momentos
        escalaG_est = (Xds * Xds) / Xmedia
        formaG_est = Xmedia / escalaG_est

        # Usar scipy.stats: a=forma, scale=escala
        gamma_dist = gamma(a=formaG_est, scale=escalaG_est)
        
        # Calcular un R2 de Gamma vs. KM para la tabla
        S_t_fitted_gamma = 1 - gamma_dist.cdf(T_unique)
        df_fitted_gamma = pd.DataFrame(S_t_fitted_gamma, index=T_unique)
        R2_gamma = calculate_r_squared(kmf.survival_function_, df_fitted_gamma)

        dist_results['Gamma'] = {'dist': gamma_dist, 'R2': R2_gamma, 'params': {'forma': formaG_est, 'escala': escalaG_est}}
        
    except Exception as e:
        st.warning(f"⚠️ Error al estimar la distribución Gamma. Causa: {e}")
    

    # --- 3. Gráfico Principal (KM + Ajustes) ---
    fig, ax = plt.subplots(figsize=(12, 6))

    # Graficar Kaplan-Meier
    kmf.plot_survival_function(ax=ax, linestyle='-', marker='o', markeredgecolor='b', color='b', markersize=4, label='Kaplan-Meier')

    color_map = {'Weibull': 'orange', 'Lognormal': 'green', 'Exponencial': 'red', 'Gamma': 'purple'}
    T_plot = kmf.survival_function_.index.values

    for name, result in dist_results.items():
        try:
            dist = result['dist']
            S_t_fitted = 1 - dist.cdf(T_plot)
            
            # Crear un DataFrame para plotear
            df_fitted = pd.DataFrame(S_t_fitted, index=T_plot, columns=[name])
            df_fitted.plot(ax=ax, color=color_map[name], linewidth=2, label=f'Ajuste {name}')
        except Exception as e:
            st.error(f"Error al graficar {name}: {e}")

    ax.set_title('Curva de Confiabilidad (Kaplan-Meier vs. Distribuciones Ajustadas)')
    ax.set_xlabel(f'Tiempo (Escalado por {scaling_factor:.2f})')
    ax.set_ylabel('Confiabilidad (Survival Function, S(t))')
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)


    # --- 4. Parámetros y Métricas ---
    st.subheader("Parámetros de Ajuste y Métricas")

    metric_data = []
    
    for name, result in dist_results.items():
        params = result['params']
        
        main_params = {}
        if name == 'Weibull':
            main_params['Pendiente (rho/forma)'] = f"{params['rho']:.4f}"
            main_params['Intersección (lambda/escala)'] = f"{params['eta'] * scaling_factor:.4f} (Original)"
        elif name == 'Lognormal':
            # μ y σ son invariantes al escalado logarítmico (mu se ajusta con el logaritmo de la escala)
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
            # AIC/Log-Likelihood no están disponibles directamente con regresión lineal
        })

    df_metrics = pd.DataFrame(metric_data)
    df_params_expanded = df_metrics['Parámetros Principales'].apply(pd.Series)
    df_final = pd.concat([df_metrics.drop('Parámetros Principales', axis=1), df_params_expanded], axis=1)
    
    st.dataframe(df_final)

    st.markdown("---")
    
    # --- 5. Análisis de Percentiles (Tiempo vs. Confiabilidad) ---
    st.subheader("Cálculo de Percentiles y Tiempos de Vida")

    # Filtra solo las distribuciones que se ajustaron exitosamente
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
                # Usar la función de supervivencia inversa (PPF) en SciPy.
                # PPF(q) es el valor x tal que CDF(x) = q.
                # Si queremos S(t) = P, entonces CDF(t) = 1 - P.
                q = 1.0 - percentil_confiabilidad
                
                # Tiempo escalado
                tiempo_percentil_scaled = selected_dist.ppf(q)
                
                # Reescalar el tiempo a la unidad original
                tiempo_percentil_original = tiempo_percentil_scaled * scaling_factor
                
                st.metric(
                    label=f"Tiempo ($t$) al {percentil_confiabilidad*100:.1f}% de Confiabilidad",
                    value=f"{tiempo_percentil_original:.2f} unidades de tiempo"
                )
            except Exception as e:
                st.error(f"Error al calcular el tiempo: {e}")

        with col_time:
            st.markdown("#### 🎯 Calcular Percentil de Confiabilidad ($S(t)$) para un Tiempo ($t$) dado")
            
            tiempo_dado_original = st.number_input(
                "Tiempo ($t$) de Operación:",
                min_value=st.session_state['Tiempo_Original'].min(), 
                max_value=st.session_state['Tiempo_Original'].max() * 1.5,
                value=st.session_state['Tiempo_Original'].mean(), 
                step=1.0, 
                key="time_input"
            )
            
            # Escalar el tiempo de entrada
            tiempo_dado_scaled = tiempo_dado_original / scaling_factor
            
            try:
                # Calcular CDF y luego la confiabilidad S(t) = 1 - CDF
                confiabilidad_st = (1.0 - selected_dist.cdf(tiempo_dado_scaled)) * 100.0
                st.metric(
                    label=f"Confiabilidad ($S(t)$) a $t={tiempo_dado_original:.2f}$",
                    value=f"{confiabilidad_st:.2f}%"
                )
            except Exception as e:
                st.error(f"Error al calcular la confiabilidad: {e}")

        st.markdown("---")

        # --- 6. Curvas de Distribución Secundarias ---
        # ... (Lógica de gráficos secundarios con manejo de errores, similar a lo anterior)

        st.header(f"2. Curvas de Distribución para {selected_distribution_name}")
        st.info("Estas curvas se basan en el ajuste paramétrico seleccionado.")

        fig_dist, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig_dist.suptitle(f'Funciones de Distribución para el Ajuste {selected_distribution_name}', fontsize=16)

        # Crear rango de tiempo para graficar
        T_max = st.session_state['Tiempo_Original'].max() * 1.2
        T_plot_original = np.linspace(0.001, T_max, 100)
        T_plot_scaled = T_plot_original / scaling_factor

        try:
            # Densidad de Probabilidad (PDF)
            pdf_values = selected_dist.pdf(T_plot_scaled) / scaling_factor # Desescalar PDF
            axes[0, 0].plot(T_plot_original, pdf_values, color='b')
            axes[0, 0].set_title('Densidad de Probabilidad (PDF)')
            axes[0, 0].set_xlabel('Tiempo')
            axes[0, 0].set_ylabel('$f(t)$')
            axes[0, 0].grid(True, linestyle='--', alpha=0.6)
        except Exception:
             axes[0, 0].text(0.5, 0.5, f"Error al generar PDF: {selected_distribution_name} inestable.", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[0, 0].transAxes)
        
        try:
            # Distribución Acumulada Directa (CDF)
            cdf_values = selected_dist.cdf(T_plot_scaled)
            axes[0, 1].plot(T_plot_original, cdf_values, color='g')
            axes[0, 1].set_title('Distribución Acumulada Directa (CDF)')
            axes[0, 1].set_xlabel('Tiempo')
            axes[0, 1].set_ylabel('$F(t)$')
            axes[0, 1].grid(True, linestyle='--', alpha=0.6)
        except Exception:
            axes[0, 1].text(0.5, 0.5, f"Error al generar CDF.", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[0, 1].transAxes)

        try:
            # Confiabilidad (Survival Function - Acumulada Inversa)
            sf_values = 1 - selected_dist.cdf(T_plot_scaled)
            axes[1, 0].plot(T_plot_original, sf_values, color='r')
            axes[1, 0].set_title('Función de Confiabilidad ($S(t)$)')
            axes[1, 0].set_xlabel('Tiempo')
            axes[1, 0].set_ylabel('$S(t)$')
            axes[1, 0].grid(True, linestyle='--', alpha=0.6)
        except Exception:
            axes[1, 0].text(0.5, 0.5, f"Error al generar S(t).", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[1, 0].transAxes)

        try:
            # Tasa de Falla (Hazard Function)
            # h(t) = f(t) / S(t)
            pdf_values = selected_dist.pdf(T_plot_scaled) / scaling_factor
            sf_values = 1 - selected_dist.cdf(T_plot_scaled)
            # Evitar división por cero o valores cercanos
            hazard_values = np.divide(pdf_values, sf_values, out=np.zeros_like(pdf_values), where=sf_values!=0)
            
            axes[1, 1].plot(T_plot_original, hazard_values, color='k')
            axes[1, 1].set_title('Tasa de Falla (Hazard Function - $h(t)$)')
            axes[1, 1].set_xlabel('Tiempo')
            axes[1, 1].set_ylabel('$h(t)$')
            axes[1, 1].grid(True, linestyle='--', alpha=0.6)
        except Exception:
            axes[1, 1].text(0.5, 0.5, f"Error al generar h(t).", horizontalalignment='center', verticalalignment='center', color='red', transform=axes[1, 1].transAxes)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig_dist)

    else:
        st.error("No hay distribuciones válidas para mostrar. Revise sus datos y el log de advertencias.")
