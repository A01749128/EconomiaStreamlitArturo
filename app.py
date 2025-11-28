import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# --- Configuración de Página ---
st.set_page_config(
    page_title="Dashboard Económico Global",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Definimos solo las dos paletas solicitadas
color_palettes = {
    "Ocean Breeze": {
        "button_color": "#008CBA",
        "sidebar_bg_color": "#E0F7FA",
        "main_bg_color": "#B2EBF2",
        "text_color": "#003B2E",
        "top_bar_color": "#006B8A"
    },
    "Forest Green": {
        "button_color": "#4CAF50",
        "sidebar_bg_color": "#E8F5E9",
        "main_bg_color": "#C8E6C9",
        "text_color": "#1B5E20",
        "top_bar_color": "#3E8E41"
    }
}

# 2. Cargar el archivo CSS base
css_content = ""
try:
        # Intento secundario para ruta de Colab
        with open("styles.css") as f:
            css_content = f.read()
except FileNotFoundError:
        st.warning("⚠️ No se encontró el archivo styles.css")


# 3. Sidebar para selección de color
with st.sidebar:
    st.header("🎨 Personalización")
    selected_palette_name = st.selectbox("Elige un estilo:", list(color_palettes.keys()))

    # Se asigna la paleta seleccionada
    selected_palette = color_palettes[selected_palette_name]

# 4. Inyección
st.markdown(
    f"""
    <style>
    :root {{
        --top-bar-color: {selected_palette["top_bar_color"]};
        --sidebar-bg-color: {selected_palette["sidebar_bg_color"]};
        --main-bg-color: {selected_palette["main_bg_color"]};
        --button-color: {selected_palette["button_color"]};
        --text-color: {selected_palette["text_color"]};
    }}
    {css_content}
    </style>
    """,
    unsafe_allow_html=True
)


# --- CARGA DE DATOS ---
@st.cache_data
def load_data():
    filepath = 'economic_indicators_dataset_2010_2023.csv'
    df = pd.read_csv(filepath)
    df = df.rename(columns={
        'Country': 'Country',
        'GDP Growth Rate (%)': 'GDP_Growth',
        'Inflation Rate (%)': 'Inflation_Rate',
        'Unemployment Rate (%)': 'Unemployment_Rate',
        'Interest Rate (%)': 'Interest_Rate'
    })
    df['Date'] = pd.to_datetime(df['Date'])
    return df

#CARGAR LOS DATOS
try:
  df = load_data()
except Exception as e:
  st.error(f"Error al cargar los datos: {e}")
  st.stop()

# --- TÍTULO PRINCIPAL ---
st.title('🌍 Dashboard de Indicadores Económicos Globales')


# --- FILTROS GLOBALES (SIDEBAR) ---
st.sidebar.header("Filtros Globales")

# Filtro 1: Rango de Fechas
min_date = df['Date'].min().date()
max_date = df['Date'].max().date()

date_range = st.sidebar.date_input(
    'Seleccione un rango de fechas:',
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.sidebar.error("Por favor seleccione un rango de dos fechas.")
    st.stop()

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

# Filtro 2: Países
all_countries = sorted(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    'Seleccione uno o más países:',
    options=all_countries,
    default=['Brazil', 'USA', 'China', 'Germany']
)

# --- CONECTAR LOS FILTROS ---
df_filtered = df[
    (df['Date'] >= start_date) & (df['Date'] <= end_date)
]

if selected_countries:
    df_filtered = df_filtered[df_filtered['Country'].isin(selected_countries)]
else:
    st.warning("Seleccione al menos un país en el sidebar para mostrar los datos.")
    st.stop()
# --- FIN DE FILTROS ---


# --- CREACIÓN DE PESTAÑAS (TABS) ---
# AGREGAMOS UNA NUEVA PESTAÑA "Mapa Geográfico"
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "KPIs - Visión General",
    "Evolución Temporal (PIB)",
    "Comparativa por País",
    "Mapa Geográfico",
    "Datos Crudos"
])


# --- Pestaña 1: MÉTRICAS CLAVE ---
with tab1:
    st.header('Métricas Clave del Periodo Seleccionado')
    st.markdown("*(Responden a los filtros del sidebar)*")

    metric_cols = 4
    cols = st.columns(metric_cols)

    avg_gdp_growth = df_filtered['GDP_Growth'].dropna().mean()
    cols[0].metric(
        label="Crecimiento PIB Promedio",
        value=f"{avg_gdp_growth:.2f}%"
    )

    avg_inflation = df_filtered['Inflation_Rate'].dropna().mean()
    cols[1].metric(
        label="Tasa de Inflación Promedio",
        value=f"{avg_inflation:.2f}%"
    )

    avg_unemployment = df_filtered['Unemployment_Rate'].dropna().mean()
    cols[2].metric(
        label="Tasa de Desempleo Promedio",
        value=f"{avg_unemployment:.2f}%"
    )

    avg_interest_rate = df_filtered['Interest_Rate'].dropna().mean()
    cols[3].metric(
        label="Tasa de Interés Promedio",
        value=f"{avg_interest_rate:.2f}%"
    )


# --- Pestaña 2: Gráfica 1 (Evolución PIB) ---
with tab2:
    st.header('Visualizaciones Económicas')

    # --- Gráfica 1: Evolución del Crecimiento del PIB (%) ---
    st.subheader("📊 Evolución del Crecimiento del PIB (%)")
    st.markdown("*(Responde a los filtros del sidebar)*")

    time_grouping = st.selectbox(
        "Agrupar por",
        ["Mensual", "Trimestral", "Anual"],
        index=0,
        key="g1_time_group"
    )

    try:
        if time_grouping == 'Trimestral':
            freq = 'Q'
        elif time_grouping == 'Anual':
            freq = 'Y'
        else: # 'Mensual'
            freq = 'M'

        df_agg = df_filtered.groupby('Country').resample(freq, on='Date').mean(numeric_only=True)
        df_agg = df_agg.reset_index()

        fig_time = px.line(df_agg,
                           x='Date',
                           y='GDP_Growth',
                           color='Country',
                           title=f"Evolución de Crecimiento PIB ({time_grouping})",
                           markers=True)

        fig_time.update_layout(hovermode='x unified')

        if time_grouping == "Anual":
            fig_time.update_xaxes(dtick="M12", tickformat="%Y")
        elif time_grouping == "Trimestral":
            fig_time.update_xaxes(dtick="M3", tickformat="%Y-Q%q")

        st.plotly_chart(fig_time, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo generar la gráfica 1. Error: {e}")

    st.markdown("---")


# --- Pestaña 3: Gráfica 2 (Comparativa Barras) ---
with tab3:
    st.header('Visualizaciones Económicas')

    # --- Gráfica 2: Comparativa de Métricas Promedio por País ---
    st.subheader("📊 Comparativa Promedio por País")
    st.markdown("*(Responde a los filtros del sidebar)*")

    metric_to_plot = st.selectbox(
        "Seleccionar Métrica a Comparar",
        ["Crecimiento PIB", "Tasa de Inflación", "Tasa de Desempleo", "Tasa de Interés"],
        index=0,
        key="bar_metric_select"
    )

    metric_map = {
        "Crecimiento PIB": "GDP_Growth",
        "Tasa de Inflación": "Inflation_Rate",
        "Tasa de Desempleo": "Unemployment_Rate",
        "Tasa de Interés": "Interest_Rate"
    }
    sort_bar_met = metric_map[metric_to_plot]

    try:
        df_bar_agg = df_filtered.groupby('Country')[sort_bar_met].mean().reset_index()
        df_bar_agg = df_bar_agg.sort_values(by=sort_bar_met, ascending=False)

        fig_bar = px.bar(df_bar_agg,
                           x='Country',
                           y=sort_bar_met,
                           color='Country',
                           title=f"Promedio de {metric_to_plot} por País",
                           text=sort_bar_met)

        fig_bar.update_layout(hovermode='x unified')
        fig_bar.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig_bar.update_yaxes(title=f"{metric_to_plot} (%)")

        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo generar la gráfica de barras. Error: {e}")

    st.markdown("---")

# --- Pestaña 5: Mapa Geográfico (NUEVO) ---
with tab5:
    st.header("Mapa Global de Indicadores")
    st.markdown("*(El mapa muestra el **promedio** del Crecimiento del PIB para el periodo seleccionado)*")

    try:
        # 1. Preparar datos para el mapa (Agrupar por país y sacar promedio)
        # Usamos df_filtered para respetar fechas y países seleccionados
        df_map = df_filtered.groupby('Country')['GDP_Growth'].mean().reset_index()

        # 2. Crear Mapa Coroplético
        # locationmode='country names' usa un JSON interno de Plotly para mapear "USA", "Brazil", etc.
        fig_map = px.choropleth(
            df_map,
            locations='Country',
            locationmode='country names',
            color='GDP_Growth',
            hover_name='Country',
            title='Promedio de Crecimiento del PIB por País',
            color_continuous_scale=px.colors.sequential.Plasma, # Escala de color atractiva
            labels={'GDP_Growth': 'Crecimiento PIB (%)'}
        )

        # 3. Ajustar proyección y diseño
        fig_map.update_layout(
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='equirectangular'
            ),
            hovermode='closest' # Muestra info al pasar el mouse
        )

        st.plotly_chart(fig_map, use_container_width=True)

    except Exception as e:
        st.error(f"No se pudo generar el mapa. Error: {e}")


# --- Pestaña 4: Datos Crudos (Movido al final) ---
with tab4:
    st.header("Datos Crudos")
    st.markdown("*(Responde a los filtros del sidebar)*")

    st.subheader("Ver Datos Crudos (Filtrados)")
    st.dataframe(df_filtered)
