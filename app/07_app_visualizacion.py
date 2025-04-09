# 07_app_visualizacion.py

###Librerias importadas####

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import cm
from matplotlib.colors import Normalize
from mplsoccer import Pitch
from scipy.ndimage import gaussian_filter
import ast 
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from fpdf import FPDF
import matplotlib.pyplot as plt
import plotly.io as pio
import tempfile
import base64
from io import BytesIO
from PIL import Image


# Configuración de página

st.set_page_config(page_title="Copa América 2024 - Análisis", layout="wide")
st.title("📊 Panel de control - Copa América 2024")
st.markdown("Visualización interactiva de estadísticas de jugadores y equipos.")

# ---------------------------
# SISTEMA DE LOGIN BÁSICO
# ---------------------------
USUARIOS_VALIDOS = {
    "admin": "1234",
    "usuario1": "clave1",
    "usuario2": "clave2"
}

if "logueado" not in st.session_state:
    st.session_state.logueado = False
if "usuario" not in st.session_state:
    st.session_state.usuario = ""

if not st.session_state.logueado:
    st.title("🔐 Inicio de sesión")
    usuario = st.text_input("Usuario")
    clave = st.text_input("Contraseña", type="password")
    if st.button("Ingresar"):
        if usuario in USUARIOS_VALIDOS and USUARIOS_VALIDOS[usuario] == clave:
            st.session_state.logueado = True
            st.session_state.usuario = usuario  # ✅ Guardamos el usuario conectado
            st.rerun()
        else:
            st.error("Usuario o contraseña incorrectos.")
    st.stop()
else:
    # ✅ Mostrar en todas las secciones
    with st.sidebar:
        st.markdown("----")
        st.markdown(f"👤 **Usuario conectado:** `{st.session_state.usuario}`")
        if st.button("🚪 Cerrar sesión"):
            st.session_state.logueado = False
            st.session_state.usuario = ""
            st.rerun()


# Cargar datos
@st.cache_data
def cargar_eventos():
    pd.read_csv("Data/eventos_copa_america/eventos_copa_america_2024.csv"), low_memory=False
    if 'location' in df.columns:
        df = df[df['location'].notna()].copy()
        df['x'] = df['location'].apply(lambda loc: eval(loc)[0] if isinstance(loc, str) and ',' in loc else None)
        df['y'] = df['location'].apply(lambda loc: eval(loc)[1] if isinstance(loc, str) and ',' in loc else None)
    return df

@st.cache_data
def cargar_descripciones():
    path = "Data/eventos_copa_america/eventos_descripciones.csv"
    if os.path.exists(path):
        descripciones_df = pd.read_csv(path)
        return dict(zip(descripciones_df['Evento'], descripciones_df['Descripción']))
    else:
        return {}

df = cargar_eventos()
desc_eventos = cargar_descripciones()


# Secciones
seccion = st.sidebar.radio("📂 Secciones", ["Eventos por Equipo", "Mapa de calor", "Red de Pases", "Mapa de pases", "Mapa de Remates", "Similares", "Agrupamientos"])

# -----------------------------
# SECCIÓN: EVENTOS POR EQUIPO
# -----------------------------
if seccion == "Eventos por Equipo":
    st.sidebar.subheader("Filtrado")
    equipos = ['Todos'] + sorted(df['team'].dropna().unique().tolist())
    seleccion_equipo = st.sidebar.selectbox("Selecciona un equipo:", equipos)
    tipo_evento = st.sidebar.selectbox("Seleccione el tipo de evento:", sorted(df['type'].dropna().unique()))

    # Filtrado
    if seleccion_equipo != 'Todos':
        df_filtrado = df[(df['team'] == seleccion_equipo) & (df['type'] == tipo_evento)]
    else:
        df_filtrado = df[df['type'] == tipo_evento]

    descripcion_evento = desc_eventos.get(tipo_evento, "")
    st.markdown(f"## Eventos de tipo **{tipo_evento}** - {descripcion_evento}")
    st.markdown(f"<span style='font-size:18px'>🔎 <strong>Cantidad de eventos encontrados:</strong> {len(df_filtrado)}</span>", unsafe_allow_html=True)

    if not df_filtrado.empty:
        # Agrupar por jugador
        resumen = df_filtrado.groupby(['player', 'team']).size().reset_index(name='Cantidad')
        resumen = resumen.sort_values(by='Cantidad', ascending=False).head(10)
        resumen = resumen.rename(columns={'player': 'Jugador', 'team': 'Equipo'})
        resumen['Estadística'] = tipo_evento

        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 5))
        norm = Normalize(vmin=resumen['Cantidad'].min(), vmax=resumen['Cantidad'].max())
        colores = [cm.Blues(norm(v)) for v in resumen['Cantidad']]
        bars = ax.barh(resumen['Jugador'], resumen['Cantidad'], color=colores)
        ax.set_xlabel("Cantidad")
        ax.set_ylabel("Jugador")
        ax.invert_yaxis()
        ax.set_title("Top 10 Jugadores por cantidad de eventos")

        for bar in bars:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    int(bar.get_width()), va='center', fontsize=9)

        st.pyplot(fig)

        # Tabla
        st.markdown("### 📝 Tabla de eventos filtrados")
        st.dataframe(resumen[['Jugador', 'Equipo', 'Estadística', 'Cantidad']].reset_index(drop=True), use_container_width=True)

        # Exportar a PDF
        st.markdown("### 📤 Exportar análisis")
        if st.button("Exportar a PDF", key="exportar_pdf_eventos"):
            with st.spinner("Generando PDF..."):
                import tempfile
                from fpdf import FPDF
                from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

                # Guardar gráfico como imagen temporal
                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(temp_plot.name, format="png")

                # Crear PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Análisis - Eventos por Equipo", ln=True, align="C")
                pdf.ln(10)
                pdf.image(temp_plot.name, w=180)

                # Guardar PDF
                pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                pdf.output(pdf_path)

                st.success("✅ PDF generado correctamente.")
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Descargar PDF", f, file_name="eventos_por_equipo.pdf")

    else:
        st.warning("No se encontraron eventos para el filtro seleccionado.")

        
                # -----------------------------
        # Exportar gráfico a PDF (solo Matplotlib)
        # -----------------------------
       
        if st.button("📄 Exportar gráfico a PDF"):
            with st.spinner("Generando PDF..."):
                # Guardar gráfico actual como imagen PNG
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                    fig.savefig(tmpfile.name, bbox_inches="tight")

                    # Crear PDF e insertar imagen
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt="Top 10 jugadores por eventos", ln=True, align="C")
                    pdf.image(tmpfile.name, x=10, y=30, w=180)

                    # Convertir PDF a base64 para descarga
                    pdf_bytes = pdf.output(dest="S").encode("latin1")
                    b64 = base64.b64encode(pdf_bytes).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="grafico_eventos.pdf">📥 Descargar PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)


# ------------------------------
# Sección: Mapa de Calor
# ------------------------------
elif seccion == "Mapa de calor":
    st.sidebar.subheader("Filtrado")
    equipo = st.sidebar.selectbox("Selecciona un equipo:", sorted(df['team'].dropna().unique()))

    # Filtro de rival
    if 'rival' not in df.columns:
        rivales_por_partido = (
            df[['match_id', 'team']]
            .drop_duplicates()
            .groupby('match_id')['team']
            .apply(lambda x: list(x))
            .to_dict()
        )

        def obtener_rival(row):
            equipos = rivales_por_partido.get(row['match_id'], [])
            return [e for e in equipos if e != row['team']][0] if len(equipos) == 2 else None

        df['rival'] = df.apply(obtener_rival, axis=1)

    df_equipo = df[df['team'] == equipo]
    rivales_disponibles = ['Todos'] + sorted(df_equipo['rival'].dropna().unique().tolist())
    rival = st.sidebar.selectbox("Selecciona un rival (opcional):", rivales_disponibles)

    if rival != 'Todos':
        df_equipo = df_equipo[df_equipo['rival'] == rival]

    jugadores_equipo = df_equipo['player'].dropna().unique()
    jugador = st.sidebar.selectbox("Selecciona un jugador (opcional):", ['Todos'] + sorted(jugadores_equipo))

    titulo = f"Mapa de Calor - {equipo}"
    if rival != 'Todos':
        titulo += f" vs {rival}"
    if jugador != 'Todos':
        titulo += f" ({jugador})"

    st.markdown(f"## {titulo}")
    st.markdown("🎯 **Mientras más intenso el color rojo, mayor concentración de acciones en esa zona del campo.**")

    # Filtrado final
    if jugador != 'Todos':
        df_jugador = df_equipo[df_equipo['player'] == jugador]
    else:
        df_jugador = df_equipo

    if df_jugador.empty:
        st.warning("No se encontraron datos para los filtros seleccionados.")
    else:
        with st.spinner("Generando mapa de calor..."):
            fig, ax = plt.subplots(figsize=(10, 7))
            pitch = Pitch(pitch_type='statsbomb', pitch_color='#eaf4ec', line_color='black')
            pitch.draw(ax=ax)

            x = df_jugador['x']
            y = df_jugador['y']
            sns.kdeplot(x=x, y=y, fill=True, cmap="Reds", bw_adjust=0.5, alpha=0.7, ax=ax, thresh=0.05)

            ax.set_title(f"Mapa de calor de {'el equipo' if jugador == 'Todos' else jugador}", fontsize=14)
            st.pyplot(fig)

        resumen_acciones = df_jugador['type'].value_counts().reset_index()
        resumen_acciones.columns = ['Tipo de Acción', 'Cantidad']
        resumen_acciones = resumen_acciones.head(10)

        st.markdown("### 📋 Acciones más frecuentes")
        st.dataframe(resumen_acciones, use_container_width=True)

# Exportar a PDF
        st.markdown("## 🧾 Exportar análisis")
        if st.button("Exportar a PDF"):
            with st.spinner("Generando PDF..."):
                # Guardar gráfico como imagen temporal
                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(temp_plot.name, bbox_inches="tight")

                # Crear PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt=titulo, ln=True, align="C")
                pdf.image(temp_plot.name, x=10, y=30, w=180)

                # Guardar PDF
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf.output(temp_pdf.name)

                st.success("✅ PDF generado correctamente.")
                with open(temp_pdf.name, "rb") as f:
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=f,
                        file_name=f"{titulo}.pdf",
                        mime="application/pdf"
                    )


# ------------------------------
# Sección: Red de Pases
# ------------------------------
elif seccion == "Red de Pases":
    st.sidebar.subheader("Filtrado")
    equipo = st.sidebar.selectbox("Selecciona un equipo:", sorted(df['team'].dropna().unique()))

    # Crear columna 'rival' si no existe
    if 'rival' not in df.columns:
        with st.spinner("🧠 Calculando rivales..."):
            rivales_por_partido = (
                df[['match_id', 'team']]
                .drop_duplicates()
                .groupby('match_id')['team']
                .apply(list)
                .to_dict()
            )

            def obtener_rival(row):
                equipos = rivales_por_partido.get(row['match_id'], [])
                return [e for e in equipos if e != row['team']][0] if len(equipos) == 2 else None

            df['rival'] = df.apply(obtener_rival, axis=1)

    df_equipo = df[df['team'] == equipo]

    rivales_disponibles = ['Todos'] + sorted(df_equipo['rival'].dropna().unique().tolist())
    rival = st.sidebar.selectbox("Selecciona un rival (opcional):", rivales_disponibles)

    if rival != 'Todos':
        df_equipo = df_equipo[df_equipo['rival'] == rival]

    jugadores_disponibles = sorted(
        pd.concat([df_equipo['player'].dropna(), df_equipo['pass_recipient'].dropna()]).unique()
    )
    jugador = st.sidebar.selectbox("Selecciona un jugador (opcional):", ['Todos'] + jugadores_disponibles)

    if jugador != 'Todos':
        df_passes = df_equipo[
            (df_equipo['type'] == 'Pass') &
            ((df_equipo['player'] == jugador) | (df_equipo['pass_recipient'] == jugador))
        ]
    else:
        df_passes = df_equipo[
            (df_equipo['type'] == 'Pass') & df_equipo['pass_recipient'].notna()
        ]

    if df_passes.empty:
        st.warning("No se encontraron pases con los filtros aplicados.")
    else:
        with st.spinner("🔄 Generando red de pases..."):
            # Posiciones promedio para pases dados
            pos_dador = df_passes.groupby('player')['location'].apply(
                lambda locs: np.mean(
                    [ast.literal_eval(loc) for loc in locs if isinstance(loc, str) and ',' in loc],
                    axis=0
                )
            )

            # Posiciones promedio para pases recibidos
            pos_receptor = df_passes.groupby('pass_recipient')['pass_end_location'].apply(
                lambda locs: np.mean(
                    [ast.literal_eval(loc) for loc in locs if isinstance(loc, str) and ',' in loc],
                    axis=0
                )
            )

            # Combinar ambas posiciones
            todos_jugadores = set(pos_dador.index).union(pos_receptor.index)
            posiciones = {}
            for j in todos_jugadores:
                pos_d = pos_dador.get(j)
                pos_r = pos_receptor.get(j)
                if isinstance(pos_d, np.ndarray) and isinstance(pos_r, np.ndarray):
                    posiciones[j] = np.mean([pos_d, pos_r], axis=0)
                elif isinstance(pos_d, np.ndarray):
                    posiciones[j] = pos_d
                elif isinstance(pos_r, np.ndarray):
                    posiciones[j] = pos_r

            # Conexiones de pase
            conexiones = df_passes.groupby(['player', 'pass_recipient']).size().reset_index(name='cantidad')
            conexiones = conexiones[conexiones['cantidad'] >= 2]

            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 7))
            pitch = Pitch(pitch_type='statsbomb', pitch_color='#eaf4ec', line_color='black')
            pitch.draw(ax=ax)

            for _, row in conexiones.iterrows():
                origen = row['player']
                destino = row['pass_recipient']
                if origen in posiciones and destino in posiciones:
                    start = posiciones[origen]
                    end = posiciones[destino]
                    ax.plot([start[0], end[0]], [start[1], end[1]],
                            lw=row['cantidad'] / 2, color='blue', alpha=0.6)

            for player, pos in posiciones.items():
                ax.scatter(pos[0], pos[1], s=300, color='green', edgecolor='black', zorder=5)
                ax.text(pos[0], pos[1], player, fontsize=8, ha='center', va='center', color='black', zorder=6)

            # Título
            titulo = f"Red de Pases - {equipo}"
            if rival != 'Todos':
                titulo += f" vs {rival}"
            if jugador != 'Todos':
                titulo += f" ({jugador})"
            ax.set_title(titulo, fontsize=14)
            st.pyplot(fig)

            # Explicación del gráfico
            st.markdown("🔵 Las líneas representan los **pases entre jugadores**. El grosor indica cantidad de conexiones. "
                        "Las posiciones se calculan como promedio entre donde dieron y recibieron pases.")

            # Resumen inferior
            pases_dados = df_passes.groupby('player').size().reset_index(name='Pases_Dados')
            pases_recibidos = df_passes.groupby('pass_recipient').size().reset_index(name='Pases_Recibidos')
            resumen = pd.merge(pases_dados, pases_recibidos, left_on='player', right_on='pass_recipient', how='outer')
            resumen['player'] = resumen['player'].combine_first(resumen['pass_recipient'])
            resumen = resumen[['player', 'Pases_Dados', 'Pases_Recibidos']].fillna(0)
            resumen[['Pases_Dados', 'Pases_Recibidos']] = resumen[['Pases_Dados', 'Pases_Recibidos']].astype(int)

            st.markdown("### 📝 Resumen por Jugador")
            st.dataframe(resumen, use_container_width=True)

# Exportar a PDF
        st.markdown("## 🧾 Exportar análisis")
        if st.button("Exportar a PDF"):
            with st.spinner("Generando PDF..."):
                import tempfile
                from fpdf import FPDF

                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(temp_plot.name, bbox_inches="tight")

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Red de Pases", ln=True, align="C")
                pdf.image(temp_plot.name, x=10, y=30, w=180)

                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf.output(temp_pdf.name)

                st.success("✅ PDF generado correctamente.")
                with open(temp_pdf.name, "rb") as f:
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=f,
                        file_name="Red_de_Pases.pdf",
                        mime="application/pdf"
                    )

# ------------------------------
# Sección: Mapa de Pases
# ------------------------------
elif seccion == "Mapa de pases":
    st.markdown("## Mapa de pases")
    st.sidebar.subheader("Filtrado")

    equipo = st.sidebar.selectbox("Selecciona un equipo:", sorted(df['team'].dropna().unique()))

    # Crear columna 'rival' si no existe
    if 'rival' not in df.columns:
        rivales_por_partido = (
            df[['match_id', 'team']]
            .drop_duplicates()
            .groupby('match_id')['team']
            .apply(lambda x: list(x))
            .to_dict()
        )
        def obtener_rival(row):
            equipos = rivales_por_partido.get(row['match_id'], [])
            return [e for e in equipos if e != row['team']][0] if len(equipos) == 2 else None
        df['rival'] = df.apply(obtener_rival, axis=1)

    df_equipo = df[df['team'] == equipo]

    # Filtro de rival
    rivales_disponibles = ['Todos'] + sorted(df_equipo['rival'].dropna().unique().tolist())
    rival = st.sidebar.selectbox("Selecciona un rival (opcional):", rivales_disponibles)
    if rival != 'Todos':
        df_equipo = df_equipo[df_equipo['rival'] == rival]

    # Filtro de jugador
    jugadores_disponibles = sorted(
        pd.concat([df_equipo['player'], df_equipo['pass_recipient']]).dropna().unique()
    )
    jugador = st.sidebar.selectbox("Selecciona un jugador (opcional):", ['Todos'] + jugadores_disponibles)

    # Filtrar los pases
    if jugador != 'Todos':
        df_passes = df_equipo[
            (df_equipo['type'] == 'Pass') & 
            ((df_equipo['player'] == jugador) | (df_equipo['pass_recipient'] == jugador))
        ]
    else:
        df_passes = df_equipo[
            (df_equipo['type'] == 'Pass') &
            (df_equipo['pass_recipient'].notna())
        ]

    if df_passes.empty:
        st.warning("No se encontraron pases con los filtros aplicados.")
    else:
        completados = df_passes[df_passes['pass_outcome'].isna()]
        fallidos = df_passes[df_passes['pass_outcome'].notna()]

        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 7))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#edf7f0', line_color='black')
        pitch.draw(ax=ax)

        # Dibujar pases completados
        for _, row in completados.iterrows():
            try:
                start = ast.literal_eval(row['location']) if isinstance(row['location'], str) else row['location']
                end = ast.literal_eval(row['pass_end_location']) if isinstance(row['pass_end_location'], str) else row['pass_end_location']
                pitch.arrows(start[0], start[1], end[0], end[1], color='blue', ax=ax, alpha=0.7, width=2, headwidth=4, headlength=5)
            except Exception as e:
                print("Error en pase completado:", e)

        # Dibujar pases fallidos
        for _, row in fallidos.iterrows():
            try:
                start = ast.literal_eval(row['location']) if isinstance(row['location'], str) else row['location']
                end = ast.literal_eval(row['pass_end_location']) if isinstance(row['pass_end_location'], str) else row['pass_end_location']
                pitch.arrows(start[0], start[1], end[0], end[1], color='red', ax=ax, alpha=0.7, width=2, headwidth=4, headlength=5, linestyle='dashed')
            except Exception as e:
                print("Error en pase fallido:", e)

               # Leyenda
        linea_completa = Line2D([0], [0], color='blue', lw=2, label='Pases completados')
        linea_fallida = Line2D([0], [0], color='red', lw=2, linestyle='dashed', label='Pases fallidos')
        ax.legend(handles=[linea_completa, linea_fallida], loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

        st.pyplot(fig)

        # Mostrar conteo simple de pases
        st.markdown("### Resumen de Pases")
        resumen = pd.DataFrame({
            "Tipo de pase": ["Completados", "Fallidos"],
            "Cantidad": [len(completados), len(fallidos)]
        })
        st.table(resumen)

               # Exportar a PDF
        st.markdown("## 🧾 Exportar análisis")
        if st.button("Exportar a PDF"):
            with st.spinner("Generando PDF..."):
                import tempfile
                from fpdf import FPDF

                # Guardar gráfico como imagen temporal
                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(temp_plot.name, bbox_inches="tight")

                # Crear PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Mapa de Pases", ln=True, align="C")
                pdf.image(temp_plot.name, x=10, y=30, w=180)

                # Guardar PDF
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf.output(temp_pdf.name)

                st.success("✅ PDF generado correctamente.")
                with open(temp_pdf.name, "rb") as f:
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=f,
                        file_name="Mapa_de_Pases.pdf",
                        mime="application/pdf"
                    )



# ------------------------------
# Sección: Mapa de Remates
# ------------------------------
elif seccion == "Mapa de Remates":
    import ast
    from matplotlib.patches import Circle
    import seaborn as sns

    st.sidebar.subheader("🎯 Filtros")
    # Crear columna 'rival' si no existe
    if 'rival' not in df.columns:
        rivales_por_partido = (
            df[['match_id', 'team']]
            .drop_duplicates()
            .groupby('match_id')['team']
            .apply(lambda x: list(x))
            .to_dict()
        )
        def obtener_rival(row):
            equipos = rivales_por_partido.get(row['match_id'], [])
            return [e for e in equipos if e != row['team']][0] if len(equipos) == 2 else None
        df['rival'] = df.apply(obtener_rival, axis=1)

    # Filtrar solo remates
    df_shots = df[df['type'] == 'Shot'].copy()

    # Conversión de coordenadas
    def convertir_a_lista(loc):
        try:
            return ast.literal_eval(loc) if isinstance(loc, str) else loc
        except:
            return None

    df_shots['location'] = df_shots['location'].apply(convertir_a_lista)

    # Filtros
    equipos = sorted(df_shots['team'].dropna().unique())
    equipo_seleccionado = st.sidebar.selectbox("Selecciona un equipo:", equipos)

    df_equipo = df_shots[df_shots['team'] == equipo_seleccionado]

    rivales_disponibles = ["Todos"] + sorted(df_equipo['rival'].dropna().unique())
    rival = st.sidebar.selectbox("Selecciona un rival (opcional):", rivales_disponibles)
    if rival != "Todos":
        df_equipo = df_equipo[df_equipo['rival'] == rival]

    jugadores_disponibles = sorted(df_equipo['player'].dropna().unique())
    jugador = st.sidebar.selectbox("Selecciona un jugador (opcional):", ['Todos'] + jugadores_disponibles)
    if jugador != "Todos":
        df_equipo = df_equipo[df_equipo['player'] == jugador]

    tiempos = ['Todos', '1T', '2T', 'ET']
    tiempo = st.sidebar.selectbox("Filtrar por tiempo:", tiempos)
    if tiempo != 'Todos':
        df_equipo = df_equipo[df_equipo['period'].map({1: '1T', 2: '2T', 3: 'ET'}) == tiempo]

    if df_equipo.empty:
        st.warning("No hay remates para los filtros seleccionados.")
    else:
        st.markdown("## 📍 Mapa de Remates (mitad ofensiva)")
        st.markdown(
            "🔵 **Gol** - 🔴 **Fallado** - 🔶 **Atajado** - ⚪ **Poste**  \n"
            "⚽ Mientras más grande el círculo, mayor xG del remate."
        )

        fig, ax = plt.subplots(figsize=(10, 7))
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#eaf4ec', line_color='black')
        pitch.draw(ax=ax)
        ax.set_xlim(60, 120)  # Solo mitad ofensiva
        ax.set_ylim(0, 80)

        for _, row in df_equipo.iterrows():
            loc = row['location']
            if isinstance(loc, list) and len(loc) == 2:
                x, y = loc
                outcome = row.get('shot_outcome', 'Unknown')
                xg = row.get('shot_statsbomb_xg', 0.01)
                color = {
                    'Goal': 'blue',
                    'Saved': 'orange',
                    'Off T': 'red',
                    'Post': 'white'
                }.get(outcome, 'gray')

                size = xg * 1000  # Escalar el tamaño del círculo
                radius = size**0.5 / 4
                ax.add_patch(Circle((x, y), radius=radius, color=color, alpha=0.6))

                # Mostrar valor de xG al lado del círculo
                ax.text(x + 1, y, f"{xg:.2f}", fontsize=7, color='black')

        st.pyplot(fig)

        # Dispersión xG vs Resultado
        st.markdown("## 📊 Dispersión de Remates")
        if 'shot_outcome' in df_equipo.columns and 'shot_statsbomb_xg' in df_equipo.columns:
            df_temp = df_equipo[['shot_outcome', 'shot_statsbomb_xg']].dropna()
            fig2, ax2 = plt.subplots()
            sns.boxplot(x='shot_outcome', y='shot_statsbomb_xg', data=df_temp, ax=ax2)
            ax2.set_title("Distribución de xG por resultado del remate")
            ax2.set_xlabel("Resultado")
            ax2.set_ylabel("Expected Goals (xG)")
            st.pyplot(fig2)
            
            
# Exportar a PDF
        st.markdown("## 🧾 Exportar análisis")
        if st.button("Exportar a PDF"):
            with st.spinner("Generando PDF..."):
                import tempfile
                from fpdf import FPDF

                # Guardar gráfico como imagen temporal
                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                fig.savefig(temp_plot.name, bbox_inches="tight")

                # Crear PDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=14)
                pdf.cell(200, 10, txt="Mapa de Remates", ln=True, align="C")
                pdf.image(temp_plot.name, x=10, y=30, w=180)

                # Guardar PDF
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf.output(temp_pdf.name)

                st.success("✅ PDF generado correctamente.")
                with open(temp_pdf.name, "rb") as f:
                    st.download_button(
                        label="📥 Descargar PDF",
                        data=f,
                        file_name="Mapa_de_Remates.pdf",
                        mime="application/pdf"
                    )

# ------------------------------
# Sección: Similares
# ------------------------------
elif seccion == "Similares":
    st.markdown("## 🔍 Búsqueda de jugadores similares")
    st.markdown("Encuentra jugadores con perfiles estadísticos similares durante la Copa América 2024.")

    # Carga de archivos externos
    df_datos = pd.read_excel("C:/Users/Usuario/OneDrive/Documentos/Cursos/Sport Data Campus/Master en Python Avanzado al deporte/Modulo 11/Proyecto Final/Datos/eventos_copa_américa/Copa América 24.xlsx")
    df_metricas = pd.read_excel("C:/Users/Usuario/OneDrive/Documentos/Cursos/Sport Data Campus/Master en Python Avanzado al deporte/Modulo 11/Proyecto Final/Datos/eventos_copa_américa/Métricas.xlsx")

    # Limpiar nombres de columnas (muy importante)
    df_datos.columns = df_datos.columns.str.strip()
    df_metricas.columns = df_metricas.columns.str.strip()

    # Selección de posición
    posiciones_disponibles = df_datos["Pos_principal"].dropna().unique().tolist()
    posicion_seleccionada = st.selectbox("Seleccione una posición:", sorted(posiciones_disponibles))

    # Rango de minutos jugados
    min_minutos = int(df_datos["minutesOnField"].min())
    max_minutos = int(df_datos["minutesOnField"].max())
    rango_minutos = st.slider("Filtrar por minutos jugados:", min_value=min_minutos, max_value=max_minutos, value=(100, max_minutos))

    # Filtrado por posición y rango de minutos
    df_filtrado = df_datos[(df_datos["Pos_principal"] == posicion_seleccionada) &
                           (df_datos["minutesOnField"] >= rango_minutos[0]) &
                           (df_datos["minutesOnField"] <= rango_minutos[1])]

    if posicion_seleccionada not in df_metricas.columns:
        st.error(f"No se encontraron métricas para la posición '{posicion_seleccionada}'")
    else:
        metricas_posicion = df_metricas[posicion_seleccionada].dropna().astype(str).str.strip().tolist()

        columnas_faltantes = [col for col in metricas_posicion if col not in df_filtrado.columns]
        if columnas_faltantes:
            st.error(f"Error: Las siguientes métricas no están en el archivo de datos: {columnas_faltantes}")
        else:
            columnas_utiles = ["Nombre_Completo", "birthAreaName"] + metricas_posicion
            df_metricas_filtradas = df_filtrado[columnas_utiles].dropna().reset_index(drop=True)
            df_metricas_filtradas = df_metricas_filtradas.rename(columns={"Nombre_Completo": "Jugador", "birthAreaName": "Nacionalidad"})

            if df_metricas_filtradas.empty:
                st.warning("No hay jugadores con suficientes datos para esta posición y rango de minutos.")
            else:
                jugador_base = st.selectbox("Selecciona un jugador:", df_metricas_filtradas["Jugador"].tolist())

                from sklearn.preprocessing import StandardScaler
                from sklearn.neighbors import NearestNeighbors

                X = df_metricas_filtradas[metricas_posicion].fillna(0)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                model = NearestNeighbors(n_neighbors=min(6, len(X)), metric='euclidean')
                model.fit(X_scaled)

                idx = df_metricas_filtradas[df_metricas_filtradas["Jugador"] == jugador_base].index[0]
                distancias, indices = model.kneighbors([X_scaled[idx]])

                similares = df_metricas_filtradas.iloc[indices[0]].copy()
                similares["Distancia"] = distancias[0]
                similares = similares[similares["Jugador"] != jugador_base]

                # Calcular similitud (%)
                similares["Similitud (%)"] = (1 - (similares["Distancia"] / similares["Distancia"].max())) * 100
                similares = similares.sort_values("Similitud (%)", ascending=False)

                # Agregar columna N°
                similares.reset_index(drop=True, inplace=True)
                similares.index = similares.index + 1  # Para que arranque en 1

                # Mostrar resultados
                st.success(f"Jugadores similares a **{jugador_base}** en la posición **{posicion_seleccionada}**:")
                st.dataframe(similares[["Jugador", "Nacionalidad", "Similitud (%)"]])
                
                # Exportar a PDF
                st.markdown("### 🧾 Exportar tabla de jugadores similares a PDF")

                if st.button("Exportar a PDF"):
                    with st.spinner("Generando PDF..."):
                        import tempfile
                        from fpdf import FPDF

                        # Crear PDF
                        pdf = FPDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=14)
                        pdf.cell(200, 10, txt=f"Jugadores similares a {jugador_base}", ln=True, align="C")
                        pdf.ln(10)
                        pdf.set_font("Arial", size=10)

                        for i, row in similares.iterrows():
                            texto = f"{i}. {row['Jugador']} ({row['Nacionalidad']}) - Similitud: {row['Similitud (%)']:.2f}%"
                            pdf.multi_cell(0, 10, txt=texto)

                        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        pdf.output(temp_pdf.name)

                        st.success("✅ PDF generado correctamente.")
                        with open(temp_pdf.name, "rb") as f:
                            st.download_button(
                                label="📥 Descargar PDF",
                                data=f,
                                file_name=f"similares_{jugador_base.replace(' ', '_')}.pdf",
                                mime="application/pdf"
                            )
                
                
                
# ------------------------------
# Sección: Agrupamientos
# ------------------------------
elif seccion == "Agrupamientos":
    st.markdown("## 🧠 Agrupamiento de Jugadores según Perfil Estadístico")
    st.markdown("Esta visualización agrupa a los jugadores en 3 clusters según sus métricas durante la Copa América 2024.")

    # Cargar archivo base
    df_datos = pd.read_excel("Datos/eventos_copa_américa/Copa América 24.xlsx")
    df.columns = df.columns.str.strip()

    # Filtro por minutos
    df = df[df["minutesOnField"] >= 150]

    # Agrupar por posición
    posiciones = {
        "ARQ": "ARQ", "DFC": "DFC", "LAT DER": "LAT DER",
        "LAT IZQ": "LAT IZQ", "MED": "MED", "MED MIX": "MED MIX",
        "MED OF": "MED OF", "EXTR": "EXTR", "DEL": "DEL"
    }

    # Cargar métricas por posición
    metricas_pos = pd.read_excel("Datos/eventos_copa_américa/Métricas.xlsx")
    
    # Seleccionar posición del usuario
    pos_sel = st.selectbox("Selecciona una posición", list(posiciones.keys()))
    columnas_metricas = metricas_pos[pos_sel].dropna().tolist()
    columnas_metricas = [col for col in columnas_metricas if col in df.columns]

    # Filtrar jugadores por posición
    df_pos = df[df["Pos_principal"] == posiciones[pos_sel]].copy()

    if df_pos.empty or not columnas_metricas:
        st.warning("No hay datos suficientes para esta posición o métricas no encontradas.")
    else:
        # Normalizar métricas
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        df_pos[columnas_metricas] = scaler.fit_transform(df_pos[columnas_metricas])

        # Clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_pos["cluster"] = kmeans.fit_predict(df_pos[columnas_metricas])

        # Mostrar top 5 por cluster con nombre del jugador
        st.markdown("### 🧑‍🏫 Top 5 jugadores por Cluster")
        for c in sorted(df_pos["cluster"].unique()):
            st.markdown(f"#### Cluster {c}")
            cols = ["Nombre_Completo", "minutesOnField", "cluster"] + columnas_metricas
            st.dataframe(df_pos[df_pos["cluster"] == c][cols].sort_values(by="minutesOnField", ascending=False).head(5), use_container_width=True)

        # Exportar a PDF
        st.markdown("### 🧾 Exportar análisis a PDF")
        if st.button("Exportar a PDF", key="exportar_agrupamientos"):
            with st.spinner("Generando PDF..."):
                try:
                    from fpdf import FPDF
                    import tempfile

                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=14)
                    pdf.cell(200, 10, txt=f"Agrupamiento - {pos_sel}", ln=True, align="C")
                    pdf.ln(10)

                    for c in sorted(df_pos["cluster"].unique()):
                        pdf.set_font("Arial", style="B", size=11)
                        pdf.cell(200, 8, txt=f"Clúster {c}", ln=True)
                        pdf.set_font("Arial", size=9)
                        top5 = df_pos[df_pos["cluster"] == c].sort_values(by="minutesOnField", ascending=False).head(5)
                        for _, row in top5.iterrows():
                            texto = f"{row['Nombre_Completo']} - Min: {int(row['minutesOnField'])}"
                            for m in columnas_metricas:
                                texto += f" | {m}: {row[m]:.2f}"
                            pdf.multi_cell(0, 6, txt=texto)
                        pdf.ln(3)

                    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    pdf.output(temp_pdf.name)

                    st.success("✅ PDF generado correctamente.")
                    with open(temp_pdf.name, "rb") as f:
                        st.download_button(
                            label="📥 Descargar PDF",
                            data=f,
                            file_name=f"agrupamientos_completo_{pos_sel}.pdf",
                            mime="application/pdf"
                        )
                except Exception as e:
                    st.error(f"Ocurrió un error al generar el PDF: {e}")
