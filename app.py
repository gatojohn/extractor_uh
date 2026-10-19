"""Extractor UH - v0.

Aplicacion Streamlit que permite dibujar un poligono cerrado sobre un mapa y
simular la deteccion, mediante IA, de unidades habitacionales dentro del
area seleccionada.
"""

import random

import folium
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import Point, Polygon
from streamlit_folium import st_folium

st.set_page_config(page_title="Extractor UH", page_icon="\U0001F3E0", layout="wide")

PATTERNFLY_CSS_CDN = "https://unpkg.com/@patternfly/patternfly@5.4.1/patternfly.min.css"


def inject_patternfly_theme() -> None:
    """Carga el CSS de PatternFly y reestiliza los widgets nativos de Streamlit
    con los tokens de color/tipografia de PatternFly (no son componentes React
    de PatternFly, sino una aproximacion visual via CSS)."""
    st.markdown(
        f"""
        <link rel="stylesheet" href="{PATTERNFLY_CSS_CDN}">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Red+Hat+Text:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {{
                font-family: "Red Hat Text", "RedHatText", Overpass, Arial, sans-serif;
            }}
            .pf-page-header {{
                border-bottom: 1px solid #d2d2d2;
                padding-bottom: 0.75rem;
                margin-bottom: 1rem;
            }}
            .pf-page-header h1 {{
                color: #151515;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }}
            .pf-page-header p {{
                color: #6a6e73;
                margin: 0;
            }}
            .pf-section-title {{
                color: #151515;
                font-weight: 600;
                font-size: 1.1rem;
                border-left: 4px solid #0066cc;
                padding-left: 0.6rem;
                margin: 1.25rem 0 0.75rem 0;
            }}
            div.stButton > button {{
                background-color: #0066cc;
                color: #ffffff;
                border: 1px solid #0066cc;
                border-radius: 3px;
                font-weight: 500;
                padding: 0.5rem 1.2rem;
            }}
            div.stButton > button:hover {{
                background-color: #004c99;
                border-color: #004c99;
                color: #ffffff;
            }}
            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextInput"] input {{
                border-radius: 3px !important;
                border-color: #8a8d90 !important;
            }}
            .pf-grid-header {{
                background-color: #f0f0f0;
                border-top: 1px solid #d2d2d2;
                border-bottom: 1px solid #d2d2d2;
                padding: 0.4rem 0;
                font-weight: 600;
                color: #151515;
            }}
            .pf-grid-row {{
                border-bottom: 1px solid #ededed;
                padding: 0.25rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_patternfly_theme()

st.markdown(
    """
    <div class="pf-page-header">
        <h1>Extractor UH</h1>
        <p>Dibuje un pol&iacute;gono sobre el mapa y simule la identificaci&oacute;n de
        unidades habitacionales mediante un modelo de IA.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "units" not in st.session_state:
    st.session_state.units = []
if "polygon_coords" not in st.session_state:
    st.session_state.polygon_coords = None
if "detection_run_id" not in st.session_state:
    st.session_state.detection_run_id = 0

with st.sidebar:
    st.markdown("### Configuración del mapa")
    center_lat = st.number_input("Latitud centro", value=40.4168, format="%.6f")
    center_lon = st.number_input("Longitud centro", value=-3.7038, format="%.6f")
    zoom = st.slider("Zoom inicial", min_value=2, max_value=20, value=17)
    map_height = st.slider("Altura del mapa (px)", min_value=400, max_value=1000, value=750, step=50)

st.markdown('<div class="pf-section-title">1. Dibuje el área de interés</div>', unsafe_allow_html=True)
st.caption("Use la herramienta de polígono del mapa (icono de pentágono) para trazar un área cerrada.")

folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=True)
Draw(
    export=False,
    draw_options={
        "polygon": True,
        "rectangle": False,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polyline": False,
    },
    edit_options={"edit": True, "remove": True},
).add_to(folium_map)

map_figure = folium.Figure(height=map_height)
map_figure.add_child(folium_map)

# returned_objects se limita a los dibujos: por defecto st_folium tambien
# devuelve center/zoom/bounds, que cambian con cada pan o zoom y disparan un
# rerun completo (remontando el componente), lo que causa el efecto visual
# de "se agranda y se achica". Al no observar esos campos, el mapa solo
# vuelve a correr el script cuando el usuario efectivamente dibuja algo.
map_state = st_folium(
    map_figure,
    height=map_height,
    use_container_width=True,
    returned_objects=["all_drawings", "last_active_drawing"],
    key="main_map",
)

drawing = None
if map_state:
    drawing = map_state.get("last_active_drawing")
    if not drawing and map_state.get("all_drawings"):
        drawing = map_state["all_drawings"][-1]

if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
    st.session_state.polygon_coords = drawing["geometry"]["coordinates"][0]

col_button, col_status = st.columns([1, 3])
with col_button:
    run_detection = st.button("Obtener unidades habitacionales", type="primary", use_container_width=True)
with col_status:
    if st.session_state.polygon_coords:
        st.success(f"Polígono capturado con {len(st.session_state.polygon_coords)} vértices.")
    else:
        st.info("Dibuje un polígono cerrado sobre el mapa para habilitar el análisis.")


def simulate_ai_detection(coords):
    """Simula la salida de un modelo de IA que identifica unidades
    habitacionales dentro del polígono dibujado. Por ahora genera inmuebles
    de ejemplo distribuidos aleatoriamente dentro del área; en una versión
    futura esta función se reemplaza por la inferencia del modelo real."""
    polygon = Polygon([(lon, lat) for lon, lat in coords])
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    n_units = random.randint(4, 10)
    units = []
    attempts = 0
    while len(units) < n_units and attempts < n_units * 50:
        attempts += 1
        lon = random.uniform(min_lon, max_lon)
        lat = random.uniform(min_lat, max_lat)
        if polygon.contains(Point(lon, lat)):
            tipo = random.choice(["Casa", "Edificio"])
            index = len(units) + 1
            units.append(
                {
                    "id": index,
                    "nombre": f"Inmueble {index}",
                    "lat": round(lat, 6),
                    "lon": round(lon, 6),
                    "tipo": tipo,
                    "pisos": random.randint(2, 15) if tipo == "Edificio" else None,
                    "apartamentos": random.randint(2, 40) if tipo == "Edificio" else None,
                }
            )
    return units


if run_detection:
    if not st.session_state.polygon_coords:
        st.warning("Primero dibuje un polígono cerrado sobre el mapa.")
    else:
        with st.spinner("Analizando el área seleccionada con el modelo de IA (simulado)..."):
            st.session_state.units = simulate_ai_detection(st.session_state.polygon_coords)
            st.session_state.detection_run_id += 1

with st.expander("2. Unidades habitacionales detectadas", expanded=True):
    if not st.session_state.units:
        st.caption("Aún no se ha ejecutado el análisis.")
    else:
        run_id = st.session_state.detection_run_id
        column_widths = [0.5, 2, 2.3, 1.3, 1, 1]

        header_cols = st.columns(column_widths)
        for col, label in zip(
            header_cols,
            ["#", "Nombre del inmueble", "Coordenadas (lat, lon)", "Tipo", "Pisos", "Apartamentos"],
        ):
            col.markdown(f"<div class='pf-grid-header'>{label}</div>", unsafe_allow_html=True)

        tipo_options = ["Casa", "Edificio"]

        for unit in st.session_state.units:
            row_cols = st.columns(column_widths)
            key_prefix = f"{run_id}_{unit['id']}"

            row_cols[0].markdown(f"<div class='pf-grid-row'>{unit['id']}</div>", unsafe_allow_html=True)

            unit["nombre"] = row_cols[1].text_input(
                "Nombre", value=unit["nombre"], key=f"nombre_{key_prefix}", label_visibility="collapsed"
            )

            row_cols[2].markdown(
                f"<div class='pf-grid-row'>{unit['lat']}, {unit['lon']}</div>", unsafe_allow_html=True
            )

            unit["tipo"] = row_cols[3].selectbox(
                "Tipo",
                tipo_options,
                index=tipo_options.index(unit["tipo"]),
                key=f"tipo_{key_prefix}",
                label_visibility="collapsed",
            )

            if unit["tipo"] == "Edificio":
                unit["pisos"] = row_cols[4].number_input(
                    "Pisos",
                    min_value=1,
                    value=unit["pisos"] or 1,
                    step=1,
                    key=f"pisos_{key_prefix}",
                    label_visibility="collapsed",
                )
                unit["apartamentos"] = row_cols[5].number_input(
                    "Apartamentos",
                    min_value=1,
                    value=unit["apartamentos"] or 1,
                    step=1,
                    key=f"apartamentos_{key_prefix}",
                    label_visibility="collapsed",
                )
            else:
                unit["pisos"] = None
                unit["apartamentos"] = None
                row_cols[4].markdown("<div class='pf-grid-row'>&mdash;</div>", unsafe_allow_html=True)
                row_cols[5].markdown("<div class='pf-grid-row'>&mdash;</div>", unsafe_allow_html=True)
