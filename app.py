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


def inject_theme() -> None:
    """Carga el CSS de PatternFly y lo combina con la paleta/tipografia de un
    design system tipo "Data-Dense Dashboard" (generado con la skill
    ui-ux-pro-max) reestilizando los widgets nativos de Streamlit (no son
    componentes React de PatternFly, sino una aproximacion visual via CSS)."""
    st.markdown(
        f"""
        <link rel="stylesheet" href="{PATTERNFLY_CSS_CDN}">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --color-primary: #3B82F6;
                --color-secondary: #60A5FA;
                --color-cta: #F97316;
                --color-cta-hover: #EA670C;
                --color-bg: #F8FAFC;
                --color-surface: #FFFFFF;
                --color-border: #E2E8F0;
                --color-text: #1E293B;
                --color-text-muted: #64748B;
            }}
            html, body, [class*="css"] {{
                font-family: "Fira Sans", "Red Hat Text", Arial, sans-serif;
                color: var(--color-text);
            }}
            [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
                background-color: var(--color-bg);
            }}
            .pf-page-header {{
                border-bottom: 1px solid var(--color-border);
                padding-bottom: 0.75rem;
                margin-bottom: 1rem;
            }}
            .pf-page-header h1 {{
                color: var(--color-text);
                font-weight: 700;
                margin-bottom: 0.25rem;
            }}
            .pf-page-header p {{
                color: var(--color-text-muted);
                margin: 0;
            }}
            .pf-section-title {{
                color: var(--color-text);
                font-weight: 600;
                font-size: 1.1rem;
                border-left: 4px solid var(--color-primary);
                padding-left: 0.6rem;
                margin: 0.25rem 0 0.75rem 0;
            }}

            /* Tarjetas: st.container(border=True) y st.expander quedan con
               look de panel de dashboard (fondo blanco, borde suave, sombra). */
            div[data-testid="stVerticalBlockBorderWrapper"],
            div[data-testid="stExpander"] {{
                background-color: var(--color-surface);
                border: 1px solid var(--color-border) !important;
                border-radius: 10px !important;
                box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06);
            }}

            div.stButton > button {{
                background-color: var(--color-cta);
                color: #ffffff;
                border: 1px solid var(--color-cta);
                border-radius: 6px;
                font-weight: 600;
                padding: 0.55rem 1.25rem;
                cursor: pointer;
                transition: background-color 150ms ease;
            }}
            div.stButton > button:hover {{
                background-color: var(--color-cta-hover);
                border-color: var(--color-cta-hover);
                color: #ffffff;
            }}
            div.stButton > button:focus-visible,
            div[data-baseweb="select"] > div:focus-within,
            div[data-testid="stTextInput"] input:focus-visible,
            div[data-testid="stNumberInput"] input:focus-visible {{
                outline: 2px solid var(--color-primary) !important;
                outline-offset: 2px !important;
            }}
            div[data-baseweb="select"] > div,
            div[data-testid="stNumberInput"] input,
            div[data-testid="stTextInput"] input {{
                border-radius: 6px !important;
                border-color: var(--color-border) !important;
            }}

            .pf-grid-header {{
                background-color: #F1F5F9;
                border-top: 1px solid var(--color-border);
                border-bottom: 1px solid var(--color-border);
                padding: 0.4rem 0;
                font-weight: 600;
                color: var(--color-text);
            }}
            .pf-grid-row {{
                border-bottom: 1px solid #F1F5F9;
                padding: 0.25rem 0;
                font-family: "Fira Code", monospace;
                font-size: 0.9rem;
                color: var(--color-text);
            }}

            /* Resalta la fila completa al pasar el mouse (cada fila de la
               grilla es un st.columns(), que Streamlit renderiza como un
               stHorizontalBlock). */
            div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"] {{
                transition: background-color 150ms ease;
                border-radius: 6px;
            }}
            div[data-testid="stExpander"] div[data-testid="stHorizontalBlock"]:hover {{
                background-color: #EFF6FF;
            }}

            /* Evita que el layout salte al alternar entre st.info/st.success/st.warning */
            div[data-testid="stAlert"] {{
                min-height: 52px;
                display: flex;
                align-items: center;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()

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

with st.container(border=True):
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
