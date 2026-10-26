"""Extractor UH.

Aplicacion Streamlit que permite dibujar un poligono cerrado sobre un mapa y
detectar, con un modelo de IA de Hugging Face, las unidades habitacionales
dentro del area seleccionada.
"""

import html
import math
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import folium
import numpy as np
import onnxruntime as ort
import requests
import streamlit as st
from folium.plugins import Draw
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw
from scipy import ndimage
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
from streamlit_folium import st_folium

st.set_page_config(page_title="Extractor UH", page_icon="\U0001F3E0", layout="wide")

# Modelo U-Net de segmentacion de huellas de edificios. Solo se publica en ONNX
# (para la libreria JavaScript GeoAI.js), por lo que transformers.pipeline no
# puede cargarlo: se ejecuta con onnxruntime replicando el pre/postproceso de
# GeoAI.js (parches RGB 256x256 en [0, 1], umbral 0.3, componentes conexas).
MODELO = "geobase/building-footprint-segmentation"
MODELO_ARCHIVO = "onnx/model_quantized.onnx"
ESRI_TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE = 256
ZOOM_ANALISIS = 18  # ~0,6 m/pixel: en pruebas segmenta mejor que 17 o 19
AREA_MAXIMA_KM2 = 0.5
UMBRAL_CONFIANZA = 0.3
AREA_MINIMA_M2 = 25
AREA_EDIFICIO_M2 = 300  # heuristica: techos mayores se proponen como "Edificio"


def inject_theme() -> None:
    """Tema oscuro/neon (Dark Mode OLED + acentos Cyberpunk, segun la skill
    ui-ux-pro-max). Todos los colores se fuerzan explicitamente para que el
    resultado sea el mismo con el tema claro u oscuro de Streamlit."""
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;500;600;700&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {
                --bg: #0A0E27;
                --bg-sidebar: #070A1C;
                --surface: #11163A;
                --surface-2: #171E4A;
                --border: #273170;
                --text: #E6EAFF;
                --text-muted: #9AA3C7;
                --cyan: #00E5FF;
                --magenta: #FF2BD6;
                --green: #2DFFB3;
                --amber: #FFC14D;
            }

            /* Base */
            .stApp, [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(1200px 600px at 85% -10%, rgba(255, 43, 214, 0.10), transparent 60%),
                    radial-gradient(900px 500px at -10% 10%, rgba(0, 229, 255, 0.08), transparent 60%),
                    var(--bg);
                color: var(--text);
                font-family: "Fira Sans", Arial, sans-serif;
            }
            header[data-testid="stHeader"] { background: transparent; }
            .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3 {
                color: var(--text);
                font-family: "Fira Sans", Arial, sans-serif;
            }
            [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
                color: var(--text-muted) !important;
            }
            section[data-testid="stSidebar"] {
                background: var(--bg-sidebar);
                border-right: 1px solid var(--border);
            }
            section[data-testid="stSidebar"] h3 {
                color: var(--cyan);
                font-family: "Fira Code", monospace;
                font-size: 0.85rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }

            /* Encabezado */
            .uh-header { padding: 0.25rem 0 1rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1.25rem; }
            .uh-kicker {
                font-family: "Fira Code", monospace; font-size: 0.75rem; letter-spacing: 0.14em;
                text-transform: uppercase; color: var(--cyan);
            }
            .uh-title {
                font-size: 2.4rem; font-weight: 700; margin: 0.2rem 0; line-height: 1.15;
                background: linear-gradient(90deg, var(--cyan), var(--magenta));
                -webkit-background-clip: text; background-clip: text; color: transparent;
                filter: drop-shadow(0 0 12px rgba(0, 229, 255, 0.25));
            }
            .uh-subtitle { color: var(--text-muted); margin: 0; }
            .uh-section-title {
                font-family: "Fira Code", monospace; font-size: 0.85rem; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; color: var(--text);
                border-left: 3px solid var(--cyan); padding-left: 0.6rem; margin: 0.25rem 0 0.5rem 0;
            }

            /* Paneles */
            div[data-testid="stVerticalBlock"]:has(> div[data-testid="stElementContainer"] .uh-section-title),
            div[data-testid="stExpander"] details {
                background: var(--surface) !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
            }
            div[data-testid="stExpander"] summary {
                background: var(--surface) !important;
                color: var(--text) !important;
                border-radius: 12px;
            }
            div[data-testid="stExpander"] summary p {
                font-family: "Fira Code", monospace; font-size: 0.85rem; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; color: var(--text) !important;
            }
            div[data-testid="stExpander"] summary:hover p,
            div[data-testid="stExpander"] summary svg { color: var(--cyan) !important; fill: var(--cyan) !important; }
            div[data-testid="stExpanderDetails"] { background: var(--surface) !important; }

            /* Mapa */
            iframe[title*="st_folium"] {
                border-radius: 10px;
                box-shadow: 0 0 0 1px var(--border), 0 0 28px rgba(0, 229, 255, 0.12);
            }

            /* Leyenda de los techos dibujados sobre el mapa */
            .uh-legend {
                display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 1.25rem;
                font-size: 0.85rem; color: var(--text-muted); margin: 0.25rem 0 0.5rem 0;
            }
            .uh-legend-item { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--text); }
            .uh-legend-item i { width: 14px; height: 14px; border-radius: 3px; display: inline-block; }
            .uh-legend-casa { background: rgba(0, 229, 255, 0.35); border: 1.5px solid var(--cyan); }
            .uh-legend-edificio { background: rgba(255, 43, 214, 0.35); border: 1.5px solid var(--magenta); }
            .uh-legend-area { border: 2px dashed var(--amber); }

            /* Boton principal: misma altura que el indicador de estado */
            div.stButton > button {
                min-height: 48px; width: 100%;
                background: linear-gradient(90deg, var(--cyan), var(--magenta));
                border: none; border-radius: 10px; cursor: pointer;
                transition: box-shadow 200ms ease, filter 200ms ease;
            }
            div.stButton > button p {
                color: #0A0E27 !important; font-weight: 700; white-space: nowrap;
            }
            div.stButton > button:hover {
                box-shadow: 0 0 18px rgba(0, 229, 255, 0.45), 0 0 28px rgba(255, 43, 214, 0.30);
                filter: brightness(1.08);
            }
            div.stButton > button:focus-visible {
                outline: 2px solid var(--text) !important; outline-offset: 3px;
            }

            /* Indicador de estado */
            .uh-status {
                min-height: 48px; display: flex; align-items: center; gap: 0.6rem;
                padding: 0 1rem; border-radius: 10px; border: 1px solid;
                font-size: 0.95rem; background: var(--surface-2);
            }
            .uh-status::before {
                content: ""; width: 8px; height: 8px; border-radius: 50%;
                background: currentColor; box-shadow: 0 0 8px currentColor; flex: none;
            }
            .uh-status span { color: var(--text); }
            .uh-status-info { color: var(--cyan); border-color: rgba(0, 229, 255, 0.35); }
            .uh-status-success { color: var(--green); border-color: rgba(45, 255, 179, 0.35); }
            .uh-status-warning { color: var(--amber); border-color: rgba(255, 193, 77, 0.40); }

            /* Inputs, selects y number inputs */
            div[data-testid="stNumberInputContainer"],
            div[data-testid="stTextInputRootElement"],
            div[data-testid="stSelectbox"] div[role="group"] {
                background: var(--surface-2) !important;
                border: 1px solid var(--border) !important;
                border-radius: 8px !important;
            }
            div[data-testid="stNumberInputContainer"]:focus-within,
            div[data-testid="stTextInputRootElement"]:focus-within,
            div[data-testid="stSelectbox"] div[role="group"]:focus-within {
                border-color: var(--cyan) !important;
                box-shadow: 0 0 0 1px var(--cyan), 0 0 10px rgba(0, 229, 255, 0.35);
            }
            .stApp input {
                color: var(--text) !important; -webkit-text-fill-color: var(--text);
                caret-color: var(--cyan); font-family: "Fira Code", monospace;
            }
            div[data-testid="stSelectbox"] svg { color: var(--text) !important; }
            div[data-testid="stNumberInputContainer"] button {
                background: var(--surface-2) !important; color: var(--text) !important;
            }
            div[data-testid="stNumberInputContainer"] button:hover {
                background: var(--border) !important; color: var(--cyan) !important;
            }
            [role="listbox"], [role="listbox"] [role="option"] {
                background: var(--surface-2) !important; color: var(--text) !important;
            }
            [role="listbox"] [role="option"]:hover, [role="listbox"] [role="option"][aria-selected="true"] {
                background: var(--border) !important; color: var(--cyan) !important;
            }

            /* Sliders: el rojo por defecto de Streamlit se rota a cian (el texto
               blanco del valor no se ve afectado por hue-rotate). */
            div[data-testid="stSlider"] div[role="group"] { filter: hue-rotate(186deg) saturate(1.4); }
            div[data-testid="stSlider"] [data-testid="stSliderTickBarMin"],
            div[data-testid="stSlider"] [data-testid="stSliderTickBarMax"] { color: var(--text-muted); }

            /* Grilla */
            .uh-grid-header {
                font-family: "Fira Code", monospace; font-size: 0.75rem; font-weight: 600;
                letter-spacing: 0.08em; text-transform: uppercase; color: var(--cyan);
                padding: 0.5rem 0; border-bottom: 1px solid rgba(0, 229, 255, 0.35);
            }
            .uh-grid-cell {
                font-family: "Fira Code", monospace; font-size: 0.9rem; color: var(--text);
                min-height: 40px; display: flex; align-items: center;
            }
            .uh-grid-muted { color: var(--text-muted); }
            /* Streamlit aplica margin-bottom: -16px a cada markdown, lo que
               desalinea las celdas de texto respecto de los inputs de la fila. */
            div[data-testid="stMarkdownContainer"]:has(.uh-grid-cell),
            div[data-testid="stMarkdownContainer"]:has(.uh-grid-header) { margin-bottom: 0; }
            div[data-testid="stExpanderDetails"] div[data-testid="stHorizontalBlock"] {
                border-radius: 8px; transition: background-color 150ms ease;
            }
            div[data-testid="stExpanderDetails"] div[data-testid="stHorizontalBlock"]:hover {
                background: rgba(0, 229, 255, 0.06);
            }

            @media (prefers-reduced-motion: reduce) {
                * { transition: none !important; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


LEAFLET_DARK_CSS = """
<style>
    .leaflet-container { background: #0A0E27 !important; }
    .uh-dark-tiles { filter: invert(1) hue-rotate(200deg) brightness(0.85) contrast(0.9) saturate(0.6); }
    .leaflet-bar a, .leaflet-control-layers, .leaflet-draw-actions a {
        background-color: #11163A !important; color: #E6EAFF !important;
        border-color: #273170 !important;
    }
    .leaflet-bar a:hover, .leaflet-draw-actions a:hover { background-color: #273170 !important; color: #00E5FF !important; }
    .leaflet-bar, .leaflet-control-layers { border: 1px solid #273170 !important; box-shadow: 0 0 12px rgba(0, 229, 255, 0.18) !important; }
    .leaflet-draw-toolbar a { filter: invert(1) hue-rotate(180deg); background-color: #E6EAFF !important; }
    .leaflet-control-layers-expanded { color: #E6EAFF !important; padding: 8px 10px; }
    .leaflet-control-attribution, .leaflet-control-scale-line {
        background: rgba(10, 14, 39, 0.75) !important; color: #9AA3C7 !important; border-color: #273170 !important;
    }
    .leaflet-control-attribution a { color: #00E5FF !important; }
    .leaflet-tooltip {
        background: #11163A; color: #E6EAFF; border: 1px solid #273170;
        font-family: "Fira Code", monospace; font-size: 12px;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.25);
    }
    .leaflet-tooltip-top:before { border-top-color: #273170; }
</style>
"""


# Mismos tonos que --cyan / --magenta: Leaflet aplica los colores como atributos
# SVG, donde no se pueden usar variables CSS.
COLOR_TIPO = {"Casa": "#00E5FF", "Edificio": "#FF2BD6"}
COLOR_AREA_DIBUJADA = "#FFC14D"


def capa_techos(units, run_id) -> folium.FeatureGroup:
    capa = folium.FeatureGroup(name="Techos detectados")
    for unit in units:
        # Se leen los valores actuales de la grilla: el mapa se dibuja antes que ella.
        clave = f"{run_id}_{unit['id']}"
        nombre = st.session_state.get(f"nombre_{clave}", unit["nombre"])
        tipo = st.session_state.get(f"tipo_{clave}", unit["tipo"])
        color = COLOR_TIPO[tipo]
        folium.Polygon(
            locations=unit["huella"],
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.35,
            tooltip=f"{unit['id']}. {html.escape(nombre)} · {tipo} · {unit['area_m2']} m²",
        ).add_to(capa)
    return capa


def status(kind: str, text: str) -> None:
    st.markdown(
        f'<div class="uh-status uh-status-{kind}" role="status" aria-live="polite">'
        f"<span>{html.escape(text)}</span></div>",
        unsafe_allow_html=True,
    )


class AnalisisError(Exception):
    """Error con un mensaje listo para mostrar al usuario."""


@st.cache_resource(show_spinner=False)
def cargar_modelo() -> ort.InferenceSession:
    ruta = hf_hub_download(MODELO, MODELO_ARCHIVO)
    return ort.InferenceSession(ruta, providers=["CPUExecutionProvider"])


def lonlat_a_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    n = TILE * 2**zoom
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def pixel_a_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    n = TILE * 2**zoom
    lon = x / n * 360 - 180
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def formato_es(valor: float, decimales: int) -> str:
    return f"{valor:,.{decimales}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def area_m2(polygon: Polygon) -> float:
    kx = 111_320 * math.cos(math.radians(polygon.centroid.y))
    return Polygon([(lon * kx, lat * 110_540) for lon, lat in polygon.exterior.coords]).area


def descargar_mosaico(tx0: int, ty0: int, cols: int, rows: int, zoom: int) -> Image.Image:
    def descargar(tile):
        tx, ty = tile
        respuesta = requests.get(ESRI_TILE_URL.format(z=zoom, x=tx, y=ty), timeout=15)
        respuesta.raise_for_status()
        return tile, Image.open(BytesIO(respuesta.content)).convert("RGB")

    tiles = [(tx0 + c, ty0 + r) for r in range(rows) for c in range(cols)]
    mosaico = Image.new("RGB", (cols * TILE, rows * TILE))
    with ThreadPoolExecutor(max_workers=8) as pool:
        for (tx, ty), imagen in pool.map(descargar, tiles):
            mosaico.paste(imagen, ((tx - tx0) * TILE, (ty - ty0) * TILE))
    return mosaico


def contorno_techo(mascara: np.ndarray) -> list[tuple[float, float]]:
    """Contorno exterior (en pixeles de la mascara) de un techo: une los tramos
    horizontales de pixeles de cada fila y simplifica el borde escalonado."""
    tramos = []
    for fila, valores in enumerate(mascara):
        columnas = np.flatnonzero(valores)
        if columnas.size == 0:
            continue
        cortes = np.flatnonzero(np.diff(columnas) > 1)
        inicios = np.r_[columnas[0], columnas[cortes + 1]]
        fines = np.r_[columnas[cortes], columnas[-1]] + 1
        tramos.extend(box(c0, fila, c1, fila + 1) for c0, c1 in zip(inicios, fines))
    forma = unary_union(tramos).simplify(1.0)
    if forma.geom_type == "MultiPolygon":
        forma = max(forma.geoms, key=lambda g: g.area)
    return list(forma.exterior.coords)


def detect_units(coords):
    """Detecta con el modelo de Hugging Face las huellas de edificios dentro
    del poligono (imagen satelital de Esri) y devuelve una unidad por cada
    edificio. El tipo se estima por el tamano del techo; pisos y apartamentos
    los completa el usuario."""
    polygon = Polygon(coords)
    area_km2 = area_m2(polygon) / 1e6
    if area_km2 > AREA_MAXIMA_KM2:
        raise AnalisisError(
            f"La zona es demasiado grande ({formato_es(area_km2, 2)} km²). Acérquese en el mapa y "
            f"dibuje un área de hasta {formato_es(AREA_MAXIMA_KM2, 1)} km²."
        )

    zoom = ZOOM_ANALISIS
    vertices_px = [lonlat_a_pixel(lon, lat, zoom) for lon, lat in coords]
    xs = [x for x, _ in vertices_px]
    ys = [y for _, y in vertices_px]
    tx0, ty0 = int(min(xs) // TILE), int(min(ys) // TILE)
    cols, rows = int(max(xs) // TILE) - tx0 + 1, int(max(ys) // TILE) - ty0 + 1

    try:
        mosaico = descargar_mosaico(tx0, ty0, cols, rows, zoom)
    except requests.RequestException as error:
        raise AnalisisError(
            "No se pudo descargar la imagen satelital de la zona. Revise la conexión y pruebe de nuevo."
        ) from error

    pixeles = np.asarray(mosaico, dtype=np.float32) / 255.0
    parches = pixeles.reshape(rows, TILE, cols, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(-1, TILE, TILE, 3)
    try:
        salida = cargar_modelo().run(None, {"input_1": parches})[0]
    except Exception as error:
        raise AnalisisError(
            "El modelo de IA no pudo analizar la imagen en este momento. Pruebe de nuevo en un minuto."
        ) from error
    probabilidad = salida.reshape(rows, cols, TILE, TILE).transpose(0, 2, 1, 3).reshape(rows * TILE, cols * TILE)

    origen_x, origen_y = tx0 * TILE, ty0 * TILE
    mascara_poligono = Image.new("L", mosaico.size, 0)
    ImageDraw.Draw(mascara_poligono).polygon(
        [(x - origen_x, y - origen_y) for x, y in vertices_px], fill=1
    )
    edificios = ndimage.binary_opening(probabilidad >= UMBRAL_CONFIANZA) & np.asarray(mascara_poligono, dtype=bool)
    etiquetas, total = ndimage.label(edificios)
    if total == 0:
        return []

    metros_por_pixel = 156_543.03392 * math.cos(math.radians(polygon.centroid.y)) / 2**zoom
    indices = range(1, total + 1)
    areas = ndimage.sum(edificios, etiquetas, indices) * metros_por_pixel**2
    centros = ndimage.center_of_mass(edificios, etiquetas, indices)
    recortes = ndimage.find_objects(etiquetas)

    candidatos = []
    for etiqueta, area, (fila, columna), recorte in zip(indices, areas, centros, recortes):
        if area < AREA_MINIMA_M2:
            continue
        lon, lat = pixel_a_lonlat(origen_x + columna + 0.5, origen_y + fila + 0.5, zoom)
        if not polygon.contains(Point(lon, lat)):
            continue
        fila0, columna0 = recorte[0].start, recorte[1].start
        huella = []
        for x, y in contorno_techo(etiquetas[recorte] == etiqueta):
            lon_v, lat_v = pixel_a_lonlat(origen_x + columna0 + x, origen_y + fila0 + y, zoom)
            huella.append([round(lat_v, 6), round(lon_v, 6)])
        candidatos.append((lat, lon, area, huella))
    # De norte a sur y de oeste a este, para una numeracion facil de seguir en el mapa.
    candidatos.sort(key=lambda c: (-round(c[0], 4), c[1]))

    return [
        {
            "id": i,
            "nombre": f"Inmueble {i}",
            "lat": round(lat, 6),
            "lon": round(lon, 6),
            "tipo": "Edificio" if area >= AREA_EDIFICIO_M2 else "Casa",
            "pisos": None,
            "apartamentos": None,
            "area_m2": round(area),
            "huella": huella,
        }
        for i, (lat, lon, area, huella) in enumerate(candidatos, start=1)
    ]


inject_theme()

st.markdown(
    """
    <div class="uh-header">
        <div class="uh-kicker">Detecci&oacute;n de edificios con IA &middot; Hugging Face</div>
        <div class="uh-title">Extractor UH</div>
        <p class="uh-subtitle">Dibuje un pol&iacute;gono sobre el mapa y obtenga las unidades
        habitacionales identificadas dentro del &aacute;rea.</p>
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
if "analizado" not in st.session_state:
    st.session_state.analizado = False
if "error_analisis" not in st.session_state:
    st.session_state.error_analisis = None

with st.sidebar:
    st.markdown("### Configuración del mapa")
    center_lat = st.number_input("Latitud centro", value=7.1200, format="%.6f")
    center_lon = st.number_input("Longitud centro", value=-66.1800, format="%.6f")
    zoom = st.slider("Zoom inicial", min_value=2, max_value=20, value=6)
    map_height = st.slider("Altura del mapa (px)", min_value=500, max_value=1200, value=800, step=50)

with st.container(border=True):
    st.markdown('<div class="uh-section-title">1. Dibuje el área de interés</div>', unsafe_allow_html=True)
    st.caption(
        "Acérquese a la zona y use la herramienta de polígono (pentágono, a la izquierda del mapa) para "
        f"trazar un área cerrada de hasta {formato_es(AREA_MAXIMA_KM2, 1)} km². La IA analizará la imagen satelital "
        "de esa área. Puede alternar entre mapa oscuro y satelital en el control de capas."
    )

    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=True, tiles=None)
    # Mapa oscuro: tiles de OpenStreetMap invertidos via CSS (LEAFLET_DARK_CSS),
    # ya que los tiles oscuros de CARTO ahora requieren API key.
    folium.TileLayer("OpenStreetMap", name="Oscuro", className="uh-dark-tiles").add_to(folium_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri",
        name="Satelital",
        show=False,
    ).add_to(folium_map)
    folium_map.get_root().header.add_child(folium.Element(LEAFLET_DARK_CSS))
    Draw(
        export=False,
        draw_options={
            "polygon": {
                "shapeOptions": {
                    "color": COLOR_AREA_DIBUJADA,
                    "weight": 2,
                    "dashArray": "6 6",
                    "fillOpacity": 0.04,
                }
            },
            "rectangle": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "polyline": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(folium_map)
    folium.LayerControl(position="topright").add_to(folium_map)

    # returned_objects se limita a los dibujos: por defecto st_folium tambien
    # devuelve center/zoom/bounds, que cambian con cada pan o zoom y disparan un
    # rerun completo que remonta el componente (efecto de "se agranda y se achica").
    # Los techos van en feature_group_to_add: st_folium actualiza esa capa sin
    # volver a montar el mapa, asi el poligono dibujado y la vista se conservan.
    techos = capa_techos(st.session_state.units, st.session_state.detection_run_id) if st.session_state.units else None
    map_state = st_folium(
        folium_map,
        height=map_height,
        use_container_width=True,
        returned_objects=["all_drawings", "last_active_drawing"],
        feature_group_to_add=techos,
        key="main_map",
    )

    if st.session_state.units:
        st.markdown(
            "<div class='uh-legend'>"
            "<span class='uh-legend-item'><i class='uh-legend-casa'></i>Casa</span>"
            "<span class='uh-legend-item'><i class='uh-legend-edificio'></i>Edificio</span>"
            "<span class='uh-legend-item'><i class='uh-legend-area'></i>Área analizada</span>"
            "<span class='uh-legend-hint'>Pase el mouse sobre un techo para ver de qué inmueble se trata.</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    drawing = None
    if map_state:
        drawing = map_state.get("last_active_drawing")
        if not drawing and map_state.get("all_drawings"):
            drawing = map_state["all_drawings"][-1]

    if drawing and drawing.get("geometry", {}).get("type") == "Polygon":
        nuevas_coords = drawing["geometry"]["coordinates"][0]
        if nuevas_coords != st.session_state.polygon_coords:
            st.session_state.polygon_coords = nuevas_coords
            st.session_state.error_analisis = None

    col_button, col_status = st.columns([1.2, 2.8], vertical_alignment="center")
    with col_button:
        run_detection = st.button("Obtener unidades habitacionales", type="primary")
    with col_status:
        polygon_coords = st.session_state.polygon_coords
        # El anillo GeoJSON repite el primer vertice al final para cerrarlo.
        n_vertices = len(polygon_coords) - 1 if polygon_coords else 0

        if run_detection and polygon_coords:
            try:
                with st.spinner("Analizando la imagen satelital con IA..."):
                    st.session_state.units = detect_units(polygon_coords)
                st.session_state.detection_run_id += 1
                st.session_state.analizado = True
                st.session_state.error_analisis = None
            except AnalisisError as error:
                st.session_state.error_analisis = str(error)
            # El mapa ya se dibujo en esta ejecucion: se repite para mostrar los techos.
            st.rerun()

        if run_detection and not polygon_coords:
            status("warning", "Primero dibuje un polígono cerrado sobre el mapa.")
        elif st.session_state.error_analisis:
            status("warning", st.session_state.error_analisis)
        else:
            if st.session_state.analizado and not st.session_state.units:
                status("warning", "No se encontraron unidades habitacionales en el área.")
            elif st.session_state.units:
                status(
                    "success",
                    f"Polígono de {n_vertices} vértices · {len(st.session_state.units)} unidades detectadas.",
                )
            elif polygon_coords:
                status("success", f"Polígono capturado con {n_vertices} vértices. Listo para analizar.")
            else:
                status("info", "Dibuje un polígono cerrado sobre el mapa para habilitar el análisis.")

with st.expander("2. Unidades habitacionales detectadas", expanded=True):
    if not st.session_state.units:
        if st.session_state.analizado:
            st.caption("El último análisis no encontró unidades habitacionales.")
        else:
            st.caption("Aún no se ha ejecutado el análisis.")
    else:
        st.caption(
            "El tipo se estima según el tamaño del techo detectado. Revise cada fila y complete "
            "pisos y apartamentos en los edificios."
        )
        run_id = st.session_state.detection_run_id
        column_widths = [0.5, 2, 2.3, 1.3, 1, 1]

        header_cols = st.columns(column_widths)
        for col, label in zip(
            header_cols,
            ["#", "Nombre del inmueble", "Coordenadas (lat, lon)", "Tipo", "Pisos", "Apartamentos"],
        ):
            col.markdown(f"<div class='uh-grid-header'>{label}</div>", unsafe_allow_html=True)

        tipo_options = ["Casa", "Edificio"]

        for unit in st.session_state.units:
            row_cols = st.columns(column_widths, vertical_alignment="center")
            key_prefix = f"{run_id}_{unit['id']}"

            row_cols[0].markdown(f"<div class='uh-grid-cell'>{unit['id']}</div>", unsafe_allow_html=True)

            unit["nombre"] = row_cols[1].text_input(
                "Nombre", value=unit["nombre"], key=f"nombre_{key_prefix}", label_visibility="collapsed"
            )

            row_cols[2].markdown(
                f"<div class='uh-grid-cell'>{unit['lat']:.6f}, {unit['lon']:.6f}</div>", unsafe_allow_html=True
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
                row_cols[4].markdown("<div class='uh-grid-cell uh-grid-muted'>&mdash;</div>", unsafe_allow_html=True)
                row_cols[5].markdown("<div class='uh-grid-cell uh-grid-muted'>&mdash;</div>", unsafe_allow_html=True)
