# AGENTS.md · Extractor UH

> Este fichero lo leen los agentes de programación (Copilot, Cursor, Codex, Gemini CLI y
> otros) antes de tocar el código. Para Claude Code, crear un `CLAUDE.md` con una sola
> línea: `Sigue las instrucciones de AGENTS.md`.

## Qué es este proyecto

App que permite a los equipos de planificación dibujar una zona sobre un mapa y obtener,
con ayuda de IA, la lista de casas y edificios de viviendas que hay dentro, para estimar
cuántas viviendas tiene la zona.

## Cómo está hecho

- Python 3.12.
- Interfaz: Streamlit (ruta A), desplegada en https://extractoruh.streamlit.app/.
- Mapa: `folium` + `streamlit-folium` (`st_folium`), con el plugin `Draw` para el polígono.
  Geometría con `shapely`.
- IA: modelo de Hugging Face `geobase/building-footprint-segmentation` (constante `MODELO`
  de `app.py`), U-Net que segmenta techos de edificios. **Solo existe en ONNX** (para
  GeoAI.js), así que `transformers.pipeline` no puede cargarlo: se descarga con
  `huggingface_hub` y se ejecuta con `onnxruntime` en `detect_units()`, que devuelve una
  lista de diccionarios con `id`, `nombre`, `lat`, `lon`, `tipo`, `pisos`, `apartamentos` y
  `area_m2`.
- Imágenes: tiles de Esri World Imagery con zoom 18 (`ZOOM_ANALISIS`). Los parámetros del
  análisis (umbral, áreas mínima y máxima, área para proponer "Edificio") son constantes al
  principio de `app.py`.
- Archivos: `app.py` (toda la app), `requirements.txt` (dependencias) y `README.md`. Además,
  la documentación del reto: `especificacion.md`, `AGENTS.md` y `hito.txt`.
- La especificación completa está en [`especificacion.md`](especificacion.md).

## Reglas

- Hoy la app no usa claves. Si se añade alguna, va SIEMPRE en secretos (`st.secrets`).
  Nunca en el código, en el README ni en el chat.
- Cada descarga de imágenes y cada ejecución del modelo van dentro de `try/except`. Los
  fallos se convierten en `AnalisisError` con un mensaje claro para el usuario.
- El modelo se carga con `@st.cache_resource` (una vez por servidor): el plan gratuito de
  Streamlit Cloud tiene poca memoria.
- Textos de la interfaz en español, sencillos, pensados para alguien sin conocimientos
  técnicos.
- Cambios pequeños: uno por petición. Al terminar, explica qué has cambiado y por qué.
- Si añades una librería, añádela también a `requirements.txt` y dilo.
- No uses datos personales reales. Para probar, inventa ejemplos.
- No añadas archivos nuevos al repositorio (por ejemplo `.streamlit/config.toml`) sin
  preguntar. El tema visual se define con CSS dentro de `app.py` (función `inject_theme`).

## Detalles técnicos que hay que respetar

- **Tema oscuro/neón:** los colores están como variables CSS en `inject_theme()` (`--bg`,
  `--cyan`, `--magenta`…). Se fuerzan todos los colores para que la app se vea igual con
  el tema claro u oscuro del navegador. Reutiliza esas variables y no metas colores sueltos.
- **Streamlit 1.66 usa react-aria, no BaseWeb:** los selectores CSS válidos son
  `stNumberInputContainer`, `stTextInputRootElement`, `stSelectbox div[role="group"]`,
  `stSlider div[role="group"]` y `[role="listbox"]`. Los selectores `data-baseweb` ya no
  funcionan.
- **Mapa:** pasa el `folium.Map` directamente a `st_folium`. Si lo envuelves en
  `folium.Figure`, el iframe queda recortado a 150 px. Mantén
  `returned_objects=["all_drawings", "last_active_drawing"]`: si observa también
  center/zoom/bounds, cada movimiento del mapa recarga la página y el mapa parpadea.
- **Tiles:** la capa "Oscuro" es OpenStreetMap con un filtro CSS (los tiles oscuros de
  CARTO piden API key). La capa "Satelital" es Esri World Imagery. Mantén las atribuciones.
- **Techos en el mapa:** cada unidad guarda su contorno en `huella` (`contorno_techo()`).
  Se dibujan con `capa_techos()` y se pasan a `st_folium(feature_group_to_add=...)`, que
  actualiza la capa sin volver a montar el mapa: así se conservan el polígono dibujado y
  la vista. Como el mapa se dibuja antes que el botón, tras analizar se hace `st.rerun()`
  y el mensaje de error se guarda en `st.session_state.error_analisis`. El color del techo
  y el tooltip leen los valores actuales de la grilla desde `st.session_state`.
- **Grilla:** cada fila es un `st.columns()`. Las claves de los widgets llevan
  `detection_run_id` para que un análisis nuevo no herede valores del anterior.

## Cómo comprobar que funciona

- `streamlit run app.py` en local, o haz push (tras mi revisión) y revisa la app en
  Streamlit Cloud.
- Para cambios visuales, comprueba el resultado en un navegador real (capturas en modo
  claro y oscuro). Que no haya errores no basta.
- Casos de prueba (entrada → lo que esperamos):
  1. Polígono sobre una manzana residencial + botón → grilla con inmuebles numerados desde 1,
     todos dentro del polígono, y mensaje con el total detectado.
  2. Cambiar el tipo de una fila a "Edificio" → aparecen Pisos y Apartamentos solo en esa
     fila y su techo pasa de cian a magenta en el mapa, sin perder el polígono dibujado.
  3. Pulsar el botón sin polígono → "Primero dibuje un polígono cerrado sobre el mapa." y no
     se llama a la IA.
  4. Polígono sobre un parque o el agua → "No se encontraron unidades habitacionales en el
     área".
  5. Polígono de más de 0,5 km² → mensaje que pide acercarse y dibujar un área más pequeña,
     sin descargar imágenes.
  6. Sin conexión para descargar la imagen → mensaje claro para revisar la conexión, sin
     romper la app.
- Referencia: un polígono de ≈ 0,15 km² sobre Chacao (Caracas, 10.4965, -66.8533) da unas
  86 unidades en unos 5 segundos.

## Qué no hacer

- No borres ni reescribas el README sin preguntar.
- No cambies el modelo ni la arquitectura sin preguntar.
- No hagas push ni despliegues sin que yo lo haya revisado.
