# Extractor UH

Aplicación web construida con [Streamlit](https://streamlit.io/) que permite
dibujar un polígono cerrado sobre un mapa interactivo e identificar, con un
modelo de IA de Hugging Face, las unidades habitacionales (casas y edificios)
que hay dentro del área seleccionada.

## Funcionalidad

1. **Mapa interactivo** ([folium](https://python-visualization.github.io/folium/) + Leaflet),
   con dos capas seleccionables desde el control de capas:
   - **Oscuro** (por defecto): tiles de OpenStreetMap con un filtro oscuro.
   - **Satelital**: imágenes de Esri World Imagery, útiles para reconocer inmuebles.
2. **Dibujo de polígono**: con la herramienta de polígono del mapa
   ([`folium.plugins.Draw`](https://python-visualization.github.io/folium/latest/user_guide/plugins/draw.html))
   el usuario traza con el mouse un área cerrada. Se puede editar o borrar.
3. **Botón "Obtener unidades habitacionales"**: el modelo de IA analiza la
   imagen satelital del área delimitada (hasta 0,5 km²) y detecta los techos de
   los edificios. Un indicador de estado junto al botón informa los vértices
   capturados y las unidades detectadas. También avisa si no se dibujó el
   polígono, si la zona es demasiado grande, si no se encontraron viviendas o
   si falló la descarga de imágenes.
4. **Grilla de resultados** (sección colapsable) con:
   - Número de fila (ascendente).
   - Nombre del inmueble (editable, por defecto "Inmueble 1", "Inmueble 2", ...).
   - Coordenadas geográficas del inmueble.
   - Lista desplegable **Tipo**: `Casa` o `Edificio`, propuesto según el tamaño
     del techo (300 m² o más → `Edificio`) y editable.
   - Si el tipo es `Edificio`, se habilitan los campos **Pisos** y **Apartamentos**.

La barra lateral permite ajustar el centro del mapa (Venezuela por defecto), el
zoom inicial y la altura del mapa (800 px por defecto).

## La pieza de IA

- **Modelo:** [`geobase/building-footprint-segmentation`](https://huggingface.co/geobase/building-footprint-segmentation),
  una red U-Net que segmenta techos de edificios en imágenes satelitales.
- **Cómo se ejecuta:** el modelo solo se publica en formato ONNX para la
  librería JavaScript [GeoAI.js](https://github.com/decision-labs/geoai.js), por
  lo que `transformers.pipeline` no puede cargarlo. La app lo descarga con
  `huggingface_hub` (versión cuantizada, 15 MB, una sola vez) y lo ejecuta con
  `onnxruntime`, replicando el procesamiento de GeoAI.js:
  1. Descarga las imágenes de Esri World Imagery del área con zoom 18 (≈ 0,6 m por píxel).
  2. El modelo estima, píxel a píxel, la probabilidad de que sea techo.
  3. Los píxeles con probabilidad ≥ 0,3 dentro del polígono se agrupan en
     techos; se descartan los de menos de 25 m².
  4. Cada techo se convierte en una fila de la grilla con su coordenada central.
- **Qué pasa si falla:** si no se pueden descargar las imágenes o el modelo
  falla, la app muestra un mensaje claro y conserva el polígono.
- **Limitaciones:** detecta bien los techos claros, pero se le escapan techos
  oscuros o de teja. Puede dividir un edificio en varios trozos y no distingue
  viviendas de comercios. El resultado es una estimación para revisar en la
  grilla.
- **Referencia:** un polígono de ≈ 0,15 km² en Chacao (Caracas) devuelve unas
  86 unidades en unos 5 segundos.

## Diseño visual

Tema **oscuro/neón** definido a partir de las recomendaciones de la skill
`ui-ux-pro-max` (estilos *Dark Mode OLED* + acentos *Cyberpunk UI*):

| Token | Valor | Uso |
|-------|-------|-----|
| Fondo | `#0A0E27` | Fondo de la página |
| Superficie | `#11163A` / `#171E4A` | Paneles, inputs |
| Cian | `#00E5FF` | Acento principal, foco, encabezados de la grilla |
| Magenta | `#FF2BD6` | Acento secundario (degradado del botón y del título) |
| Texto | `#E6EAFF` / `#9AA3C7` | Texto principal / secundario |

- Tipografía: **Fira Sans** (interfaz) y **Fira Code** (datos, coordenadas, títulos de sección).
- Brillo neón moderado (sin animaciones *glitch* ni *scanlines*, para preservar
  legibilidad); se respeta `prefers-reduced-motion`.
- Todos los colores se fuerzan vía CSS, por lo que la app se ve igual con el
  tema claro u oscuro de Streamlit/navegador.

## Instalación y ejecución local

```bash
git clone https://github.com/gatojohn/extractor_uh.git
cd extractor_uh
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación quedará disponible en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Entrar a [share.streamlit.io](https://share.streamlit.io/) con la cuenta de GitHub.
2. Crear una nueva app, seleccionar este repositorio y la rama `master`, e
   indicar `app.py` como archivo principal.
3. Streamlit instala automáticamente las dependencias de `requirements.txt`.

Demo: https://extractoruh.streamlit.app/

## Estructura del repositorio

```
.
├── app.py             # Aplicación Streamlit (lógica + UI + detección con IA)
├── requirements.txt   # Dependencias de Python
├── README.md          # Este documento
├── especificacion.md  # Especificación del producto (reto)
├── AGENTS.md          # Instrucciones para agentes de programación
└── hito.txt           # Ficha del hito de la semana 2
```

## Roadmap (próximas versiones)

- Dibujar sobre el mapa los techos detectados para validar el resultado a simple vista.
- Clasificar casa / edificio y estimar pisos con un modelo, en lugar de por el
  tamaño del techo.
- Persistir los polígonos y resultados (base de datos / almacenamiento).
- Exportar la grilla de unidades habitacionales (CSV / GeoJSON).
- Soportar múltiples polígonos por sesión.

## Créditos de mapas

© colaboradores de [OpenStreetMap](https://www.openstreetmap.org/copyright) ·
Imágenes satelitales © Esri · Modelo de segmentación de
[Geobase / GeoAI](https://huggingface.co/geobase/building-footprint-segmentation).
