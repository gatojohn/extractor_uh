# Extractor UH

Aplicación web construida con [Streamlit](https://streamlit.io/) que permite
dibujar un polígono cerrado sobre un mapa interactivo e identificar, con un
modelo de IA de Hugging Face, las unidades habitacionales (casas y edificios)
que hay dentro del área seleccionada. Los nombres de los edificios salen de
OpenStreetMap.

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
   los edificios. Los techos se dibujan sobre el mapa (cian = casa, magenta =
   edificio) y, al pasar el mouse, muestran número, nombre, tipo y superficie;
   el color se actualiza si se cambia el tipo en la grilla. Un indicador de
   estado junto al botón informa los vértices
   capturados y las unidades detectadas. También avisa si no se dibujó el
   polígono, si la zona es demasiado grande, si no se encontraron viviendas o
   si falló la descarga de imágenes.
4. **Grilla de resultados** (sección colapsable) con:
   - Número de fila (ascendente).
   - Nombre del inmueble (editable): el de OpenStreetMap si coincide con el techo;
     si no, "Inmueble 1", "Inmueble 2", ...
   - Coordenadas geográficas del inmueble.
   - Lista desplegable **Tipo** (editable). Con nombre de OpenStreetMap, es
     `Edificio` si empieza por Torre, Residencias o Edificio, y `Casa` si no. Sin
     nombre, se estima por el tamaño del techo (300 m² o más → `Edificio`).
   - Si el tipo es `Edificio`, se habilitan los campos **Pisos** y **Apartamentos**.
   - Botón **Eliminar**: quita la fila de la grilla y su techo del mapa. Los
     números de las demás filas no cambian y el indicador de estado muestra
     cuántas quedan. Un análisis nuevo vuelve a traer todas las unidades.

La barra lateral permite ajustar el centro del mapa (por defecto, la avenida
principal de La Urbina, en Caracas), el zoom inicial (17) y la altura del mapa
(800 px por defecto).

## La pieza de IA

La app combina dos modelos de Hugging Face:

- **Modelo principal:** [`geobase/building-footprint-segmentation`](https://huggingface.co/geobase/building-footprint-segmentation),
  una red U-Net que segmenta techos de edificios en imágenes satelitales. Decide
  qué inmuebles hay en la zona. Solo se publica en formato ONNX para la librería
  JavaScript [GeoAI.js](https://github.com/decision-labs/geoai.js), por lo que
  `transformers.pipeline` no puede cargarlo. La app lo descarga con
  `huggingface_hub` (versión cuantizada, 15 MB, una sola vez) y lo ejecuta con
  `onnxruntime`, replicando el procesamiento de GeoAI.js.
- **Segunda opinión:** [`sawthiha/segformer-b0-finetuned-deprem-satellite`](https://huggingface.co/sawthiha/segformer-b0-finetuned-deprem-satellite),
  un SegFormer de 15 MB que se ejecuta con
  `pipeline("image-segmentation", model=...)` de `transformers`. Apenas ve casas,
  pero completa algunos techos grandes que el modelo principal deja a medias.

Cómo se combinan:

1. La app descarga las imágenes de Esri World Imagery del área con zoom 18
   (≈ 0,6 m por píxel) para el modelo principal y con zoom 19 (≈ 0,3 m por píxel)
   para SegFormer.
2. El modelo principal estima, píxel a píxel, la probabilidad de que sea techo
   (se toma ≥ 0,3).
3. Una región de SegFormer solo se añade si mide 300 m² o más, es muy compacta
   y toca un techo ya detectado. Sin este filtro, junta casas vecinas con sus
   patios y marca como techo la imagen de relleno que Esri sirve sobre el agua.
4. Los píxeles dentro del polígono se agrupan en techos, se descartan los de
   menos de 25 m² y cada techo se convierte en una fila de la grilla.

## Nombres de OpenStreetMap

Tras detectar los techos, la app pide a [Overpass](https://overpass-api.de/) los
edificios con nombre de la zona y los cruza con cada techo:

- Hay coincidencia si se solapa al menos el 30 % del techo o del edificio de
  OpenStreetMap. Si varios coinciden, gana el de mayor solape.
- La fila toma ese nombre. Es `Edificio` si empieza por Torre, Residencias o
  Edificio (singular o plural, sin distinguir mayúsculas), y `Casa` si empieza de
  otra forma.
- Un techo sin coincidencia no tiene nombre: es `Edificio` si mide 300 m² o más,
  y `Casa` si es más pequeño.

## Qué pasa si falla algo

- Si no se pueden descargar las imágenes o el modelo principal falla, la app
  muestra un mensaje claro y conserva el polígono.
- Si falla SegFormer, la app sigue con el modelo principal y lo indica junto al
  botón.
- Si Overpass no responde (sus servidores públicos se saturan a menudo), la app
  prueba un segundo servidor, espera como mucho 10 segundos por cada uno y, si
  ambos fallan, sigue sin nombres, aplica a todas las filas la regla del tamaño
  del techo y lo indica junto al botón.

## Limitaciones

- Detecta bien los techos claros, pero se le escapan techos oscuros o de teja.
- Puede dividir un edificio en varios trozos: varias filas tendrán el mismo
  nombre de OpenStreetMap y conviene eliminar las que sobren.
- No distingue viviendas de comercios: una oficina llamada "Edificio X" cuenta
  como `Edificio`.
- La clasificación depende de lo completo que esté OpenStreetMap. En La Urbina
  casi todos los edificios tienen nombre. Donde no los hay, manda el tamaño del
  techo: un edificio alto de planta pequeña sale como `Casa`, y una nave grande
  como `Edificio`.
- Un edificio con otro nombre sale como `Casa` aunque sea grande: "Hotel X" o
  "Automercado Y".
- SegFormer cambia pocas unidades: en las pruebas no alteró zonas de casas y solo
  mejoró algunos techos grandes.

El resultado es una estimación para revisar en la grilla.

**Referencia:** un polígono de ≈ 0,15 km² en Chacao (Caracas) tarda unos
15 segundos. En La Urbina, uno similar devuelve unas 95 unidades: unas 55 con
nombre de OpenStreetMap, casi todas `Edificio`.

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

- Clasificar casa / edificio y estimar pisos con un modelo, para no depender de
  los nombres de OpenStreetMap.
- Persistir los polígonos y resultados (base de datos / almacenamiento).
- Exportar la grilla de unidades habitacionales (CSV / GeoJSON).
- Soportar múltiples polígonos por sesión.

## Créditos de mapas

Mapa y nombres de edificios © colaboradores de
[OpenStreetMap](https://www.openstreetmap.org/copyright) ·
Imágenes satelitales © Esri · Modelos de segmentación de
[Geobase / GeoAI](https://huggingface.co/geobase/building-footprint-segmentation) y
[sawthiha](https://huggingface.co/sawthiha/segformer-b0-finetuned-deprem-satellite).
