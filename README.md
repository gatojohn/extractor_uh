# Extractor UH

Aplicación web construida con [Streamlit](https://streamlit.io/) que permite
dibujar un polígono cerrado sobre un mapa interactivo y simular la
identificación, mediante un modelo de IA, de las unidades habitacionales
(casas y edificios) existentes dentro del área seleccionada.

Esta es la **versión 0** del producto: el análisis de IA está **simulado**
(genera inmuebles de ejemplo distribuidos dentro del polígono) a fin de
validar el flujo de usuario antes de integrar un modelo real.

## Funcionalidad

1. **Mapa interactivo** ([folium](https://python-visualization.github.io/folium/) + Leaflet),
   con dos capas seleccionables desde el control de capas:
   - **Oscuro** (por defecto): tiles de OpenStreetMap con un filtro oscuro.
   - **Satelital**: imágenes de Esri World Imagery, útiles para reconocer inmuebles.
2. **Dibujo de polígono**: con la herramienta de polígono del mapa
   ([`folium.plugins.Draw`](https://python-visualization.github.io/folium/latest/user_guide/plugins/draw.html))
   el usuario traza con el mouse un área cerrada. Se puede editar o borrar.
3. **Botón "Obtener unidades habitacionales"**: simula que un modelo de IA
   analiza el área delimitada e identifica los inmuebles presentes. Un
   indicador de estado junto al botón informa vértices capturados y unidades
   detectadas.
4. **Grilla de resultados** (sección colapsable) con:
   - Número de fila (ascendente).
   - Nombre del inmueble (editable, por defecto "Inmueble 1", "Inmueble 2", ...).
   - Coordenadas geográficas del inmueble.
   - Lista desplegable **Tipo**: `Casa` o `Edificio`.
   - Si el tipo es `Edificio`, se habilitan los campos **Pisos** y **Apartamentos**.

La barra lateral permite ajustar el centro del mapa, el zoom inicial y la
altura del mapa (800 px por defecto).

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
├── app.py             # Aplicación Streamlit (lógica + UI)
├── requirements.txt   # Dependencias de Python
└── README.md          # Este documento
```

## Roadmap (próximas versiones)

- Reemplazar `simulate_ai_detection()` por la inferencia de un modelo real de
  detección de inmuebles sobre imágenes satelitales/aéreas.
- Persistir los polígonos y resultados (base de datos / almacenamiento).
- Exportar la grilla de unidades habitacionales (CSV / GeoJSON).
- Soportar múltiples polígonos por sesión.

## Créditos de mapas

© colaboradores de [OpenStreetMap](https://www.openstreetmap.org/copyright) ·
Imágenes satelitales © Esri.
