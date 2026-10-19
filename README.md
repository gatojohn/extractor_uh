# Extractor UH

Aplicación web construida con [Streamlit](https://streamlit.io/) que permite
dibujar un polígono cerrado sobre un mapa interactivo y simular la
identificación, mediante un modelo de IA, de las unidades habitacionales
(casas y edificios) existentes dentro del área seleccionada.

Esta es la **versión 0** del producto: el análisis de IA está **simulado**
(genera inmuebles de ejemplo distribuidos dentro del polígono) a fin de
validar el flujo de usuario antes de integrar un modelo real.

## Funcionalidad

1. **Mapa interactivo**: se carga un mapa (Leaflet, vía [folium](https://python-visualization.github.io/folium/))
   centrado en las coordenadas configuradas en la barra lateral.
2. **Dibujo de polígono**: usando la herramienta de dibujo del mapa
   ([`folium.plugins.Draw`](https://python-visualization.github.io/folium/latest/user_guide/plugins/draw.html)),
   el usuario traza con el mouse un polígono cerrado sobre la zona de interés.
3. **Botón "Obtener unidades habitacionales"**: al presionarlo, se simula que
   un modelo de IA analiza el área delimitada por el polígono e identifica
   los inmuebles presentes.
4. **Grilla de resultados**: debajo del mapa se despliega una tabla con:
   - Número de fila (ascendente).
   - Nombre del inmueble (editable, por defecto "Inmueble 1", "Inmueble 2", ...).
   - Coordenadas geográficas del inmueble.
   - Lista desplegable **Tipo**: `Casa` o `Edificio`.
   - Si el tipo es `Edificio`, se habilitan los campos **Pisos** y
     **Apartamentos** para registrar esa información.

## Diseño visual

La interfaz utiliza el CSS de [PatternFly](https://www.patternfly.org/) (cargado
desde CDN) junto con una hoja de estilos propia que reutiliza los tokens de
color y tipografía de PatternFly (azul primario `#0066cc`, fuente *Red Hat
Text*, bordes y espaciados) para dar a los widgets nativos de Streamlit
(botones, selects, inputs) una apariencia consistente con ese sistema de
diseño.

> **Nota técnica**: PatternFly es una librería de componentes React. Streamlit
> renderiza sus propios widgets (no React de PatternFly), por lo que la
> integración actual consiste en aplicar el CSS/tokens visuales de PatternFly
> sobre los componentes de Streamlit, en lugar de usar los componentes React
> originales. Una integración 1:1 con componentes PatternFly requeriría
> desarrollar un *custom component* de Streamlit en JavaScript/React.

## Instalación y ejecución local

```bash
git clone https://github.com/<tu-usuario>/extractor_uh.git
cd extractor_uh
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación quedará disponible en `http://localhost:8501`.

## Despliegue en Streamlit Community Cloud

1. Subí este repositorio a GitHub (público).
2. Entrá a [share.streamlit.io](https://share.streamlit.io/) e iniciá sesión
   con tu cuenta de GitHub.
3. Creá una nueva app, seleccioná este repositorio y rama, e indicá
   `app.py` como archivo principal.
4. Deploy. Streamlit instalará automáticamente las dependencias listadas en
   `requirements.txt`.

## Estructura del repositorio

```
.
├── app.py             # Aplicación Streamlit (lógica + UI)
├── requirements.txt   # Dependencias de Python
└── README.md          # Este documento
```

## Roadmap (próximas versiones)

- Reemplazar `simulate_ai_detection()` por la inferencia de un modelo real de
  detección de inmuebles (por ejemplo, sobre imágenes satelitales/aéreas).
- Persistir los polígonos y resultados (base de datos / almacenamiento).
- Exportar la grilla de unidades habitacionales (CSV / GeoJSON).
- Soportar múltiples polígonos por sesión.
- Autenticación y manejo de usuarios/proyectos.
