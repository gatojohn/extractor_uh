# Especificación · Extractor UH

> Una buena especificación es la que otro asistente de IA podría construir sin haceros
> preguntas. De ella sale la ficha del **hito de la semana 2**.

## 1. Problema y usuario

- **Quién tiene el problema:** equipos de planificación (de una empresa de servicios como
  telecomunicaciones o energía, o de un ayuntamiento) que necesitan saber cuántas viviendas
  hay en una zona antes de decidir una inversión o un despliegue.
- **Cómo lo resuelve hoy:** recorre la zona a pie, o cuenta casas y edificios uno a uno
  mirando Google Maps o Street View, y apunta los datos en un Excel. A veces consulta el
  catastro, que no siempre está actualizado ni es fácil de usar.
- **Qué le molesta de cómo lo hace hoy:** tarda horas por cada zona, se equivoca al contar
  (repite o se salta inmuebles) y no tiene la ubicación exacta de cada uno.
- **Qué parte resuelve la IA:** mira la imagen satelital del área elegida y localiza el techo
  de cada edificio. A partir del tamaño del techo, la app propone si es una casa o un
  edificio. **Queda fuera:** saber cuántos pisos y apartamentos tiene cada edificio (lo
  completa la persona), distinguir viviendas de comercios o naves, y cualquier dato de quién
  vive en cada vivienda.

## 2. Qué hace el producto

1. Muestra un mapa (oscuro o satelital) donde el usuario dibuja con el mouse un polígono
   cerrado sobre la zona que le interesa.
2. Al pulsar **"Obtener unidades habitacionales"**, un modelo de IA analiza la imagen
   satelital del polígono y devuelve la lista de edificios que encuentra dentro, con sus
   coordenadas.
3. Presenta los resultados en una grilla editable: número de fila, nombre del inmueble
   ("Inmueble 1", "Inmueble 2"… si no tiene nombre), coordenadas, tipo (Casa / Edificio) y,
   si es un edificio, número de pisos y de apartamentos.

## 3. Entradas y salidas

| El usuario da… | La app devuelve… | Ejemplo |
|---|---|---|
| Un polígono dibujado sobre el mapa | Una lista de inmuebles dentro del polígono, cada uno con nombre, coordenadas y tipo estimado | Polígono de 4 vértices (≈ 0,15 km²) sobre Chacao, Caracas → 86 inmuebles en unos 5 segundos |
| Un cambio en la columna **Tipo** de una fila | Si elige *Edificio*, aparecen los campos **Pisos** y **Apartamentos**; si elige *Casa*, desaparecen | "Inmueble 4": Casa → Edificio, 6 pisos, 24 apartamentos |
| Un cambio en el nombre de un inmueble | La grilla guarda el nombre nuevo | "Inmueble 2" → "Edificio Los Álamos" |

## 4. La pieza de IA

- **Opción elegida:** modelo de Hugging Face
  [`geobase/building-footprint-segmentation`](https://huggingface.co/geobase/building-footprint-segmentation),
  una red U-Net que segmenta huellas (techos) de edificios en imágenes satelitales.
- **Por qué:** está entrenado justo para esta tarea, corre gratis en el propio servidor de
  la app (sin clave de API ni cuota) y analiza una manzana en pocos segundos.
- **Cómo se usa:** el modelo solo se publica en formato ONNX para la librería JavaScript
  GeoAI.js, así que la librería `transformers` de Python (`pipeline()`) no puede cargarlo.
  La app lo descarga con `huggingface_hub` y lo ejecuta con `onnxruntime`, replicando el
  procesamiento de GeoAI.js:
  1. Descarga las imágenes de Esri World Imagery que cubren el polígono, con zoom 18
     (unos 0,6 m por píxel; en pruebas da mejores resultados que 17 o 19).
  2. Pasa al modelo parches RGB de 256 × 256 píxeles con valores entre 0 y 1. El modelo
     devuelve, para cada píxel, la probabilidad de que sea techo de edificio.
  3. Marca como edificio los píxeles con probabilidad de 0,3 o más que caen dentro del
     polígono, y cuenta cada techo separado como un inmueble. Descarta los techos de menos
     de 25 m².
  4. Calcula el centro de cada techo en latitud y longitud. Propone "Edificio" si el techo
     mide 300 m² o más, y "Casa" si es más pequeño.
- **Limitaciones conocidas:** detecta bien los techos claros, pero se le escapan techos
  oscuros o de teja. A veces divide un edificio en varios trozos y no distingue viviendas
  de comercios o naves. Por eso la grilla es editable y el resultado es una estimación que
  la persona revisa.

## 5. Flujo del usuario

1. El usuario entra y ve un mapa grande centrado en Venezuela, con un mensaje que le pide
   dibujar un polígono. Se acerca con el zoom hasta la zona que le interesa. En la barra
   lateral puede cambiar el centro, el zoom inicial y la altura del mapa.
2. Elige la herramienta de polígono (el pentágono, a la izquierda del mapa), marca los
   vértices de la zona y cierra la figura haciendo clic en el primer punto. El mensaje junto
   al botón confirma "Polígono capturado con N vértices".
3. Pulsa **"Obtener unidades habitacionales"**. Recibe la grilla con los inmuebles
   encontrados y un mensaje con cuántos se detectaron. Revisa cada fila, corrige el tipo y
   completa pisos y apartamentos en los edificios. Puede contraer la grilla para volver a
   ver el mapa.

## 6. Restricciones

- Coste cero: modelo abierto de Hugging Face ejecutado en el servidor e imágenes públicas
  de Esri, con despliegue gratuito en Streamlit Community Cloud (ruta A). No hace falta
  ninguna clave de API. Si en el futuro se añade una, irá en secretos (`st.secrets`) y
  nunca en el código.
- El servidor gratuito tiene poca memoria: el modelo se carga una sola vez (versión
  cuantizada, 15 MB) y se reutiliza entre análisis.
- Sin datos personales reales: la app solo maneja coordenadas y características de
  inmuebles, nunca nombres de propietarios ni de residentes.
- Lenguaje sencillo, para alguien sin conocimientos técnicos.
- Área máxima por análisis: 0,5 km². Así se limita el número de imágenes que hay que
  descargar y el tiempo de espera.
- Los mapas deben mostrar la atribución de OpenStreetMap y de Esri.
- El repositorio contiene solo los archivos de la app (`app.py`, `requirements.txt`,
  `README.md`) y los de documentación del reto (`especificacion.md`, `AGENTS.md`,
  `hito.txt`).

## 7. Criterios de aceptación

La app está bien si:

| Entrada | Resultado esperado |
|---|---|
| Polígono sobre una manzana residencial y pulsar el botón | La grilla muestra los inmuebles numerados desde 1, todos con coordenadas dentro del polígono, y el mensaje indica cuántos se detectaron |
| Cambiar el tipo de una fila de *Casa* a *Edificio* | Aparecen los campos Pisos y Apartamentos solo en esa fila |
| Pulsar el botón sin haber dibujado un polígono | Mensaje: "Primero dibuje un polígono cerrado sobre el mapa." No se llama a la IA |
| Polígono sobre un parque o el agua, sin viviendas | Mensaje "No se encontraron unidades habitacionales en el área" y grilla vacía |
| Polígono de más de 0,5 km² | Mensaje que pide acercarse y dibujar una zona más pequeña. No se descargan imágenes ni se llama a la IA |
| No se puede descargar la imagen satelital (sin conexión) | Mensaje claro: "No se pudo descargar la imagen satelital de la zona. Revise la conexión y pruebe de nuevo." La app no se rompe y conserva el polígono |
| El modelo falla al analizar | Mensaje: "El modelo de IA no pudo analizar la imagen en este momento. Pruebe de nuevo en un minuto." |
| Contraer y volver a expandir la grilla | El título sigue visible y los datos editados se mantienen |

## 8. Fuera de alcance (por ahora)

- Dibujar sobre el mapa los techos detectados, para comprobar el resultado de un vistazo.
- Clasificar casa / edificio y estimar pisos con un modelo, en lugar de por el tamaño del
  techo.
- Exportar la grilla a CSV o GeoJSON.
- Guardar polígonos y resultados entre sesiones.
- Analizar varios polígonos a la vez.
- Calcular el total de viviendas de la zona (casas + apartamentos) y mostrarlo en un resumen.
- Cruzar los datos con el catastro o con direcciones postales.

## 9. Despliegue

- **Ruta:** A (Streamlit Community Cloud) → https://extractoruh.streamlit.app/
- **Repositorio:** https://github.com/gatojohn/extractor_uh
