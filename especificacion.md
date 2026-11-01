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
  de cada edificio. Después, la app cruza esos techos con OpenStreetMap para obtener su
  nombre y proponer si es una casa o un edificio. **Queda fuera:** saber cuántos pisos y
  apartamentos tiene cada edificio (lo
  completa la persona), distinguir viviendas de comercios o naves, y cualquier dato de quién
  vive en cada vivienda.

## 2. Qué hace el producto

1. Muestra un mapa (oscuro o satelital) donde el usuario dibuja con el mouse un polígono
   cerrado sobre la zona que le interesa.
2. Al pulsar **"Obtener unidades habitacionales"**, un modelo de IA analiza la imagen
   satelital del polígono y devuelve la lista de edificios que encuentra dentro, con sus
   coordenadas. Los techos detectados se dibujan sobre el mapa (cian = casa, magenta =
   edificio), con su número y nombre al pasar el mouse.
3. Busca en OpenStreetMap los edificios con nombre de la zona y los cruza con los techos
   detectados. Si un techo coincide, toma ese nombre. Es **Edificio** si el nombre empieza
   por Torre, Residencias o Edificio, y **Casa** si empieza de otra forma. Si no hay
   coincidencia, propone el tipo por el tamaño del techo.
4. Presenta los resultados en una grilla editable: número, nombre del inmueble ("Inmueble 1",
   "Inmueble 2"… si no tiene nombre), coordenadas, tipo (Casa / Edificio), número de pisos y
   de apartamentos si es un edificio, y un botón **Eliminar**.
5. Al eliminar una fila, desaparecen también su techo del mapa y su unidad del total.

## 3. Entradas y salidas

| El usuario da… | La app devuelve… | Ejemplo |
|---|---|---|
| Un polígono dibujado sobre el mapa | Una lista de inmuebles dentro del polígono, cada uno con nombre, coordenadas y tipo estimado | Polígono de 4 vértices (≈ 0,15 km²) sobre La Urbina, Caracas → unos 95 inmuebles en unos 15 segundos, unos 55 con nombre de OpenStreetMap |
| Un techo que coincide con "Residencias Vaduz" en OpenStreetMap | La fila se llama así y su tipo es Edificio | Un techo que coincide con "Automercado Luz" → Casa |
| Un techo sin coincidencia en OpenStreetMap | La fila se llama "Inmueble N" y su tipo sale del tamaño del techo | Techo de 391 m² → Edificio; techo de 120 m² → Casa |
| Un cambio en la columna **Tipo** de una fila | Si elige *Edificio*, aparecen los campos **Pisos** y **Apartamentos**; si elige *Casa*, desaparecen | "Inmueble 4": Casa → Edificio, 6 pisos, 24 apartamentos |
| Un cambio en el nombre de un inmueble | La grilla guarda el nombre nuevo | "Inmueble 2" → "Edificio Los Álamos" |
| Pulsar **Eliminar** en una fila | La fila y su techo desaparecen; el resto de filas conserva su número y sus datos | Eliminar la fila 7 → la grilla pasa de la 6 a la 8 y el indicador muestra "93 unidades en la grilla · eliminadas: 1" |

## 4. La pieza de IA

- **Opción elegida:** dos modelos de Hugging Face. El principal es
  [`geobase/building-footprint-segmentation`](https://huggingface.co/geobase/building-footprint-segmentation),
  una red U-Net que segmenta huellas (techos) de edificios en imágenes satelitales. El
  segundo es
  [`sawthiha/segformer-b0-finetuned-deprem-satellite`](https://huggingface.co/sawthiha/segformer-b0-finetuned-deprem-satellite),
  un SegFormer que se ejecuta con `pipeline("image-segmentation")` de `transformers`.
- **Por qué:** están entrenados justo para esta tarea, corren gratis en el propio servidor
  de la app (sin clave de API ni cuota) y analizan una manzana en pocos segundos. El
  principal da la lista de inmuebles y SegFormer completa algunos techos grandes.
- **Instrucciones para el modelo (system prompt):** no aplica. Son modelos de segmentación
  de imágenes, no modelos de lenguaje.
- **Cómo se usa:** el modelo principal solo se publica en formato ONNX para la librería
  JavaScript GeoAI.js, así que `pipeline()` no puede cargarlo. La app lo descarga con
  `huggingface_hub` y lo ejecuta con `onnxruntime`, replicando el procesamiento de GeoAI.js:
  1. Descarga las imágenes de Esri World Imagery que cubren el polígono, con zoom 18
     (unos 0,6 m por píxel; en pruebas da mejores resultados que 17 o 19).
  2. Pasa al modelo parches RGB de 256 × 256 píxeles con valores entre 0 y 1. El modelo
     devuelve, para cada píxel, la probabilidad de que sea techo de edificio.
  3. Marca como edificio los píxeles con probabilidad de 0,3 o más.
  4. Pasa a SegFormer las imágenes de la misma zona con zoom 19 (unos 0,3 m por píxel) y
     añade solo las regiones de 300 m² o más que sean muy compactas y toquen un techo ya
     detectado. Sin este filtro, SegFormer junta casas vecinas con sus patios y marca como
     techo la imagen de relleno que Esri sirve sobre el agua.
  5. Dentro del polígono, cuenta cada techo separado como un inmueble y descarta los de
     menos de 25 m².
  6. Calcula el centro de cada techo en latitud y longitud.
  7. Pide a Overpass (OpenStreetMap) los edificios con nombre de la zona. Un techo coincide
     con uno si se solapa al menos el 30 % del techo o del edificio; si coinciden varios,
     gana el de mayor solape. La fila toma ese nombre y es "Edificio" si empieza por Torre,
     Residencias o Edificio (singular o plural, sin distinguir mayúsculas), y "Casa" si
     empieza de otra forma. Un techo sin coincidencia no tiene nombre: es "Edificio" si mide
     300 m² o más, y "Casa" si es más pequeño. Estas reglas no son IA.
- **Limitaciones conocidas:** detecta bien los techos claros, pero se le escapan techos
  oscuros o de teja. A veces divide un edificio en varios trozos y no distingue viviendas
  de comercios o naves. SegFormer cambia pocas unidades: en las pruebas no alteró zonas de
  casas y solo mejoró algunos techos grandes. Si un edificio se divide, varias filas tienen
  el mismo nombre. El tipo depende de lo completo que esté OpenStreetMap: en La Urbina casi
  todos los edificios tienen nombre, pero en zonas sin nombres manda el tamaño del techo, que
  se equivoca con edificios altos de planta pequeña y con naves grandes. Un edificio con otro
  nombre, como "Hotel X", sale como Casa aunque sea grande. Por eso la grilla es editable,
  las filas se pueden eliminar y el resultado es una estimación que la persona revisa.

## 5. Flujo del usuario

1. El usuario entra y ve un mapa grande centrado en la avenida principal de La Urbina
   (Caracas), con un mensaje que le pide dibujar un polígono. Se mueve hasta la zona que le
   interesa. En la barra lateral puede cambiar el centro, el zoom inicial y la altura del mapa.
2. Elige la herramienta de polígono (el pentágono, a la izquierda del mapa), marca los
   vértices de la zona y cierra la figura haciendo clic en el primer punto. El mensaje junto
   al botón confirma "Polígono capturado con N vértices".
3. Pulsa **"Obtener unidades habitacionales"**. Recibe la grilla con los inmuebles
   encontrados y un mensaje con cuántos se detectaron. Revisa cada fila, corrige el tipo,
   completa pisos y apartamentos en los edificios y elimina las filas que sobren. Puede
   contraer la grilla para volver a ver el mapa.

## 6. Restricciones

- Coste cero: modelos abiertos de Hugging Face ejecutados en el servidor, imágenes públicas
  de Esri y datos abiertos de OpenStreetMap (servidores públicos de Overpass), con despliegue
  gratuito en Streamlit Community Cloud (ruta A). No hace falta ninguna clave de API. Si en
  el futuro se añade una, irá en secretos (`st.secrets`) y nunca en el código.
- A Overpass solo se envían los límites de la zona. La API principal de OpenStreetMap no se
  usa: su política la reserva para editar el mapa.
- El servidor gratuito tiene poca memoria: cada modelo se carga una sola vez (15 MB cada
  uno) y se reutiliza entre análisis. `torch` se instala en su versión CPU.
- Sin datos personales reales: la app solo maneja coordenadas y características de
  inmuebles, nunca nombres de propietarios ni de residentes.
- Lenguaje sencillo, para alguien sin conocimientos técnicos.
- Área máxima por análisis: 0,5 km². Así se limita el número de imágenes que hay que
  descargar y el tiempo de espera.
- Los mapas deben mostrar la atribución de OpenStreetMap y de Esri, y la grilla la de
  OpenStreetMap cuando muestra sus nombres.
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
| SegFormer no está disponible | La grilla se llena con el modelo principal y un aviso indica que el segundo modelo no estuvo disponible |
| Un techo coincide con un edificio de OpenStreetMap cuyo nombre empieza por Torre, Residencias o Edificio | La fila toma ese nombre y su tipo es Edificio |
| Un techo que coincide con un nombre que empieza de otra forma | La fila toma ese nombre y su tipo es Casa, aunque el techo sea grande |
| Un techo sin coincidencia | Sin nombre de OpenStreetMap. Su tipo es Edificio si mide 300 m² o más, y Casa si es más pequeño |
| OpenStreetMap no responde | La grilla se llena sin nombres, todas las filas usan la regla del tamaño del techo y un aviso lo indica |
| Pulsar **Eliminar** en una fila | Desaparecen la fila y su techo; se conservan el polígono, las demás filas y sus datos editados |
| Eliminar todas las filas | Mensaje que invita a analizar de nuevo. Un análisis nuevo vuelve a traer todas las unidades |
| Contraer y volver a expandir la grilla | El título sigue visible y los datos editados se mantienen |

## 8. Fuera de alcance (por ahora)

- Clasificar casa / edificio y estimar pisos con un modelo, para no depender de los nombres
  de OpenStreetMap.
- Unir en una sola fila los trozos de un mismo edificio.
- Exportar la grilla a CSV o GeoJSON.
- Guardar polígonos y resultados entre sesiones.
- Analizar varios polígonos a la vez.
- Calcular el total de viviendas de la zona (casas + apartamentos) y mostrarlo en un resumen.
- Cruzar los datos con el catastro o con direcciones postales.

## 9. Despliegue

- **Ruta:** A (Streamlit Community Cloud) → https://extractoruh.streamlit.app/
- **Repositorio:** https://github.com/gatojohn/extractor_uh
