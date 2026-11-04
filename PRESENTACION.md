# Guion de la presentación (≈ 8-9 minutos)

**Asistente de Tráfico y Permisos de Moto**: RAG sobre el BOE con Gemini, ChromaDB, LangGraph y Streamlit.

| # | Parte | Tiempo | Acumulado |
|---|---|---|---|
| 1 | El problema y el dominio | 1 min | 1:00 |
| 2 | Fuentes y base de conocimiento | 2 min | 3:00 |
| 3 | El agente: reformular → recuperar → generar | 1:30 min | 4:30 |
| 4 | Demo en vivo | 3 min | 7:30 |
| 5 | Limitaciones, mejoras y cierre | 1 min | 8:30 |

Si vas justo de tiempo, recorta la parte 2 (quédate con la historia del A2) y no la demo. Si te sobra, enseña el notebook en la parte 3.

---

## Antes de empezar (5 minutos antes)

- [ ] **Abre la app y haz una pregunta cualquiera**, para despertarla. En la capa gratuita se duerme y la primera carga tarda. Después pulsa «Nueva conversación».
- [ ] Ten abierta una **segunda pestaña con una conversación ya hecha** (las 5 preguntas de la demo) por si Gemini va lento.
- [ ] Ten el **notebook abierto en la sección 4** con las salidas ya ejecutadas. Es el plan B y además enseña la «pregunta interpretada».
- [ ] Ten abierto el **README en GitHub**, con el diagrama de arquitectura.

---

## 1. El problema y el dominio · 1 min

**Qué decir:**
- «Quien quiere sacarse el carnet de moto se hace siempre las mismas preguntas: qué edad necesito para el A2, si puedo llevar una moto con el carnet de coche, cómo paso del A2 al A.»
- «Las respuestas están en el BOE, pero repartidas en tres normas largas y en lenguaje jurídico.»
- «Un LLM normal responde de memoria. Puede estar desactualizado o mezclar normativa de otros países, y no te dice de dónde lo saca. En un tema con consecuencias legales, eso es un problema.»
- «Por eso he hecho un asistente RAG: busca los artículos relevantes en el BOE, responde solo con ellos y cita la norma y el artículo para que puedas comprobarlo.»

## 2. Fuentes y base de conocimiento · 2 min

**Qué decir:**
- **Fuentes**: «Tres textos consolidados del BOE, o sea, la versión vigente: el Reglamento General de Conductores (permisos y edades), el de Circulación (casco, velocidades) y la Ley de Tráfico (infracciones y puntos). En total, 913 páginas.»
- **Limpieza**: «Quito cabeceras y pies, el índice y unas 600 páginas que son imágenes de señales. Me quedan 272 páginas con texto útil.»
- **La historia del A2.** Es el momento clave de esta parte, cuéntalo como una historia:
  - «En la primera versión, pregunté "¿qué edad necesito para el A2?" y no aparecía la respuesta.»
  - «El motivo: el PDF extrae la tabla de permisos como líneas sueltas, "A2", luego la descripción y luego "18". El trozo con la edad no decía ni "permiso" ni "edad".»
  - «Lo arreglé de dos formas: reescribo cada fila de la tabla como una frase completa ("Edad mínima para el permiso de la clase A2: 18 años") y le pongo a cada trozo una cabecera con la norma y el artículo. Después de eso, sale el primero.»
- **Chunking**: «Trozos de 1000 caracteres con solapamiento de 150, cortando preferentemente por artículos. Salen 1324 chunks, cada uno con metadatos de norma, artículo y página para poder citar.»
- **Búsqueda**: «Embeddings de Gemini en ChromaDB, combinados con BM25, que busca por palabras clave. Los embeddings entienden el significado, pero confunden códigos cortos como "A2" o "B", y ahí BM25 acierta. Los fusiono y me quedo con los 5 mejores.»

**Qué enseñar:** el diagrama de construcción de la base de conocimiento en el README o, si da tiempo, la tabla del notebook con el chunk del A2.

## 3. El agente · 1:30 min

**Qué decir:** (enseña el diagrama de arquitectura del README)
- «El agente es un grafo de LangGraph con tres pasos.»
- **Reformular**: «Primero reescribo la pregunta para que se entienda sola y en el vocabulario del BOE. "Carnet de coche" pasa a ser "permiso de la clase B". Y si pregunto "¿y cuánto tiempo tengo que tener el anterior?", lo convierte en "¿cuánto tiempo hay que tener el A2 para obtener el A?". Sin esto, la búsqueda no encontraría nada.»
- **Recuperar**: «Busca los 5 fragmentos con esa pregunta reformulada.»
- **Generar**: «Gemini responde solo con esos fragmentos. El system prompt le pide: tono de profesor de autoescuela, la respuesta directa primero, citar norma y artículo, decir "no lo encuentro" en vez de inventar, un aviso legal breve y rechazar con amabilidad lo que no sea de tráfico.»
- **Memoria**: «LangGraph guarda el historial de cada conversación. Guardo solo preguntas y respuestas, no los fragmentos del BOE, para que la memoria no crezca sin control.»
- **Robustez** (frase corta): «La capa gratuita de Gemini se satura a menudo. Por eso tiene reintentos, modelos de respaldo y un límite de tiempo, y la app muestra en pantalla cuándo está reintentando.»

## 4. Demo en vivo · 3 min

Hazla en la **app de Streamlit**, en **una sola conversación** y en este orden. Las preguntas 1, 2 y 4 están como botones en la barra lateral; las otras se escriben.

| # | Pregunta | Qué debe responder | Qué señalar |
|---|---|---|---|
| 1 | **¿Qué edad necesito para el carnet A2?** *(botón)* | **18 años** (Reglamento General de Conductores, art. 4), más los límites del A2: 35 kW y 0,2 kW/kg | La respuesta va **directa al grano** y cita el artículo. Abre el desplegable **«Fuentes»**: norma, artículo y página. Es el chunk que antes fallaba. |
| 2 | **Después de tener el A2, ¿cómo paso al A?** *(botón)* | **2 años** con el A2 y **superar una formación** (art. 5) | Pregunta coloquial. Sin la reformulación, en las pruebas respondía «no lo encuentro». |
| 3 | **¿Y cuánto tiempo tengo que tener el anterior?** *(escribirla)* | **2 años** con el A2 (art. 5) | **Prueba de memoria**: la pregunta sola no se entiende. Sabe que «el anterior» es el A2 porque recuerda la conversación. |
| 4 | **¿Es obligatorio el casco en moto?** *(botón)* | **Sí**, para conductor y pasajero, con la excepción de vehículos con estructura de autoprotección, e infracción grave (arts. 116 y 118 del Reglamento de Circulación, art. 47 de la Ley) | Cambia de tema y **no arrastra** la conversación anterior. Mezcla dos normas distintas. |
| 5 | **¿Cuál es la mejor receta de paella?** *(escribirla)* | Rechaza con amabilidad y propone preguntas de ejemplo | **Fuera de tema**: no inventa, no cita normas, no añade aviso legal y no muestra fuentes. |

**Para cerrar la demo** (opcional, 10 s): pulsa «Nueva conversación» y di «cada sesión tiene su propia memoria».

**Si Gemini va lento** (verás «Gemini está saturado, reintentando…»):
- No te quedes callado. Aprovecha: «Esto es la capa gratuita de Gemini saturada; veis que la app avisa de cada reintento y pasa a un modelo de respaldo, no está colgada.»
- Si una respuesta pasa de ~1 minuto, cambia a la **pestaña con la conversación ya hecha** o al **notebook** y sigue desde ahí.

**Si da tiempo, enseña en el notebook** la línea «🔎 PREGUNTA INTERPRETADA» del ejemplo de memoria. Es la prueba más clara de que la reformulación usa el historial.

## 5. Limitaciones, mejoras y cierre · 1 min

**Qué decir:**
- «He revisado las respuestas a mano contra el BOE, y no todo es perfecto. El caso más claro: en "¿qué motos puedo llevar con el carnet de coche?", los requisitos están bien (más de 3 años, solo en España, motos del A1). Pero el buscador no trae el artículo con los límites del A1 y el modelo los toma de una disposición sobre permisos antiguos: dice 0,11 kW/kg en vez de 0,1. El modelo hace lo que se le pide, usar solo el contexto, pero el contexto no era el bueno. La mejora sería recuperar el artículo completo o añadir un *reranker*.»
- «Otras limitaciones: solo tres normas (falta, por ejemplo, la Orden que regula la formación del A2 al A), las señales son imágenes y no están indexadas, y la memoria se pierde si la app se reinicia.»
- «Mejoras: *reranking*, evaluación automática con un conjunto de preguntas, más fuentes de la DGT y memoria persistente.»
- **Cierre**: «En resumen: un asistente que responde dudas reales de motoristas con el texto vigente del BOE, que dice de qué artículo sale cada dato y que reconoce cuándo no sabe algo. Está desplegado en Streamlit Cloud y el código y el notebook están en GitHub. ¿Preguntas?»

---

## Preguntas que te pueden hacer

| Pregunta | Respuesta corta |
|---|---|
| ¿Por qué no usar directamente ChatGPT o Gemini? | Responden de memoria, sin fuente y quizá con datos desactualizados. Aquí cada dato sale del BOE vigente y va citado. |
| ¿Cómo evitas las alucinaciones? | El system prompt obliga a usar solo el contexto y a decir «no lo encuentro». Las citas salen de los metadatos, no las inventa el modelo. Y solo se muestran como fuentes los artículos que la respuesta cita. |
| ¿Por qué búsqueda híbrida? | Los embeddings distinguen mal «A2» de «A1» o «B»; BM25 los acierta por palabra exacta. RRF combina las dos listas. |
| ¿Por qué k = 5? | Cabe el artículo principal y los relacionados (p. ej. art. 4 y 5) sin llenar el contexto de ruido. Son unos 5.000 caracteres. |
| ¿Por qué chunks de 1000 caracteres? | Un apartado de un artículo cabe entero y el embedding representa una sola idea. Más grande mezclaría temas; más pequeño cortaría las frases. |
| ¿Por qué 768 dimensiones? | El modelo de embeddings permite reducirlas perdiendo muy poca calidad, y la base ocupa 4 veces menos, lo que importa porque va en el repositorio. |
| ¿Cómo funciona la memoria? | LangGraph guarda el estado por `thread_id`. Cada sesión de Streamlit tiene el suyo. Se guardan solo preguntas y respuestas. |
| ¿Qué pasa si el BOE cambia? | Se descargan los PDF nuevos y se reindexa. La indexación es incremental: solo procesa los trozos que han cambiado. |
| ¿Por qué tarda a veces? | La capa gratuita de Gemini devuelve errores 503 en horas punta. Hay reintentos y modelos de respaldo, y la app lo muestra. |
