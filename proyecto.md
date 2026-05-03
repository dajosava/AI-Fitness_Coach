# Documentación del proyecto (archivo de referencia)

Este archivo resume **qué hace** la aplicación, **cómo funciona** por dentro y **cómo modificarla**. Sirve como guía rápida si archivás el repositorio, lo compartís con otra persona o retomás el trabajo más adelante.

---

## Qué hace el proyecto

Es un **entrenador / coach de fitness con IA**: interfaz web con **Streamlit** donde el usuario ingresa datos (edad, peso, objetivos, días preferidos, etc.) y el sistema genera un **plan de entrenamiento personalizado** en texto, en **español** (orientado a América Latina / Costa Rica en los prompts).

Incluye flujo para **analizar comentarios del usuario** y **ajustar el plan**, además de pasos de **seguimiento de progreso** y un **mensaje motivacional**. Todo eso corre en cadena dentro de un **grafo LangGraph** que llama repetidamente al mismo modelo de lenguaje (**OpenAI**, `gpt-4o-mini` por defecto).

**Importante:** no sustituye consejo médico ni el de un profesional certificado; las recomendaciones son generales.

---

## Cómo funciona

### Stack principal

| Pieza | Rol |
|--------|-----|
| `main.py` | Toda la lógica: estado del grafo, agentes, clase `AIFitnessCoach`, interfaz Streamlit. |
| **LangGraph** | Orquesta los pasos como nodos enlazados en secuencia hasta `END`. |
| **LangChain** | Plantillas de prompt (`ChatPromptTemplate`), cadena con el LLM y parser de texto. |
| **OpenAI** | Modelo vía `langchain_openai.ChatOpenAI`. |
| **python-dotenv** | Carga variables desde `.env` en desarrollo local. |

### Estado del grafo (`State`)

Es un `TypedDict` con:

- `user_data`: diccionario con el perfil y datos del formulario (y enriquecido por el primer agente).
- `fitness_plan`: texto del plan actual.
- `feedback`: texto de retroalimentación (puede venir vacío en el primer recorrido).
- `progress`: lista de textos con actualizaciones de progreso.
- `messages`: historial de mensajes LangChain (humano + IA), con `add_messages` para acumular.

### Flujo de los nodos (orden fijo)

1. **user_input** — Normaliza / enriquece el perfil; intenta devolver JSON con claves en inglés (`age`, `weight`, …) y valores en español.
2. **routine_generation** — Genera el plan semanal detallado.
3. **feedback_collection** — Analiza el plan + feedback (en la primera ejecución el feedback puede estar vacío).
4. **routine_adjustment** — Reescribe el plan según el feedback.
5. **progress_monitoring** — Resume progreso y sugiere ajustes.
6. **motivation** — Mensaje motivacional breve.

El punto de entrada es siempre `user_input`. La clase `AIFitnessCoach` compila el grafo en `create_graph()` y `run()` invoca `self.graph.invoke(initial_state)` con los datos del formulario o con `{"feedback": "..."}` en la pestaña de actualización.

### Interfaz Streamlit

- `main()` configura la página, aplica **CSS** para que las pastillas del `st.multiselect` tengan texto oscuro sobre el color primario del tema, y define pestañas, inputs y botones.
- Al pulsar **Generar plan** se arma `user_data` y se llama `fitness_coach.run(user_data)`.
- Al pulsar **Actualizar plan** se llama `run({"feedback": feedback})` (el flujo completo del grafo igual se ejecuta; el diseño actual no salta nodos).

### Tema y colores

- **`.streamlit/config.toml`** — Tema oscuro y `primaryColor` (acentos de la UI de Streamlit).
- **CSS en `main()`** — Legibilidad de las etiquetas del multiselect.

### Despliegue (p. ej. Railway)

- **`Procfile`** — Arranca Streamlit con `$PORT` y `0.0.0.0`.
- **`runtime.txt`** — Versión de Python para el build.
- En el servidor hay que definir **`OPENAI_API_KEY`** en variables de entorno (no subir `.env` al repositorio).

---

## Cómo modificar el proyecto

### Cambiar el modelo o la temperatura

En `main.py`, función `get_openai_llm()`:

- Parámetro `model=` (por ejemplo otro modelo de OpenAI compatible con la API de chat).
- `temperature=` (0 = más determinista).

### Cambiar a modelo local (Ollama)

En `AIFitnessCoach.__init__` está comentada la línea de `get_ollama_llm()`. Para usarla hace falta importar `ChatOllama` (p. ej. desde `langchain_community.chat_models`), tener **Ollama** corriendo en la máquina y ajustar dependencias; en **Railway** no suele haber Ollama salvo que montes otro servicio.

### Cambiar tono o idioma de las respuestas del LLM

- Constante **`LOCALE_ES_CR`**: instrucciones globales de idioma y estilo.
- Texto dentro de cada **`ChatPromptTemplate.from_template(...)`** en las funciones `*_agent`: ahí se cambian reglas, estructura del plan, advertencias médicas, etc.

### Cambiar campos del formulario

En `main()` (pestaña “Crear plan”): widgets `st.number_input`, `st.selectbox`, etc. y el diccionario **`user_data`** que se envía a `run()` deben mantenerse alineados. Si renombrás claves, revisá que los prompts que usan `user_data` sigan teniendo sentido (y el agente `user_input` que pide claves JSON en inglés).

### Cambiar el orden del pipeline o añadir un paso

En `AIFitnessCoach.create_graph()`:

- `workflow.add_node("nombre", ...)`
- `workflow.add_edge(...)` y eventualmente `set_entry_point`.

Cualquier nodo nuevo debe recibir y devolver el mismo tipo de `State` (o un superconjunto coherente).

### Tema visual (colores globales Streamlit)

Editar **`.streamlit/config.toml`** (`primaryColor`, `backgroundColor`, etc.).

### Pastillas del multiselect (color del texto)

El bloque **`st.markdown(..., unsafe_allow_html=True)`** con `<style>` al inicio de `main()`; los selectores apuntan a `data-testid="stMultiSelect"` y `data-baseweb="tag"`. Si una versión futura de Streamlit cambia el DOM, puede hacer falta actualizar el CSS.

### Dependencias

**`requirements.txt`** — Añadir o fijar versiones aquí; reinstalar con `pip install -r requirements.txt`.

---

## Archivos clave (mapa rápido)

| Archivo | Contenido |
|---------|-----------|
| `main.py` | App completa: LLM, grafo, agentes, UI. |
| `requirements.txt` | Paquetes Python. |
| `.env` | Solo local; `OPENAI_API_KEY` (no versionar). |
| `.gitignore` | Excluye `.env`, `.venv`, etc. |
| `.streamlit/config.toml` | Tema Streamlit. |
| `Procfile` | Comando de arranque para plataformas tipo Railway. |
| `runtime.txt` | Versión de Python del contenedor. |
| `README.md` | Instalación, uso, despliegue (inglés). |
| `LICENSE` | Licencia del repositorio. |
| `proyecto.md` | Este documento de archivo / referencia interna. |

---

## Cómo ejecutarlo en local

```bash
python -m venv .venv
# Activar el venv según tu OS
pip install -r requirements.txt
```

Crear `.env` en la raíz:

```env
OPENAI_API_KEY=sk-...
```

```bash
streamlit run main.py
```

---

## Para archivar el proyecto

1. **Repositorio Git** con último commit estable; **sin** secretos en el historial (revisar que `.env` no esté trackeado).
2. Conservar **`proyecto.md`** + **`README.md`** para contexto humano y para despliegue.
3. Anotar en un gestor de contraseñas o en la plataforma de deploy dónde quedó la **`OPENAI_API_KEY`** de producción.
4. Si usás **Railway** (u otro PaaS): exportar o anotar URL del servicio, variables y política de facturación.
5. Opcional: etiqueta **git tag** (ej. `v1.0-archivo`) marcando la versión archivada.

---

*Última referencia: documentación orientada al estado del código en este repositorio. Si el código cambia, conviene actualizar este archivo en los mismos cambios.*
