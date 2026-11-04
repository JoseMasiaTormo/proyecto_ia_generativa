"""Interfaz de chat en Streamlit para el asistente de normativa de tráfico.

Toda la lógica (RAG, memoria, citas) está en src/agent.py; aquí solo va la interfaz.
Ejecutar en local:  streamlit run app.py
"""

# --- Arreglo de sqlite para Chroma -------------------------------------------------
# Chroma necesita sqlite3 >= 3.35. Si el Python del servidor trae uno más antiguo
# (pasa en algunas imágenes de Streamlit Cloud), se sustituye por pysqlite3-binary.
# Tiene que hacerse ANTES de importar chromadb (lo importa src.knowledge_base).
import sqlite3
import sys

if tuple(map(int, sqlite3.sqlite_version.split("."))) < (3, 35, 0):
    try:
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
    except ImportError:
        pass  # Chroma avisará con un error claro

import os
import uuid

import streamlit as st

# --- API key: st.secrets (Streamlit Cloud) o .env (local) -------------------------
# Se define antes de importar src/, que hace load_dotenv() sin sobrescribir variables.
try:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:  # no hay secrets.toml: se usará el .env
    pass

from src.agent import RETRY_NOTIFIER, LLMTimeoutError, build_agent, chat  # noqa: E402

st.set_page_config(page_title="Asistente de Tráfico y Permisos de Moto", page_icon="🏍️")

EJEMPLOS = [
    "¿Qué edad necesito para el carnet A2?",
    "¿Qué motos puedo llevar con el carnet de coche?",
    "Después de tener el A2, ¿cómo paso al A?",
    "¿Es obligatorio el casco en moto?",
    "¿Qué infracciones quitan puntos del carnet?",
]


@st.cache_resource(show_spinner="Cargando la base de conocimiento…")
def get_agent():
    """Un único agente para toda la app (usa chroma_db/ ya indexado; no reindexa).
    La memoria va por thread_id, así que cada sesión tiene su propia conversación."""
    return build_agent()


def new_conversation() -> None:
    st.session_state.thread_id = f"web-{uuid.uuid4().hex[:12]}"
    st.session_state.messages = []


def _md(text: str) -> str:
    return text.replace("$", r"\$")  # evita que Streamlit lo interprete como LaTeX


def show_sources(fuentes: list[dict]) -> None:
    if not fuentes:
        return
    with st.expander(f"Fuentes ({len(fuentes)})"):
        for f in fuentes:
            st.markdown(
                f"- **{f['norma']}** · {f['articulo']} · pág. {f['pagina']}  \n"
                f"  <small>{f['referencia']}</small>",
                unsafe_allow_html=True,
            )


if "thread_id" not in st.session_state:
    new_conversation()

# --- Barra lateral ------------------------------------------------------------------
with st.sidebar:
    st.header("🏍️ Sobre el asistente")
    st.markdown(
        "Responde dudas sobre **normativa de tráfico** y **permisos de moto** "
        "(AM, A1, A2, A y B) a partir de los textos consolidados del BOE:\n"
        "- Reglamento General de Conductores\n"
        "- Reglamento General de Circulación\n"
        "- Ley de Tráfico y Seguridad Vial\n\n"
        "Cada respuesta indica la norma y el artículo de donde sale."
    )

    st.subheader("Prueba con…")
    for i, ejemplo in enumerate(EJEMPLOS):
        if st.button(ejemplo, key=f"ej{i}", width="stretch"):
            st.session_state.pending = ejemplo

    st.divider()
    if st.button("🔄 Nueva conversación", width="stretch", type="primary"):
        new_conversation()
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ **Aviso legal:** información orientativa generada por IA a partir del BOE. "
        "La normativa puede cambiar y no sustituye a la DGT ni al asesoramiento de un "
        "profesional (autoescuela, abogado)."
    )

# --- Chat ---------------------------------------------------------------------------
st.title("Asistente de Tráfico y Permisos de Moto")
st.caption("Pregunta sobre carnets de moto, edades, requisitos, casco, velocidades, puntos…")

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🏍️"):
        st.markdown(
            "¡Hola! Soy tu asistente de normativa de tráfico. Pregúntame lo que quieras sobre "
            "permisos de moto y normas de circulación, o elige un ejemplo en la barra lateral."
        )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🏍️" if msg["role"] == "assistant" else None):
        st.markdown(_md(msg["content"]))
        if msg["role"] == "assistant":
            show_sources(msg.get("fuentes", []))

pregunta = st.chat_input("Escribe tu pregunta…") or st.session_state.pop("pending", None)

if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(_md(pregunta))

    with st.chat_message("assistant", avatar="🏍️"):
        status = st.status("Consultando la normativa…", expanded=False)

        def avisar(msg: str) -> None:
            # Reintentos del LLM (servidores de Gemini saturados): se muestran para que
            # se vea que la app sigue trabajando y no se ha colgado.
            status.update(label="Gemini está saturado, reintentando…", state="running", expanded=True)
            status.write(msg)

        token = RETRY_NOTIFIER.set(avisar)
        try:
            respuesta, fuentes = chat(get_agent(), pregunta, thread_id=st.session_state.thread_id)
        except LLMTimeoutError:
            status.update(label="Sin respuesta", state="error", expanded=False)
            respuesta, fuentes = None, []
            st.error(
                "Los servidores de Gemini están saturados y no han respondido a tiempo. "
                "Vuelve a intentarlo en unos minutos."
            )
        except Exception as e:  # noqa: BLE001
            status.update(label="Error", state="error", expanded=False)
            respuesta, fuentes = None, []
            st.error(f"Ha ocurrido un error al generar la respuesta: {e}")
        else:
            status.update(label="Respuesta lista", state="complete", expanded=False)
            st.markdown(_md(respuesta))
            show_sources(fuentes)
        finally:
            RETRY_NOTIFIER.reset(token)

    if respuesta is not None:
        st.session_state.messages.append({"role": "assistant", "content": respuesta, "fuentes": fuentes})
    else:
        # La pregunta sin respuesta no se queda en el historial visible.
        st.session_state.messages.pop()
