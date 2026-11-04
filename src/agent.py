"""Agente RAG con LangGraph: reformular -> recuperar -> generar, con memoria por hilo.

Uso:
    from src.agent import build_agent, chat
    agent = build_agent()
    respuesta, fuentes = chat(agent, "¿Qué edad necesito para el A2?", thread_id="demo")
"""

from __future__ import annotations

import contextvars
import random
import re
import time
from typing import Annotated, TypedDict

import httpx

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.knowledge_base import get_retriever
from src.prompts import CONTEXT_TEMPLATE, REFORMULATE_PROMPT, SYSTEM_PROMPT

# Modelo Flash más reciente disponible en la API (comprobado con client.models.list()).
# Si está saturado (503) o no disponible, se pasa a los de respaldo.
LLM_MODEL = "gemini-3.8-flash"
LLM_FALLBACKS = ["gemini-flash-latest", "gemini-3.5-flash"]

# Tiempos de espera: cada intento se corta a los LLM_TIMEOUT s y, en total, una llamada
# (reintentos + modelos de respaldo incluidos) no puede pasar de LLM_TOTAL_BUDGET s.
LLM_TIMEOUT = 45
LLM_ATTEMPTS_PER_MODEL = 2
LLM_TOTAL_BUDGET = 300
RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}

RETRIEVAL_K = 5
HISTORY_FOR_REFORMULATION = 6  # mensajes (3 turnos) para reescribir la pregunta
HISTORY_FOR_GENERATION = 10  # mensajes (5 turnos) que ve el modelo al responder


# ---------------------------------------------------------------------------
# Estado
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # historial (persistido por el checkpointer)
    question: str  # pregunta autónoma (reformulada) usada para buscar
    context: list[Document]  # chunks recuperados para la pregunta actual


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
# Dónde se avisa de los reintentos. Por defecto, print (notebook/terminal); la app de
# Streamlit lo cambia por una función que actualiza la interfaz. Es un ContextVar para
# que, con el agente compartido entre sesiones, cada sesión reciba solo sus avisos.
RETRY_NOTIFIER: contextvars.ContextVar = contextvars.ContextVar("retry_notifier", default=None)


def _notify(msg: str) -> None:
    fn = RETRY_NOTIFIER.get()
    if fn is None:
        print(msg, flush=True)
    else:
        fn(msg)


class LLMTimeoutError(TimeoutError):
    """Se agotó el presupuesto de tiempo de una llamada al LLM."""


def _error_code(exc: BaseException) -> int | None:
    """Código HTTP del error (langchain envuelve el ClientError/ServerError de google-genai)."""
    while exc is not None:
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            return code
        exc = exc.__cause__
    return None


def _is_retryable(exc: BaseException) -> bool:
    e = exc
    while e is not None:
        if isinstance(e, (httpx.TimeoutException, httpx.TransportError, TimeoutError)):
            return True
        e = e.__cause__
    return _error_code(exc) in RETRYABLE_CODES


def _describe(exc: BaseException) -> str:
    if (code := _error_code(exc)) is not None:
        return f"HTTP {code}"
    e = exc
    while e is not None:
        if isinstance(e, (httpx.TimeoutException, TimeoutError)):
            return f"sin respuesta en {LLM_TIMEOUT}s"
        e = e.__cause__
    return type(exc).__name__


class ResilientLLM:
    """Gemini con timeout por intento, reintentos visibles y modelos de respaldo.

    Sustituye a los reintentos internos del SDK (silenciosos y sin límite de tiempo total):
    cada reintento o cambio de modelo se imprime, para saber que no está colgado.
    """

    def __init__(self, models: list[str], temperature: float | None = None, name: str = "LLM"):
        self.name = name
        self.models = []
        for model in models:
            # max_retries=0 -> un único intento por llamada al SDK; los reintentos son nuestros.
            kwargs = {"model": model, "max_retries": 0, "timeout": LLM_TIMEOUT}
            if temperature is not None:
                kwargs["temperature"] = temperature
            self.models.append((model, ChatGoogleGenerativeAI(**kwargs)))

    def invoke(self, messages):
        start = time.monotonic()
        last_exc: BaseException | None = None
        for i, (model, llm) in enumerate(self.models):
            if i > 0:
                _notify(f"🔀 [{self.name}] Probando modelo de respaldo {model}…")
            for attempt in range(1, LLM_ATTEMPTS_PER_MODEL + 1):
                if time.monotonic() - start > LLM_TOTAL_BUDGET:
                    raise LLMTimeoutError(
                        f"[{self.name}] Sin respuesta tras {LLM_TOTAL_BUDGET}s "
                        f"(último error: {_describe(last_exc) if last_exc else '-'})"
                    ) from last_exc
                try:
                    return llm.invoke(messages)
                except Exception as exc:
                    if not _is_retryable(exc):
                        raise
                    last_exc = exc
                    if attempt < LLM_ATTEMPTS_PER_MODEL:
                        wait = 2**attempt + random.random()
                        _notify(
                            f"⏳ [{self.name}] {model}: {_describe(exc)} — reintentando en {wait:.0f}s "
                            f"(intento {attempt + 1}/{LLM_ATTEMPTS_PER_MODEL})…"
                        )
                        time.sleep(wait)
                    else:
                        _notify(f"⚠️ [{self.name}] {model}: {_describe(exc)} — se agotaron los intentos.")
        raise LLMTimeoutError(f"[{self.name}] Fallaron todos los modelos: {_describe(last_exc)}") from last_exc


def get_llm(temperature: float | None = None, name: str = "LLM") -> ResilientLLM:
    """Gemini Flash con timeout, reintentos visibles y modelos de respaldo."""
    return ResilientLLM([LLM_MODEL, *LLM_FALLBACKS], temperature=temperature, name=name)


def _text(msg: AnyMessage) -> str:
    """Texto plano de un mensaje (Gemini 3 puede devolver el contenido como lista de bloques)."""
    return msg.text if hasattr(msg, "text") else str(msg.content)


def _format_history(messages: list[AnyMessage]) -> str:
    lines = []
    for m in messages:
        role = "Usuario" if isinstance(m, HumanMessage) else "Asistente"
        lines.append(f"{role}: {_text(m)}")
    return "\n".join(lines)


def cite_label(metadata: dict) -> str:
    """Cita lista para usar: 'Reglamento General de Conductores, art. 4', '…, anexo II',
    '…, arts. 9 y 10', '…, disposición adicional cuarta' o '…, preámbulo'."""
    partes = [a.strip() for a in metadata.get("articulos", metadata.get("articulo", "")).split(",") if a.strip()]
    nums = [a.removeprefix("Artículo ").strip() for a in partes if a.startswith("Artículo ")]
    if nums and len(nums) == len(partes):
        ref = f"art. {nums[0]}" if len(nums) == 1 else f"arts. {', '.join(nums[:-1])} y {nums[-1]}"
    else:
        ref = " y ".join(a[:1].lower() + a[1:] for a in partes) or "preámbulo"
    return f"{metadata['norma']}, {ref}"


def format_context(docs: list[Document]) -> str:
    """Numera cada chunk y le pone delante la cita exacta que debe usar el modelo.

    No se incluye la página: si se le da, el modelo a veces cita «(pág. 12)» en vez
    de la norma y el artículo."""
    return "\n\n".join(
        f"[{i}] CITA: ({cite_label(d.metadata)})\n{d.page_content}" for i, d in enumerate(docs, 1)
    )


_PAGE_REF = re.compile(r"\s*\((?:pág(?:ina)?\.?|p\.)\s*(\d+)\)", re.IGNORECASE)
_FRAGMENT_REF = re.compile(r"\s*\[(?:fragmento\s*)?(\d+)\]|\s*\(fragmento\s*(\d+)\)", re.IGNORECASE)


def fix_citations(answer: str, docs: list[Document]) -> str:
    """Red de seguridad: cambia las citas que quedan como «(pág. 12)» o «[3]» por
    «(norma, artículo)» del chunk correspondiente; si no corresponde a ninguno, se quitan."""

    def by_page(m: re.Match) -> str:
        n = int(m.group(1))
        for d in docs:
            ini = d.metadata.get("pagina")
            fin = d.metadata.get("pagina_fin", ini)
            if ini is not None and ini <= n <= fin:
                label = cite_label(d.metadata)
                return "" if label in answer else f" ({label})"
        return ""

    def by_index(m: re.Match) -> str:
        n = int(m.group(1) or m.group(2))
        if 1 <= n <= len(docs):
            label = cite_label(docs[n - 1].metadata)
            return "" if label in answer else f" ({label})"
        return m.group(0)

    return _FRAGMENT_REF.sub(by_index, _PAGE_REF.sub(by_page, answer))


_ROMAN = r"[IVXLC]+"


def _is_cited(answer: str, articulo: str) -> bool:
    """¿Cita la respuesta este artículo/anexo/disposición? (p. ej. 'art. 4', 'artículo 118', 'anexo II')."""
    a = answer.lower()
    if m := re.match(r"Artículo (\d+(?: bis| ter)?)", articulo):
        return re.search(rf"\barts?(?:[íi]culos?|\.)?\s*{re.escape(m.group(1))}\b", a) is not None
    if m := re.match(rf"Anexo ({_ROMAN})", articulo):
        return re.search(rf"\banexo\s+{m.group(1).lower()}\b", a) is not None
    if articulo.startswith("Disposición"):
        return articulo.lower() in a
    if articulo == "Preámbulo":
        return "preámbulo" in a
    return False


def extract_sources(answer: str, docs: list[Document]) -> list[dict]:
    """Fuentes recuperadas (sin duplicar norma+artículo), indicando si la respuesta las cita."""
    seen, out = set(), []
    for d in docs:
        m = d.metadata
        key = (m["norma"], m["articulo"])
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "norma": m["norma"],
                "referencia": m.get("referencia", ""),
                "articulo": m["articulo"],
                "titulo": m.get("titulo", ""),
                "pagina": m.get("pagina"),
                "citada": _is_cited(answer, m["articulo"]),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Grafo
# ---------------------------------------------------------------------------
def build_agent(k: int = RETRIEVAL_K, checkpointer=None):
    """Construye y compila el grafo reformular -> recuperar -> generar."""
    llm = get_llm(name="generar")
    llm_reformulate = get_llm(temperature=0, name="reformular")
    retriever = get_retriever(k=k)

    def reformular(state: AgentState) -> dict:
        """Convierte la última pregunta en autónoma (usando el historial, si lo hay) y la
        pasa al vocabulario del BOE. Se hace también en la primera pregunta: las consultas
        coloquiales ("¿cómo paso al A?") recuperan mucho peor que las formales
        ("requisitos para obtener el permiso de la clase A siendo titular del A2")."""
        last = _text(state["messages"][-1])
        history = state["messages"][:-1][-HISTORY_FOR_REFORMULATION:]
        prompt = REFORMULATE_PROMPT.format(
            history=_format_history(history) or "(sin historial: es la primera pregunta)", question=last
        )
        standalone = _text(llm_reformulate.invoke(prompt)).strip().strip('"«»')
        return {"question": standalone or last}

    def recuperar(state: AgentState) -> dict:
        """Busca en ChromaDB (híbrido BM25 + vectores) con la pregunta reformulada."""
        return {"context": retriever.invoke(state["question"])}

    def generar(state: AgentState) -> dict:
        """Responde con system prompt + historial + contexto recuperado."""
        last = _text(state["messages"][-1])
        history = state["messages"][:-1][-HISTORY_FOR_GENERATION:]
        user_turn = CONTEXT_TEMPLATE.format(
            context=format_context(state["context"]), question=last, standalone=state["question"]
        )
        # El contexto solo se inyecta en el turno actual: el historial guardado queda limpio
        # (preguntas y respuestas), así no crece con fragmentos del BOE de turnos anteriores.
        messages = [SystemMessage(SYSTEM_PROMPT), *history, HumanMessage(user_turn)]
        answer = fix_citations(_text(llm.invoke(messages)), state["context"])
        return {"messages": [AIMessage(answer)]}

    graph = StateGraph(AgentState)
    graph.add_node("reformular", reformular)
    graph.add_node("recuperar", recuperar)
    graph.add_node("generar", generar)
    graph.add_edge(START, "reformular")
    graph.add_edge("reformular", "recuperar")
    graph.add_edge("recuperar", "generar")
    graph.add_edge("generar", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def chat(agent, mensaje: str, thread_id: str = "default", solo_citadas: bool = True):
    """Envía un mensaje al agente dentro de un hilo de conversación.

    Devuelve (respuesta, fuentes). Con solo_citadas=True, las fuentes son las
    recuperadas que la respuesta cita expresamente; si no cita ninguna (p. ej.
    pregunta fuera de tema o sin respuesta en el contexto), la lista queda vacía.
    """
    config = {"configurable": {"thread_id": thread_id}}
    state = agent.invoke({"messages": [HumanMessage(mensaje)]}, config=config)
    respuesta = _text(state["messages"][-1])
    fuentes = extract_sources(respuesta, state["context"])
    if solo_citadas:
        fuentes = [f for f in fuentes if f["citada"]]
    return respuesta, fuentes
