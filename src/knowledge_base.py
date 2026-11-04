"""Base de conocimiento: carga, limpieza, chunking e indexación de la normativa
de tráfico española (BOE) en ChromaDB con embeddings de Google Gemini.

Uso rápido:
    from src.knowledge_base import build_knowledge_base, get_retriever
    build_knowledge_base()          # indexa solo si la colección no está completa
    retriever = get_retriever(k=5)
    retriever.invoke("edad mínima para el permiso A2")
"""

from __future__ import annotations

import bisect
import hashlib
import os
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "normativa_trafico"

# Modelo de embeddings de Gemini (verificado con client.models.list()).
EMBEDDING_MODEL = "models/gemini-embedding-001"
# gemini-embedding-001 usa Matryoshka (MRL): 768 dimensiones mantienen casi toda
# la calidad de 3072 y reducen 4x el tamaño de chroma_db/ (que se versiona en git).
EMBEDDING_DIM = 768

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Límites de la capa gratuita: lotes pequeños + pausa entre lotes + reintentos.
BATCH_SIZE = 20
PAUSE_BETWEEN_BATCHES = 15  # segundos
MAX_RETRIES = 6
INITIAL_BACKOFF = 30  # segundos (se duplica en cada reintento)

# Documentos de la base de conocimiento.
DOCUMENTS = {
    "BOE-A-2009-9481-consolidado.pdf": {
        "norma": "Reglamento General de Conductores",
        "referencia": "RD 818/2009 (BOE-A-2009-9481)",
    },
    "BOE-A-2003-23514-consolidado.pdf": {
        "norma": "Reglamento General de Circulación",
        "referencia": "RD 1428/2003 (BOE-A-2003-23514)",
    },
    "BOE-A-2015-11722-consolidado.pdf": {
        "norma": "Ley de Tráfico y Seguridad Vial",
        "referencia": "RDL 6/2015 (BOE-A-2015-11722)",
    },
}

load_dotenv(ROOT_DIR / ".env")


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
_HEADER_FOOTER_RE = re.compile(
    r"^\s*(BOLET[ÍI]N OFICIAL DEL ESTADO|LEGISLACI[ÓO]N CONSOLIDADA|P[áa]gina\s+\d+)\s*$",
    re.MULTILINE,
)
# Línea de índice: "Artículo 4. Edad mínima ............ 12"
_TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$", re.MULTILINE)
_LOWER = "a-záéíóúñü"
# Palabra cortada con guion al final de línea: "conduc-\nción" -> "conducción"
_HYPHEN_RE = re.compile(rf"([{_LOWER}])-\n([{_LOWER}])")
# Salto de línea "blando" (la frase continúa en minúscula en la línea siguiente).
_SOFT_BREAK_RE = re.compile(rf"(?<![.:;])\n(?=[{_LOWER}(«\"])")

MIN_PAGE_CHARS = 80  # páginas con menos texto útil = imágenes (p. ej. señales)


_PERMIT_TABLE_HEADER = "Clase de permiso Vehículos que autoriza a conducir Edad: años cumplidos Excepciones"
_PERMIT_CLASS_RE = re.compile(r"^(AM|A1|A2|B|B \+ E|C1|C1 \+ E|C|C \+ E|D1|D1 \+ E|D|D \+ E)$")
_PERMIT_CLASS_A_RE = re.compile(r"^A (Motocicletas y triciclos de motor\.) (\d{2}) (.*)$")
_PERMIT_AGE_RE = re.compile(r"^(1[4-9]|2[0-4])$")


def normalize_permit_table(text: str) -> str:
    """Convierte la tabla de clases de permiso (art. 4 del Reglamento General de
    Conductores) en filas autocontenidas.

    pypdf la extrae como líneas sueltas ("A2" / descripción / "18"), y la edad
    queda separada de la clase. Cada fila se reescribe como
    "Permiso de la clase A2. Vehículos que autoriza a conducir: ... Edad mínima
    para el permiso de la clase A2: 18 años cumplidos." para que un chunk
    contenga la clase, los vehículos y la edad juntos.
    """
    start = text.find(_PERMIT_TABLE_HEADER)
    if start == -1:
        return text
    end = text.find("\n3. Para la conducción profesional", start)
    if end == -1:
        return text
    out, current = [], None
    for line in text[start:end].split("\n"):
        if line == _PERMIT_TABLE_HEADER:
            continue  # cabecera de la tabla (se repite al cambiar de página)
        if m := _PERMIT_CLASS_RE.match(line):
            current = m.group(1)
            out.append(f"Permiso de la clase {current}. Vehículos que autoriza a conducir:")
        elif m := _PERMIT_CLASS_A_RE.match(line):
            current = "A"
            out.append(f"Permiso de la clase A. Vehículos que autoriza a conducir: {m.group(1)}")
            out.append(f"Edad mínima para el permiso de la clase A: {m.group(2)} años cumplidos.")
            out.append(f"Excepciones: {m.group(3)}")
        elif current and (m := _PERMIT_AGE_RE.match(line)):
            out.append(f"Edad mínima para el permiso de la clase {current}: {m.group(1)} años cumplidos.")
        else:
            out.append(line)
    return text[:start] + "\n".join(out) + text[end:]


def is_index_page(raw_text: str) -> bool:
    """Las páginas del índice del BOE están llenas de líneas con puntos guía."""
    return len(_TOC_LINE_RE.findall(raw_text)) >= 5


def clean_text(text: str) -> str:
    """Quita cabeceras/pies del BOE, une palabras cortadas y normaliza espacios."""
    text = text.replace(" ", " ").replace("­", "")
    text = _HEADER_FOOTER_RE.sub("", text)
    text = _HYPHEN_RE.sub(r"\1\2", text)
    # Normalizar espacios en cada línea y eliminar líneas vacías.
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    text = "\n".join(ln for ln in lines if ln)
    # Unir líneas cortadas a mitad de frase para que los chunks sean prosa continua.
    text = _SOFT_BREAK_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Detección de artículos / disposiciones / anexos
# ---------------------------------------------------------------------------
_ORDINAL = r"(?:única|único|primera|segunda|tercera|cuarta|quinta|sexta|séptima|octava|novena|décima|\w+)"
_HEADING_RE = re.compile(
    r"^(?:"
    r"(?P<art>Artículo\s+(?:\d+(?:\s+(?:bis|ter|quater|quinquies))?|único))\."
    r"|(?P<disp>Disposición\s+(?:adicional|transitoria|derogatoria|final)\s+" + _ORDINAL + r")\."
    r"|(?P<anexo>ANEXO\s+[IVXLC]+)\b"
    r")",
    re.MULTILINE,
)


def _normalize_heading(m: re.Match) -> str:
    label = m.group("art") or m.group("disp") or m.group("anexo")
    label = re.sub(r"\s+", " ", label)
    if label.startswith("ANEXO"):
        label = "Anexo " + label.split()[1]
    return label


def _heading_title(text: str, m: re.Match, max_len: int = 150) -> str:
    """Línea completa del encabezado: 'Artículo 4. Clases de permiso de ... obtenerlo.'"""
    line_end = text.find("\n", m.start())
    line = text[m.start() : line_end if line_end != -1 else len(text)].strip()
    if line.startswith("ANEXO"):
        line = "Anexo" + line[len("ANEXO") :]
    return line if len(line) <= max_len else line[:max_len].rsplit(" ", 1)[0] + "…"


def context_header(norma: str, titulo: str) -> str:
    """Cabecera de contexto que se antepone a cada chunk.

    Un chunk que continúa un artículo largo (p. ej. la mitad de una tabla) no
    contiene el título del artículo, y su embedding pierde el tema ("edad mínima",
    "permiso"...). Anteponer norma + título del artículo lo recupera, y además le
    dice al LLM de dónde sale cada fragmento."""
    return f"[{norma} · {titulo or 'Preámbulo'}]"


# ---------------------------------------------------------------------------
# Carga y chunking
# ---------------------------------------------------------------------------
@dataclass
class LoadedDocument:
    filename: str
    norma: str
    referencia: str
    total_pages: int
    text_pages: int  # páginas con texto útil tras la limpieza
    text: str  # texto completo limpio (todas las páginas útiles concatenadas)
    page_offsets: list[int]  # offset de inicio de cada página dentro de `text`
    page_numbers: list[int]  # número de página (1-based) correspondiente


def load_document(path: Path) -> LoadedDocument:
    """Carga un PDF con PyPDFLoader, limpia cada página y las concatena
    guardando dónde empieza cada página (para poder asignar la página a cada chunk).
    Se concatenan las páginas para que un artículo partido entre dos páginas
    no se corte artificialmente."""
    info = DOCUMENTS[path.name]
    pages = PyPDFLoader(str(path)).load()
    parts, numbers = [], []
    for page in pages:
        raw = page.page_content
        if is_index_page(raw):
            continue
        cleaned = clean_text(raw)
        if len(cleaned) < MIN_PAGE_CHARS:
            continue
        numbers.append(page.metadata.get("page", 0) + 1)
        parts.append(cleaned)
    # Marcadores de página invisibles (\x00N\x00) para poder normalizar tablas que
    # cruzan páginas y, aun así, recalcular después dónde empieza cada página.
    marked = "\n".join(f"\x00{n}\x00{p}" for n, p in zip(numbers, parts))
    if path.name == "BOE-A-2009-9481-consolidado.pdf":
        marked = _normalize_permit_table_marked(marked)
    text_parts, offsets, pos = [], [], 0
    segs = marked.split("\x00")[1:]  # [nº página, texto, nº página, texto, ...]
    page_nums, page_texts = segs[0::2], segs[1::2]
    for p in page_texts:
        p = p.removesuffix("\n")
        offsets.append(pos)
        text_parts.append(p)
        pos += len(p) + 1
    return LoadedDocument(
        filename=path.name,
        norma=info["norma"],
        referencia=info["referencia"],
        total_pages=len(pages),
        text_pages=len(parts),
        text="\n".join(text_parts),
        page_offsets=offsets,
        page_numbers=[int(n) for n in page_nums],
    )


def _normalize_permit_table_marked(marked: str) -> str:
    """Aplica normalize_permit_table respetando los marcadores de página: los
    marcadores se apartan a líneas propias, se normaliza y se vuelven a pegar."""
    tmp = re.sub(r"\x00(\d+)\x00", lambda m: f"\n\x00{m.group(1)}\x00\n", marked)
    tmp = normalize_permit_table(tmp)
    return re.sub(r"\n?\x00(\d+)\x00\n", lambda m: f"\x00{m.group(1)}\x00", tmp)


def load_documents(data_dir: Path = DATA_DIR) -> list[LoadedDocument]:
    docs = []
    for filename in DOCUMENTS:
        path = data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Falta el PDF {path}. Descárgalo del BOE en data/.")
        docs.append(load_document(path))
    return docs


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Separadores de mayor a menor preferencia: primero el inicio de un artículo,
    disposición o anexo; luego títulos/capítulos; después párrafos, frases y palabras."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            r"\n(?=Artículo \d+(?: bis| ter)?\.|Artículo único\.)",
            r"\n(?=Disposición (?:adicional|transitoria|derogatoria|final) )",
            r"\n(?=ANEXO [IVXLC]+)",
            r"\n(?=(?:TÍTULO|CAPÍTULO|Sección) )",
            r"\n(?=Permiso de la clase )",  # filas de la tabla de clases de permiso
            r"\n",
            r"(?<=\.) ",
            r" ",
            "",
        ],
        is_separator_regex=True,
        keep_separator="start",
        add_start_index=True,
    )


def chunk_document(doc: LoadedDocument, splitter=None) -> list[Document]:
    """Trocea un documento y añade metadatos: norma, página y artículo(s)."""
    splitter = splitter or get_text_splitter()
    matches = list(_HEADING_RE.finditer(doc.text))
    headings = [(m.start(), _normalize_heading(m)) for m in matches]
    titles = [_heading_title(doc.text, m) for m in matches]
    heading_pos = [h[0] for h in headings]

    chunks = splitter.create_documents([doc.text])
    stem = doc.filename.removesuffix(".pdf")
    out = []
    for i, ch in enumerate(chunks):
        start = ch.metadata.get("start_index", 0)
        end = start + len(ch.page_content)
        # Página donde empieza el chunk
        p_idx = max(bisect.bisect_right(doc.page_offsets, start) - 1, 0)
        p_end = max(bisect.bisect_right(doc.page_offsets, max(end - 1, start)) - 1, 0)
        # Artículo vigente al inicio del chunk + los que empiezan dentro de él
        h_idx = bisect.bisect_right(heading_pos, start) - 1
        arts = [headings[h_idx][1]] if h_idx >= 0 else []
        j = h_idx + 1
        while j < len(headings) and headings[j][0] < end:
            if headings[j][1] not in arts:
                arts.append(headings[j][1])
            j += 1
        titulo = titles[h_idx] if h_idx >= 0 else ""  # título del artículo vigente al inicio
        content = f"{context_header(doc.norma, titulo)}\n{ch.page_content.strip()}"
        digest = hashlib.sha1(content.encode()).hexdigest()[:10]
        out.append(
            Document(
                page_content=content,
                metadata={
                    "norma": doc.norma,
                    "referencia": doc.referencia,
                    "fuente": doc.filename,
                    "pagina": doc.page_numbers[p_idx],
                    "pagina_fin": doc.page_numbers[p_end],
                    "articulo": arts[0] if arts else "Preámbulo",
                    "articulos": ", ".join(arts) if arts else "Preámbulo",
                    "titulo": titulo or "Preámbulo",
                    # ID determinista que incluye un hash del contenido: si cambia
                    # la limpieza o el chunking, el ID cambia y se reindexa solo ese chunk.
                    "chunk_id": f"{stem}-{i:05d}-{digest}",
                },
            )
        )
    return out


def build_chunks(docs: list[LoadedDocument] | None = None) -> list[Document]:
    docs = docs if docs is not None else load_documents()
    splitter = get_text_splitter()
    return [c for d in docs for c in chunk_document(d, splitter)]


# ---------------------------------------------------------------------------
# Embeddings y vector store
# ---------------------------------------------------------------------------
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Falta GOOGLE_API_KEY (defínela en .env o en los secrets de Streamlit).")
    # task_type: la librería usa RETRIEVAL_DOCUMENT al indexar y RETRIEVAL_QUERY al consultar.
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, output_dimensionality=EMBEDDING_DIM)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def _is_rate_limit(err: Exception) -> bool:
    msg = str(err).upper()
    return any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "QUOTA", "RATE LIMIT", "503", "UNAVAILABLE"))


def _add_with_retries(vs: Chroma, docs: list[Document]) -> None:
    delay = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vs.add_documents(docs, ids=[d.metadata["chunk_id"] for d in docs])
            return
        except Exception as err:  # noqa: BLE001
            if not _is_rate_limit(err) or attempt == MAX_RETRIES:
                raise
            print(f"  Límite de la API (intento {attempt}/{MAX_RETRIES}); espero {delay}s...")
            time.sleep(delay)
            delay *= 2


def build_knowledge_base(force: bool = False, verbose: bool = True) -> Chroma:
    """Crea (o completa) el índice en chroma_db/.

    - Si la colección ya contiene todos los chunks, no hace nada.
    - Si está a medias (p. ej. se agotó la cuota diaria), indexa solo lo que falta:
      los IDs de chunk son deterministas, así que se puede relanzar sin duplicar.
    - force=True borra la colección y reindexa todo.
    """
    vs = get_vectorstore()
    if force:
        vs.delete_collection()
        vs = get_vectorstore()

    chunks = build_chunks()
    current = {c.metadata["chunk_id"] for c in chunks}
    existing = set(vs.get(include=[])["ids"])
    # Chunks de una versión anterior de la limpieza/chunking: se eliminan.
    stale = sorted(existing - current)
    if stale:
        for i in range(0, len(stale), 500):
            vs.delete(ids=stale[i : i + 500])
        existing -= set(stale)
    pending = [c for c in chunks if c.metadata["chunk_id"] not in existing]
    if verbose:
        print(
            f"Chunks totales: {len(chunks)} | ya indexados: {len(existing)} | "
            f"obsoletos eliminados: {len(stale)} | pendientes: {len(pending)}"
        )
    if not pending:
        if verbose:
            print("La colección ya está completa; no se reindexa (usa force=True para rehacerla).")
        return vs

    n_batches = (len(pending) + BATCH_SIZE - 1) // BATCH_SIZE
    for b in range(n_batches):
        batch = pending[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        _add_with_retries(vs, batch)
        if verbose:
            print(f"  Lote {b + 1}/{n_batches} indexado ({len(batch)} chunks)")
        if b < n_batches - 1:
            time.sleep(PAUSE_BETWEEN_BATCHES)
    if verbose:
        print(f"Indexación completa: {vs._collection.count()} chunks en '{COLLECTION_NAME}'.")
    return vs


# ---------------------------------------------------------------------------
# Retrieval híbrido: BM25 (palabras clave) + vectores (semántica) con RRF
# ---------------------------------------------------------------------------
_STOPWORDS = set(
    "de la el en y a los las del se que con por para un una su al lo o es como más sus le "
    "ya sin sobre este entre cuando también hasta desde nos durante ni contra ese eso ante "
    "bajo cual cuál qué puedo son ser será".split()
)


def bm25_tokenize(text: str) -> list[str]:
    """Minúsculas, sin tildes, sin stopwords. Conserva códigos como 'a2', 'b', '125'."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [w for w in re.findall(r"\w+", text) if w not in _STOPWORDS]


class HybridRetriever(BaseRetriever):
    """Fusiona BM25 y búsqueda vectorial con Reciprocal Rank Fusion (RRF).

    Los embeddings captan bien el significado ("casco obligatorio" ~ "elementos
    de protección"), pero no distinguen bien códigos cortos como "A2", "B" o
    "125 cm3", que es justo lo que BM25 hace mejor. RRF suma 1/(rrf_k + rango)
    de cada lista, así que un chunk que sale arriba en las dos gana."""

    vectorstore: Chroma
    bm25: BM25Retriever
    k: int = 5
    fetch_k: int = 20
    rrf_k: int = 60
    weights: tuple[float, float] = (0.5, 0.5)  # (vectorial, BM25)

    def _vector_search(self, query: str, attempts: int = 3) -> list[Document]:
        """Búsqueda vectorial con reintentos. Si la API de embeddings sigue fallando
        (500/503 puntuales), devuelve [] y el retriever sigue funcionando solo con BM25."""
        for attempt in range(attempts):
            try:
                return self.vectorstore.similarity_search(query, k=self.fetch_k)
            except Exception as err:  # noqa: BLE001
                if attempt == attempts - 1:
                    warnings.warn(f"Búsqueda vectorial no disponible ({err}); se usa solo BM25.", stacklevel=2)
                    return []
                time.sleep(2 ** (attempt + 1))
        return []

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        vec = self._vector_search(query)
        self.bm25.k = self.fetch_k
        kw = self.bm25.invoke(query)
        scores: dict[str, float] = {}
        docs: dict[str, Document] = {}
        for weight, results in zip(self.weights, (vec, kw)):
            for rank, d in enumerate(results, start=1):
                cid = d.metadata["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + weight / (self.rrf_k + rank)
                docs.setdefault(cid, d)
        best = sorted(scores, key=scores.get, reverse=True)[: self.k]
        return [docs[cid] for cid in best]


def get_bm25_retriever(vectorstore: Chroma | None = None) -> BM25Retriever:
    """BM25 construido con los chunks guardados en Chroma (no necesita los PDF)."""
    vs = vectorstore or get_vectorstore()
    data = vs.get(include=["documents", "metadatas"])
    if not data["ids"]:
        raise RuntimeError("La colección está vacía: ejecuta build_knowledge_base() primero.")
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    return BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenize)


def get_retriever(k: int = 5, mode: str = "hybrid"):
    """Retriever sobre la colección persistente.

    mode="hybrid" (por defecto): BM25 + vectores con RRF.
    mode="vector": solo similitud vectorial.
    mode="bm25": solo palabras clave.
    """
    vs = get_vectorstore()
    if mode == "vector":
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
    bm25 = get_bm25_retriever(vs)
    if mode == "bm25":
        bm25.k = k
        return bm25
    return HybridRetriever(vectorstore=vs, bm25=bm25, k=k)
