import os

import faiss
import numpy as np
import streamlit as st
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="METALYS AI",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GEMINI_MODEL = "gemini-2.0-flash"

# ======================================================================
# STYLE (apparence industrielle : fond clair, titres condensés noirs,
# accents orange, bandes sombres, étiquettes monospace)
# ======================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800;900&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');
:root{--bg:#f7f7f4;--ink:#0d0f10;--orange:#ff4d1f;--dark:#15191b;--line:#d8dad6;--muted:#646b6d;}
.stApp{background:var(--bg);color:var(--ink);font-family:'DM Sans',sans-serif;}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stSidebar"],[data-testid="collapsedControl"],[data-testid="stToolbar"]{display:none !important;}
.block-container{max-width:1320px;padding:0 2.2rem 3rem 2.2rem !important;}
[data-testid="stBottom"] > div{background:var(--bg);}
/* ---------- navigation ---------- */
.nav{display:flex;align-items:center;justify-content:space-between;padding:22px 0;border-bottom:1px solid var(--line);}
.brand{display:flex;align-items:center;gap:12px;}
.mark{width:30px;height:30px;background:var(--orange);color:#0d0f10;font:900 22px 'Barlow Condensed',sans-serif;display:grid;place-items:center;clip-path:polygon(0 0,100% 0,100% 72%,72% 100%,0 100%);}
.brand b{font:800 18px 'Barlow Condensed',sans-serif;letter-spacing:2px;}
.brand small{display:block;font:500 8px 'DM Mono',monospace;letter-spacing:2px;color:var(--muted);}
.links{display:flex;gap:34px;font:500 10px 'DM Mono',monospace;letter-spacing:1.4px;color:#3b4143;}
.cta{background:var(--ink);color:#fff;padding:12px 18px;font:500 10px 'DM Mono',monospace;letter-spacing:1.2px;}
/* ---------- hero ---------- */
.hero{display:flex;gap:30px;align-items:stretch;min-height:470px;padding-top:34px;}
.hero-copy{flex:1.05;display:flex;flex-direction:column;justify-content:center;}
.eyebrow{font:500 10px 'DM Mono',monospace;letter-spacing:1.6px;color:var(--muted);display:flex;align-items:center;gap:12px;}
.eyebrow:before{content:"+";font-size:16px;}
.hero h1{font:900 clamp(58px,7.4vw,112px)/.86 'Barlow Condensed',sans-serif;text-transform:uppercase;margin:26px 0 22px;letter-spacing:.5px;}
.hero h1 em{color:var(--orange);font-style:normal;}
.hero p{max-width:430px;font-size:15px;line-height:1.65;color:#4a5153;margin:0 0 26px;}
.pill-row{display:flex;gap:0;border-top:1px solid var(--line);padding-top:16px;font:500 10px 'DM Mono',monospace;letter-spacing:1.2px;max-width:470px;}
.pill-row span{padding:0 18px;border-left:1px solid var(--line);}
.pill-row span:first-child{padding-left:0;border:0;}
.hero-art{flex:1;position:relative;background:radial-gradient(ellipse at 50% 55%,#cfd3d0 0%,#e7e8e4 48%,#f1f1ee 100%);overflow:hidden;border:1px solid var(--line);min-height:420px;}
.orbit{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);border:1px solid #9ea6a2;border-radius:50%;}
.o1{width:360px;height:210px;} .o2{width:270px;height:270px;}
.disc{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:210px;height:210px;border-radius:50%;background:repeating-conic-gradient(#4f5759 0deg 7deg,#b4bbb8 7deg 12deg,#7a8283 12deg 16deg);border:10px solid #747c7d;box-shadow:inset 0 0 0 11px #c9cdc9,0 22px 40px #4b525155;}
.disc:after{content:"";position:absolute;inset:56px;border-radius:50%;background:radial-gradient(circle,#40474a 0 24%,#b0b7b4 26% 43%,#51595b 45% 60%,#c7ccc8 62%);border:5px solid #5f6768;}
.tag{position:absolute;font:500 9px 'DM Mono',monospace;letter-spacing:1.2px;background:#fffffff0;border:1px solid var(--line);padding:7px 10px;}
.tag:before{content:"";position:absolute;width:34px;height:1px;background:var(--orange);top:50%;}
.t1{top:70px;left:36px;} .t1:before{left:100%;}
.t2{top:120px;right:30px;} .t2:before{right:100%;}
.t3{bottom:70px;left:50px;} .t3:before{left:100%;}
.t4{bottom:46px;right:40px;} .t4:before{right:100%;}
.cross{position:absolute;right:30%;top:30%;color:var(--orange);font-size:26px;}
/* ---------- bande sombre ---------- */
.band{background:var(--dark);color:#d9ddda;display:flex;justify-content:space-around;padding:17px 10px;margin-top:0;font:500 10px 'DM Mono',monospace;letter-spacing:2px;}
/* ---------- sections ---------- */
.sec{padding-top:56px;}
.sec h2{font:900 clamp(40px,4.6vw,66px)/.9 'Barlow Condensed',sans-serif;text-transform:uppercase;margin:16px 0 8px;}
.sec h2 em{color:var(--orange);font-style:normal;}
.sec .sub{color:#4a5153;font-size:14px;max-width:420px;line-height:1.6;}
.step-h{display:flex;align-items:center;gap:12px;margin:34px 0 12px;}
.step-h span{font:500 22px 'DM Mono',monospace;color:var(--orange);}
.step-h i{flex:1;height:1px;background:var(--orange);}
.step-t{font:800 22px 'Barlow Condensed',sans-serif;text-transform:uppercase;letter-spacing:.8px;}
.step-p{font-size:12px;color:var(--muted);margin:2px 0 12px;}
/* ---------- widgets Streamlit ---------- */
.stButton > button{background:var(--orange) !important;color:#111 !important;border:0 !important;border-radius:0 !important;padding:.85rem 1.2rem !important;font:500 11px 'DM Mono',monospace !important;letter-spacing:1.3px !important;text-transform:uppercase;width:100%;}
.stButton > button:hover{background:#ff6c45 !important;color:#000 !important;}
.stTextInput input{border-radius:0 !important;border:1px solid #20272b !important;background:#fff !important;font-family:'DM Mono',monospace;}
.stTextInput label p,.stFileUploader label p{font:500 10px 'DM Mono',monospace !important;letter-spacing:1.4px;text-transform:uppercase;color:var(--muted);}
[data-testid="stFileUploaderDropzone"]{border:1px dashed #5a6466 !important;border-radius:0 !important;background:#fff !important;}
[data-testid="stFileUploaderDropzone"] button{border-radius:0 !important;}
[data-testid="stAlert"]{border-radius:0 !important;}
/* ---------- statistiques ---------- */
.stats{display:grid;grid-template-columns:repeat(4,1fr);border-top:1px solid var(--line);border-bottom:1px solid var(--line);margin-top:42px;}
.stat{padding:22px 10px;text-align:center;border-left:1px solid var(--line);}
.stat:first-child{border:0;}
.stat b{display:block;font:800 34px 'Barlow Condensed',sans-serif;letter-spacing:2px;}
.stat small{font:500 9px 'DM Mono',monospace;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;}
/* ---------- section sombre ---------- */
.dark{background:var(--dark);color:#fff;margin:60px -2.2rem 0 -2.2rem;padding:54px 2.2rem;}
.dark h2{font:900 clamp(40px,5vw,72px)/.9 'Barlow Condensed',sans-serif;text-transform:uppercase;margin:0 0 14px;}
.dark h2 em{color:var(--orange);font-style:normal;}
.dark p{max-width:420px;color:#c3c9c6;line-height:1.6;font-size:14px;}
.route{display:flex;gap:0;margin-top:34px;border-top:1px solid #384044;}
.route div{flex:1;padding:18px 16px 0 0;font:500 10px 'DM Mono',monospace;letter-spacing:1.6px;color:#c9cfcc;}
.route div b{display:block;font:700 28px 'Barlow Condensed',sans-serif;color:var(--orange);margin-bottom:6px;}
/* ---------- chat ---------- */
[data-testid="stChatMessage"]{background:#fff;border:1px solid var(--line);border-radius:0;padding:14px 18px;}
[data-testid="stChatInput"]{border-radius:0 !important;}
[data-testid="stChatInput"] textarea{font-family:'DM Sans',sans-serif;}
/* ---------- pied de page ---------- */
.foot{background:var(--dark);color:#aeb5b2;margin:60px -2.2rem -3rem -2.2rem;padding:26px 2.2rem;display:flex;justify-content:space-between;align-items:center;font:500 9px 'DM Mono',monospace;letter-spacing:1.4px;}
.foot .brand b{color:#fff;font-size:16px;}
@media (max-width:900px){.hero{flex-direction:column}.links{display:none}.stats{grid-template-columns:repeat(2,1fr)}.route{flex-direction:column}}
</style>
"""


def html(s: str):
    """Affiche du HTML (sans lignes vides ni indentation, pour que le Markdown ne casse rien)."""
    st.markdown("\n".join(l.strip() for l in s.splitlines() if l.strip()), unsafe_allow_html=True)


# ======================================================================
# LOGIQUE (RAG : PDF -> extraits -> embeddings -> FAISS -> Gemini)
# ======================================================================
@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


def read_pdfs(files):
    texts = []
    for f in files:
        reader = PdfReader(f)
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
    return "\n".join(texts)


def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def build_index(chunks, embedder):
    emb = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    emb = np.asarray(emb, dtype="float32")
    faiss.normalize_L2(emb)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def retrieve(question, index, chunks, embedder, k=4):
    q = np.asarray(embedder.encode([question], convert_to_numpy=True), dtype="float32")
    faiss.normalize_L2(q)
    _, ids = index.search(q, min(k, len(chunks)))
    return [chunks[i] for i in ids[0] if i >= 0]


def ask_gemini(client, question, context_chunks):
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "Tu es l'assistant du projet METALYS AI (inspection visuelle de défauts "
        "sur acier laminé à chaud). Réponds en français, uniquement à partir du "
        "contexte ci-dessous. Si la réponse n'y est pas, dis-le clairement.\n\n"
        f"CONTEXTE:\n{context}\n\nQUESTION: {question}"
    )
    return client.models.generate_content(model=GEMINI_MODEL, contents=prompt).text


# ======================================================================
# ÉTAT
# ======================================================================
for key, default in {"index": None, "chunks": [], "history": [], "n_docs": 0}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ======================================================================
# INTERFACE
# ======================================================================
st.markdown(CSS, unsafe_allow_html=True)

html("""
<div class="nav">
<div class="brand"><div class="mark">M</div><div><b>METALYS AI</b><small>INSPECTION & DOCUMENTS</small></div></div>
<div class="links"><span>DOCUMENTS</span><span>INDEXATION</span><span>ASSISTANT</span><span>À PROPOS</span></div>
<div class="cta">OUVRIR L'ASSISTANT ↗</div>
</div>
<div class="hero">
<div class="hero-copy">
<div class="eyebrow">ASSISTANT DOCUMENTAIRE / ACIER LAMINÉ À CHAUD</div>
<h1>Vos documents.<br>Des réponses<br><em>précises.</em></h1>
<p>Importez vos rapports techniques en PDF, indexez-les, puis interrogez-les en langage naturel. Chaque réponse s'appuie uniquement sur vos documents.</p>
<div class="pill-row"><span>6 CLASSES NEU-DET</span><span>PDF → RAG</span><span>100% LOCAL INDEX</span></div>
</div>
<div class="hero-art">
<div class="orbit o1"></div><div class="orbit o2"></div><div class="disc"></div><div class="cross">+</div>
<div class="tag t1">CRAZING</div><div class="tag t2">INCLUSION</div>
<div class="tag t3">SCRATCHES</div><div class="tag t4">PITTED SURFACE</div>
</div>
</div>
<div class="band"><span>PATCHES</span><span>CRAZING</span><span>INCLUSION</span><span>ROLLED-IN SCALE</span><span>SCRATCHES</span></div>
<div class="sec">
<div class="eyebrow">UN FLUX / TROIS ÉTAPES</div>
<h2>Prêt en<br><em>trois étapes.</em></h2>
<div class="sub">Configuration, import et indexation fonctionnent comme un seul système connecté.</div>
</div>
""")

c1, c2, c3 = st.columns(3, gap="large")

with c1:
    html("""<div class="step-h"><span>01</span><i></i></div>
    <div class="step-t">Clé API</div><div class="step-p">Clé Gemini nécessaire pour générer les réponses.</div>""")
    api_key = st.text_input("Clé API Gemini", type="password", value=os.getenv("GEMINI_API_KEY", ""))

with c2:
    html("""<div class="step-h"><span>02</span><i></i></div>
    <div class="step-t">Documents</div><div class="step-p">Un ou plusieurs rapports PDF à interroger.</div>""")
    files = st.file_uploader("Importer des PDF", type=["pdf"], accept_multiple_files=True)

with c3:
    html("""<div class="step-h"><span>03</span><i></i></div>
    <div class="step-t">Indexation</div><div class="step-p">Découpe et vectorise le texte pour la recherche.</div>""")
    if st.button("Indexer les documents"):
        if not files:
            st.error("Ajoute au moins un PDF.")
        else:
            text = read_pdfs(files)
            if not text.strip():
                st.error("Aucun texte extractible (PDF scanné ?).")
            else:
                chunks = chunk_text(text)
                st.session_state.chunks = chunks
                st.session_state.index = build_index(chunks, load_embedder())
                st.session_state.n_docs = len(files)
                st.success(f"{len(chunks)} extraits indexés.")

n_q = sum(1 for r, _ in st.session_state.history if r == "user")
ready = "PRÊT" if st.session_state.index is not None else "EN ATTENTE"
html(f"""
<div class="stats">
<div class="stat"><b>{st.session_state.n_docs}</b><small>Documents</small></div>
<div class="stat"><b>{len(st.session_state.chunks)}</b><small>Extraits indexés</small></div>
<div class="stat"><b>{n_q}</b><small>Questions posées</small></div>
<div class="stat"><b>{ready}</b><small>Statut</small></div>
</div>
<div class="dark">
<div class="eyebrow" style="color:#9aa3a0">DU DOCUMENT À LA RÉPONSE</div>
<h2 style="margin-top:18px">Un seul flux.<br><em>Zéro supposition.</em></h2>
<p>Les réponses sont générées à partir des passages retrouvés dans vos PDF. Si l'information n'y est pas, l'assistant le dit.</p>
<div class="route">
<div><b>01</b>LIRE</div><div><b>02</b>DÉCOUPER</div><div><b>03</b>INDEXER</div><div><b>04</b>RÉPONDRE</div>
</div>
</div>
<div class="sec">
<div class="eyebrow">ASSISTANT</div>
<h2>Posez votre <em>question.</em></h2>
</div>
""")

for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

question = st.chat_input("Pose ta question sur les documents...")

if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        if not api_key:
            answer = "Entre ta clé API Gemini (étape 01)."
        elif st.session_state.index is None:
            answer = "Importe des PDF puis clique sur « Indexer les documents » (étapes 02 et 03)."
        else:
            try:
                ctx = retrieve(question, st.session_state.index, st.session_state.chunks, load_embedder())
                answer = ask_gemini(genai.Client(api_key=api_key), question, ctx)
            except Exception as exc:
                answer = f"Erreur : {exc}"
        st.markdown(answer)
    st.session_state.history.append(("assistant", answer))

html("""
<div class="foot">
<div class="brand"><div class="mark">M</div><b>METALYS AI</b></div>
<span>DOCUMENTS · INDEXATION · ASSISTANT</span>
<span>© 2026 METALYS AI</span>
</div>
""")
