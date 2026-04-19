"""
MovieMind AI — Movie Intelligence Extractor
Premium Streamlit App | Production-Level | 2026 Design
"""

import streamlit as st
import json
import time
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

load_dotenv()

# ─────────────────────────────────────────────
#  PYDANTIC SCHEMA
# ─────────────────────────────────────────────

class MovieInfo(BaseModel):
    title: str = Field(description="Movie title", default="NULL")
    release_year: str = Field(description="Release year", default="NULL")
    genre: list[str] = Field(description="List of genres", default=[])
    director: str = Field(description="Director name", default="NULL")
    main_cast: list[str] = Field(description="Main cast members", default=[])
    setting_location: str = Field(description="Setting or location of the movie", default="NULL")
    plot: str = Field(description="Brief plot description", default="NULL")
    themes: list[str] = Field(description="Key themes", default=[])
    ratings: str = Field(description="Ratings if mentioned", default="NULL")
    notable_features: str = Field(description="Notable features of the movie", default="NULL")
    short_summary: str = Field(description="2-3 line summary", default="NULL")


# ─────────────────────────────────────────────
#  BACKEND LOGIC
# ─────────────────────────────────────────────

@st.cache_resource
def get_model():
    return ChatMistralAI(model='mistral-small-2506')


def extract_text_mode(paragraph: str) -> str:
    """Prompt-based extractor returning clean readable text."""
    model = get_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """
You are a professional Movie Information Extraction Assistant.

Your task:
Extract useful structured information from a movie paragraph and present it in a clean readable format.

Rules:
- Do NOT add explanations
- Do NOT add extra commentary
- Follow the exact format
- If information is missing → write NULL
- Keep summary short (2–3 lines max)
- Do NOT guess unknown facts

Output Format:

Movie Title:
Release Year:
Genre:
Director:
Main Cast:
Setting/Location:
Plot:
Themes:
Ratings:
Notable Features:

Short Summary:
"""),
        ("human", "Extract information from this paragraph:\n{paragraph}")
    ])
    chain = prompt | model
    result = chain.invoke({"paragraph": paragraph})
    return result.content


def extract_structured_mode(paragraph: str) -> MovieInfo:
    """Pydantic-based structured extractor."""
    model = get_model()
    parser = PydanticOutputParser(pydantic_object=MovieInfo)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a movie information extraction AI.
Extract all available movie details from the paragraph.
Return ONLY valid JSON matching this schema — no explanations, no markdown fences.

{format_instructions}

If a field is unknown, use "NULL" for strings or [] for lists."""),
        ("human", "Extract movie info from:\n{paragraph}")
    ]).partial(format_instructions=parser.get_format_instructions())
    chain = prompt | model | parser
    return chain.invoke({"paragraph": paragraph})


# ─────────────────────────────────────────────
#  SAMPLE INPUT
# ─────────────────────────────────────────────

SAMPLE_INPUT = """Inception (2010) is a mind-bending science fiction thriller directed by Christopher Nolan.
The film stars Leonardo DiCaprio as Dom Cobb, a skilled thief who specializes in the art of extraction —
stealing valuable secrets from deep within the subconscious mind during the dream state.
Set across multiple dream levels including Paris, a snowy mountain fortress, and a zero-gravity hotel,
the story follows Cobb's team as they attempt the reverse of extraction — planting an idea instead.
The ensemble cast includes Joseph Gordon-Levitt, Elliot Page, Tom Hardy, and Ken Watanabe.
The movie earned an IMDb rating of 8.8/10 and won four Academy Awards for cinematography, visual effects,
sound editing, and sound mixing. Notable for its revolutionary practical effects and Hans Zimmer's iconic score."""


# ─────────────────────────────────────────────
#  CSS INJECTION
# ─────────────────────────────────────────────

def inject_css():
    st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Base Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #020408 !important;
    font-family: 'DM Sans', sans-serif;
    color: #e2e8f0;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(99,40,200,0.35) 0%, transparent 70%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(0,200,200,0.15) 0%, transparent 60%),
        radial-gradient(ellipse 40% 40% at 10% 90%, rgba(180,0,255,0.12) 0%, transparent 60%),
        #020408 !important;
    animation: bgPulse 12s ease-in-out infinite alternate;
}

@keyframes bgPulse {
    0%   { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(15deg); }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 10, 25, 0.85) !important;
    border-right: 1px solid rgba(139, 92, 246, 0.25) !important;
    backdrop-filter: blur(24px) !important;
}

[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    color: #a78bfa !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Hero Header ── */
.hero-wrap {
    text-align: center;
    padding: 3.5rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(139,92,246,0.15);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 999px;
    padding: 0.28rem 1rem;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 1.4rem;
    animation: fadeSlideDown 0.8s ease both;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, #e879f9 0%, #a78bfa 40%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: fadeSlideDown 0.9s ease 0.1s both;
    filter: drop-shadow(0 0 40px rgba(167,139,250,0.45));
}

.hero-sub {
    margin-top: 0.8rem;
    font-size: 1.05rem;
    font-weight: 300;
    color: #94a3b8;
    letter-spacing: 0.02em;
    animation: fadeSlideDown 1s ease 0.2s both;
}

.hero-divider {
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #a78bfa, #22d3ee, transparent);
    margin: 1.8rem auto 0;
    border-radius: 999px;
    animation: fadeSlideDown 1.1s ease 0.3s both;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-18px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Glass Panel ── */
.glass {
    background: rgba(15, 15, 35, 0.6);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 20px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    padding: 1.8rem;
    margin: 1rem 0;
    animation: fadeIn 0.7s ease both;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Textarea override ── */
[data-testid="stTextArea"] textarea {
    background: rgba(10,10,30,0.7) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 14px !important;
    color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.97rem !important;
    transition: border-color 0.3s, box-shadow 0.3s;
    resize: vertical !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(167,139,250,0.7) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15), 0 0 30px rgba(167,139,250,0.2) !important;
    outline: none !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 50%, #0e7490 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.4) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    filter: brightness(1.15) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* Secondary buttons */
[data-testid="stButton"] > button[kind="secondary"] {
    background: rgba(30,30,60,0.6) !important;
    border: 1px solid rgba(139,92,246,0.35) !important;
    color: #a78bfa !important;
    box-shadow: none !important;
}
[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: rgba(139,92,246,0.15) !important;
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.25) !important;
}

/* ── Mode pills (radio) ── */
[data-testid="stRadio"] > div {
    flex-direction: row !important;
    gap: 0.6rem !important;
}
[data-testid="stRadio"] label {
    background: rgba(20,20,50,0.6) !important;
    border: 1px solid rgba(139,92,246,0.25) !important;
    border-radius: 999px !important;
    padding: 0.5rem 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.25s !important;
    color: #94a3b8 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(135deg, rgba(124,58,237,0.4), rgba(14,116,144,0.4)) !important;
    border-color: rgba(167,139,250,0.6) !important;
    color: #e2e8f0 !important;
    box-shadow: 0 0 20px rgba(167,139,250,0.2) !important;
}

/* ── Output text block ── */
.output-text-block {
    background: rgba(8,8,24,0.75);
    border: 1px solid rgba(34,211,238,0.2);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.96rem;
    line-height: 1.85;
    color: #cbd5e1;
    white-space: pre-wrap;
    box-shadow: 0 0 40px rgba(34,211,238,0.06), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
}

.output-text-block .field-label {
    color: #a78bfa;
    font-weight: 600;
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Movie cards ── */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}

.movie-card {
    background: rgba(12,12,30,0.7);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(16px);
    transition: transform 0.25s, box-shadow 0.25s, border-color 0.25s;
    animation: fadeIn 0.6s ease both;
}
.movie-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(124,58,237,0.3);
    border-color: rgba(167,139,250,0.45);
}

.card-label {
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #64748b;
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.5rem;
}
.card-value {
    font-size: 1.05rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1.4;
}
.card-value.big {
    font-family: 'Syne', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    background: linear-gradient(135deg, #e879f9, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Tags & chips ── */
.tags-wrap { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.3rem; }
.tag {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    color: #c4b5fd;
    letter-spacing: 0.03em;
}
.chip {
    display: inline-block;
    background: rgba(14,116,144,0.2);
    border: 1px solid rgba(34,211,238,0.3);
    border-radius: 999px;
    padding: 0.22rem 0.75rem;
    font-size: 0.78rem;
    color: #67e8f9;
}

/* ── Rating badge ── */
.rating-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: linear-gradient(135deg, rgba(234,179,8,0.15), rgba(234,179,8,0.05));
    border: 1px solid rgba(234,179,8,0.35);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #fde68a;
}

/* ── Summary card ── */
.summary-card {
    background: linear-gradient(135deg, rgba(124,58,237,0.12), rgba(14,116,144,0.1));
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 18px;
    padding: 1.8rem 2rem;
    margin-top: 1.2rem;
    position: relative;
    overflow: hidden;
}
.summary-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #e879f9, #a78bfa, #22d3ee);
}
.summary-card .s-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.8rem;
}
.summary-card .s-text {
    font-size: 1rem;
    font-weight: 300;
    color: #e2e8f0;
    line-height: 1.8;
    font-style: italic;
}

/* ── JSON block ── */
.json-block {
    background: rgba(5,5,15,0.9);
    border: 1px solid rgba(34,211,238,0.15);
    border-radius: 14px;
    padding: 1.4rem;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    color: #67e8f9;
    overflow-x: auto;
    white-space: pre;
    line-height: 1.7;
}

/* ── Welcome empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #475569;
}
.empty-state .es-icon {
    font-size: 3.5rem;
    margin-bottom: 1rem;
    opacity: 0.5;
    animation: floatIcon 4s ease-in-out infinite;
}
@keyframes floatIcon {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.9rem; color: #475569; line-height: 1.6; }

/* ── Error card ── */
.error-card {
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.3);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    color: #fca5a5;
    font-size: 0.92rem;
}
.error-card strong { color: #f87171; font-family: 'Syne', sans-serif; }

/* ── Spinner override ── */
[data-testid="stSpinner"] {
    color: #a78bfa !important;
}

/* ── Selectbox / number ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(10,10,30,0.7) !important;
    border: 1px solid rgba(139,92,246,0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.7rem;
}

/* ── Copy notice ── */
.copy-notice {
    font-size: 0.75rem;
    color: #22d3ee;
    margin-top: 0.5rem;
    text-align: right;
    opacity: 0.7;
}

/* ── Divider ── */
hr {
    border: none;
    border-top: 1px solid rgba(139,92,246,0.15);
    margin: 1.5rem 0;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(139,92,246,0.4); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(167,139,250,0.6); }

/* ── Stagger delay for cards ── */
.movie-card:nth-child(1) { animation-delay: 0.05s; }
.movie-card:nth-child(2) { animation-delay: 0.12s; }
.movie-card:nth-child(3) { animation-delay: 0.19s; }
.movie-card:nth-child(4) { animation-delay: 0.26s; }
.movie-card:nth-child(5) { animation-delay: 0.33s; }
.movie-card:nth-child(6) { animation-delay: 0.40s; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  UI COMPONENTS
# ─────────────────────────────────────────────

def render_hero():
    st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">✦ AI · Movie Intelligence · 2026</div>
    <div class="hero-title">🎬 MovieMind AI</div>
    <div class="hero-sub">Extract deep movie intelligence from any paragraph — instantly.</div>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


def render_empty_state():
    st.markdown("""
<div class="empty-state">
    <div class="es-icon">🎥</div>
    <h3>Ready to extract movie intelligence</h3>
    <p>Paste any movie description, synopsis, or review above.<br>
    CineMind will decode the details in seconds.</p>
</div>
""", unsafe_allow_html=True)


def render_error(msg: str):
    st.markdown(f"""
<div class="error-card">
    <strong>⚠ Extraction Failed</strong><br>
    {msg}
</div>
""", unsafe_allow_html=True)


def render_tags(items: list[str], style: str = "tag") -> str:
    if not items:
        return "<span style='color:#475569'>NULL</span>"
    return "".join(f'<span class="{style}">{i}</span>' for i in items)


def render_text_output(text: str):
    """Render the prompt-based text output with syntax highlighting."""
    lines = text.strip().split("\n")
    html_lines = []
    for line in lines:
        if ":" in line and not line.startswith(" ") and not line.startswith("-"):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[0].strip():
                label = parts[0].strip()
                value = parts[1].strip()
                html_lines.append(
                    f'<span class="field-label">{label}:</span> '
                    f'<span style="color:#e2e8f0">{value if value else "<span style=\'color:#475569\'>—</span>"}</span>'
                )
                continue
        html_lines.append(f'<span style="color:#94a3b8">{line}</span>' if line.strip() == "" else f'<span>{line}</span>')

    body = "<br>".join(html_lines)
    st.markdown(f'<div class="output-text-block">{body}</div>', unsafe_allow_html=True)

    # Copy hint
    st.markdown('<div class="copy-notice">⌘C to copy raw text from the code block below</div>', unsafe_allow_html=True)
    with st.expander("📋 Raw Text (copy-friendly)"):
        st.code(text, language="text")


def render_structured_output(info: MovieInfo, show_json: bool):
    """Render Pydantic result as premium cards UI."""

    # ── Row 1: Title + Year ──
    st.markdown('<div class="card-grid">', unsafe_allow_html=True)

    title_html = f"""
<div class="movie-card">
    <div class="card-label">🎬 Movie Title</div>
    <div class="card-value big">{info.title}</div>
</div>"""

    year_html = f"""
<div class="movie-card">
    <div class="card-label">📅 Release Year</div>
    <div class="card-value big" style="background: linear-gradient(135deg,#22d3ee,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text">{info.release_year}</div>
</div>"""

    director_html = f"""
<div class="movie-card">
    <div class="card-label">🎥 Director</div>
    <div class="card-value">{info.director}</div>
</div>"""

    location_html = f"""
<div class="movie-card">
    <div class="card-label">📍 Setting / Location</div>
    <div class="card-value">{info.setting_location}</div>
</div>"""

    st.markdown(title_html + year_html + director_html + location_html + '</div>', unsafe_allow_html=True)

    # ── Row 2: Genre, Cast, Rating ──
    genre_tags = render_tags(info.genre, "tag")
    cast_chips = render_tags(info.main_cast, "chip")
    theme_tags = render_tags(info.themes, "tag")

    rating_html = f"""
<div class="movie-card">
    <div class="card-label">⭐ Rating</div>
    <div style="margin-top:0.4rem">
        <span class="rating-badge">⭐ {info.ratings}</span>
    </div>
</div>"""

    genre_html = f"""
<div class="movie-card">
    <div class="card-label">🎭 Genre</div>
    <div class="tags-wrap">{genre_tags}</div>
</div>"""

    cast_html = f"""
<div class="movie-card" style="grid-column: span 2">
    <div class="card-label">🎭 Main Cast</div>
    <div class="tags-wrap">{cast_chips}</div>
</div>"""

    themes_html = f"""
<div class="movie-card">
    <div class="card-label">💡 Themes</div>
    <div class="tags-wrap">{theme_tags}</div>
</div>"""

    st.markdown(f'<div class="card-grid">{genre_html}{rating_html}{cast_html}{themes_html}</div>', unsafe_allow_html=True)

    # ── Plot card (full width) ──
    st.markdown(f"""
<div class="movie-card" style="margin: 0.5rem 0">
    <div class="card-label">📖 Plot</div>
    <div class="card-value" style="font-weight:300; line-height:1.75; color:#cbd5e1">{info.plot}</div>
</div>""", unsafe_allow_html=True)

    # ── Notable Features ──
    st.markdown(f"""
<div class="movie-card" style="margin: 0.5rem 0">
    <div class="card-label">✨ Notable Features</div>
    <div class="card-value" style="font-weight:300; line-height:1.75; color:#cbd5e1">{info.notable_features}</div>
</div>""", unsafe_allow_html=True)

    # ── Summary card ──
    st.markdown(f"""
<div class="summary-card">
    <div class="s-label">✦ Short Summary</div>
    <div class="s-text">{info.short_summary}</div>
</div>""", unsafe_allow_html=True)

    # ── JSON toggle ──
    if show_json:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">📦 Raw JSON Output</div>', unsafe_allow_html=True)
        raw = json.dumps(info.model_dump(), indent=2)
        st.markdown(f'<div class="json-block">{raw}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
<div style="text-align:center;padding:1.2rem 0 0.5rem">
    <div style="font-size:2.5rem">🎬</div>
    <div style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;
         background:linear-gradient(135deg,#e879f9,#a78bfa);
         -webkit-background-clip:text;-webkit-text-fill-color:transparent;
         background-clip:text;margin-top:0.3rem">MovieMind AI</div>
    <div style="font-size:0.72rem;letter-spacing:0.12em;text-transform:uppercase;
         color:#475569;margin-top:0.2rem">Movie Intelligence Extractor</div>
</div>
<hr style="border-color:rgba(139,92,246,0.2);margin:1rem 0">
""", unsafe_allow_html=True)

        st.markdown('<div class="section-label">⚙ Extraction Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "mode",
            ["🧾 Text Mode", "🧠 Structured Mode"],
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        show_json = False
        if "Structured" in mode:
            show_json = st.toggle("📦 Show JSON Output", value=False)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
<div style="font-size:0.82rem;color:#475569;line-height:1.7">
<strong style="color:#64748b">🧾 Text Mode</strong><br>
Uses a structured prompt to return clean, readable extraction.<br><br>
<strong style="color:#64748b">🧠 Structured Mode</strong><br>
Uses Pydantic schema + LangChain parser for JSON-structured results with card UI.
</div>
""", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.pop("result", None)
            st.session_state.pop("mode_used", None)
            st.session_state["input_text"] = ""
            st.rerun()

    return mode, show_json


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="MovieMind AI",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_css()

    # Session init
    if "result" not in st.session_state:
        st.session_state.result = None
    if "mode_used" not in st.session_state:
        st.session_state.mode_used = None
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

    mode, show_json = render_sidebar()
    render_hero()

    # ── Input section ──
    col_inp, col_pad = st.columns([3, 1])
    with col_inp:
        st.markdown('<div class="section-label">✍ Movie Paragraph Input</div>', unsafe_allow_html=True)
        user_input = st.text_area(
            "paragraph",
            value=st.session_state.input_text,
            placeholder="Paste any movie description, synopsis, review, or trivia...",
            height=180,
            label_visibility="collapsed",
            key="textarea_input"
        )

        col_btn1, col_btn2, col_sp = st.columns([1.5, 1.5, 3])
        with col_btn1:
            extract_clicked = st.button("🚀 Extract Intelligence", use_container_width=True)
        with col_btn2:
            sample_clicked = st.button("💡 Load Sample", use_container_width=True)

    if sample_clicked:
        st.session_state.input_text = SAMPLE_INPUT
        st.rerun()

    # ── Extraction ──
    if extract_clicked:
        paragraph = user_input.strip()
        if not paragraph:
            render_error("Please enter a movie paragraph before extracting.")
        else:
            st.session_state.result = None
            is_structured = "Structured" in mode

            with st.spinner("🎬 MovieMind is analyzing the movie..."):
                try:
                    if is_structured:
                        result = extract_structured_mode(paragraph)
                    else:
                        result = extract_text_mode(paragraph)
                    st.session_state.result = result
                    st.session_state.mode_used = "structured" if is_structured else "text"
                except Exception as e:
                    render_error(str(e))

    # ── Output ──
    st.markdown("<hr>", unsafe_allow_html=True)

    result = st.session_state.result
    mode_used = st.session_state.mode_used

    if result is None:
        render_empty_state()
    else:
        st.markdown('<div class="section-label">📊 Extraction Results</div>', unsafe_allow_html=True)
        if mode_used == "text":
            render_text_output(result)
        elif mode_used == "structured":
            render_structured_output(result, show_json)


if __name__ == "__main__":
    main()
