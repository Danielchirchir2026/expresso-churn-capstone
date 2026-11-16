import streamlit as st

st.set_page_config(
    page_title="Expresso Customer Churn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Headline numbers (edit here if the model or data change)
# ---------------------------------------------------------------------------
TOTAL_CUSTOMERS = 2_154_048
CHURNED_CUSTOMERS = 403_986
MODEL_AUC = 0.9317
N_FEATURES = 69
N_SEGMENTS = 5
MODEL_NAME = "LightGBM"

churn_share = CHURNED_CUSTOMERS / TOTAL_CUSTOMERS
churn_pct = f"{churn_share * 100:.2f}%"
stay_pct = f"{(1 - churn_share) * 100:.2f}%"
churn_deg = round(churn_share * 360, 1)

UNDERSTAND = [
    ("📈", "Executive Dashboard", "Overall churn risk, customer distribution and key insights."),
    ("👥", "Customer Segmentation", f"Explore the {N_SEGMENTS} customer groups and how each one churns."),
    ("🤖", "Model Performance", "AUC, ROC curve, confusion matrix and feature importance."),
]
ACT = [
    ("🎯", "Churn Prediction", "Score the churn probability of one customer."),
    ("📂", "Batch Prediction", "Upload a CSV and score many customers at once."),
    ("📑", "Reports", "Download prediction results and summary reports."),
]
TEAM = [
    "Daniel Chirchir",
    "Daniel Tabut",
    "Livingstone Talel",
    "Sandra Bosibori",
    "Patrick Palex",
    "Immanuel Ochieng",
]

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Sora:wght@500;700;800&display=swap');

:root {
  --ink: #E8EEF7;
  --muted: #8FA3BF;
  --line: rgba(143, 163, 191, 0.22);
  --churn: #FF5D73;
  --stay: #35D0BA;
  --understand: #6EA8FE;
  --act: #FFB454;
}

html, body, [class*="css"], .stMarkdown, .stMarkdown p {
  font-family: 'DM Sans', sans-serif;
}
.block-container { padding-top: 2.5rem; max-width: 1180px; }
footer { visibility: hidden; }

/* Hero ------------------------------------------------------------------ */
.hero {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: 2.5rem;
  align-items: center;
  padding: 1.5rem 0 2rem 0;
}
.hero h1 {
  font-family: 'Sora', sans-serif;
  font-weight: 800;
  font-size: 2.9rem;
  line-height: 1.12;
  letter-spacing: -0.02em;
  margin: 0 0 1rem 0;
  padding: 0;
  color: var(--ink);
}
.hero p {
  font-size: 1.12rem;
  line-height: 1.65;
  color: var(--muted);
  max-width: 34rem;
  margin: 0;
}

/* The churn ring: the one animated moment on the page --------------------- */
@property --sweep { syntax: '<angle>'; inherits: false; initial-value: 0deg; }
.ring-wrap { position: relative; width: 270px; max-width: 100%; margin: 0 auto; }
.ring {
  --sweep: 0deg;
  width: 100%;
  aspect-ratio: 1;
  border-radius: 50%;
  background: conic-gradient(var(--churn) 0deg var(--sweep), var(--stay) var(--sweep) 360deg);
  -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 28px), #000 calc(100% - 27px));
  mask: radial-gradient(farthest-side, transparent calc(100% - 28px), #000 calc(100% - 27px));
  animation: sweep 1.6s cubic-bezier(0.2, 0.7, 0.2, 1) forwards;
}
@keyframes sweep { to { --sweep: __DEG__deg; } }
.ring-label {
  position: absolute; inset: 0;
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  text-align: center;
}
.ring-label b {
  font-family: 'Sora', sans-serif;
  font-size: 2.3rem; font-weight: 800; color: var(--churn);
}
.ring-label span { color: var(--muted); font-size: 0.95rem; line-height: 1.35; }
.ring-key {
  display: flex; justify-content: center; gap: 1.4rem;
  margin-top: 1rem; color: var(--muted); font-size: 0.9rem;
}
.dot { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 50%; margin-right: 0.4rem; }

/* Facts strip ------------------------------------------------------------ */
.facts {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  margin: 0.5rem 0 2.5rem 0;
}
.fact { padding: 1.2rem 1.4rem; border-left: 1px solid var(--line); }
.fact:first-child { border-left: none; padding-left: 0; }
.fact b {
  display: block;
  font-family: 'Sora', sans-serif;
  font-size: 1.7rem; font-weight: 700; color: var(--ink);
}
.fact span { color: var(--muted); font-size: 0.92rem; }

/* Modules ---------------------------------------------------------------- */
.modules { display: grid; grid-template-columns: 1fr 1fr; gap: 2.5rem; }
.modules h3 {
  font-family: 'Sora', sans-serif;
  font-size: 1.25rem; font-weight: 700; color: var(--ink);
  margin: 0 0 0.3rem 0; padding: 0;
}
.modules .sub { color: var(--muted); font-size: 0.95rem; margin-bottom: 1rem; }
.mod {
  display: flex; gap: 0.9rem; align-items: flex-start;
  padding: 0.85rem 1rem;
  margin-bottom: 0.7rem;
  border-left: 3px solid var(--accent);
  background: rgba(143, 163, 191, 0.07);
  border-radius: 0 10px 10px 0;
}
.mod .ico { font-size: 1.4rem; line-height: 1.3; }
.mod b { display: block; color: var(--ink); font-size: 1.02rem; }
.mod span { color: var(--muted); font-size: 0.93rem; line-height: 1.45; }
.understand { --accent: var(--understand); }
.act { --accent: var(--act); }

/* Hint and team ---------------------------------------------------------- */
.hint {
  margin: 2.2rem 0 0.5rem 0;
  padding: 0.9rem 1.2rem;
  border: 1px dashed var(--line);
  border-radius: 10px;
  color: var(--muted);
}
.team-title {
  font-family: 'Sora', sans-serif;
  font-size: 1.25rem; font-weight: 700; color: var(--ink);
  margin: 2.5rem 0 0.2rem 0;
}
.team-sub { color: var(--muted); margin-bottom: 1rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.6rem; }
.chip {
  padding: 0.45rem 0.95rem;
  border: 1px solid var(--line);
  border-radius: 999px;
  color: var(--ink); font-size: 0.95rem;
}

@media (max-width: 860px) {
  .hero, .modules { grid-template-columns: 1fr; }
  .hero h1 { font-size: 2.1rem; }
  .facts { grid-template-columns: 1fr 1fr; }
  .fact:nth-child(3) { border-left: none; padding-left: 0; }
}
@media (prefers-reduced-motion: reduce) {
  .ring { animation: none; --sweep: __DEG__deg; }
}
</style>
""".replace("__DEG__", str(churn_deg))

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers (HTML is built without indentation or blank lines so Streamlit's
# markdown parser does not treat it as a code block)
# ---------------------------------------------------------------------------
def module_list(items, css_class):
    rows = [
        f'<div class="mod {css_class}"><div class="ico">{icon}</div>'
        f"<div><b>{name}</b><span>{desc}</span></div></div>"
        for icon, name, desc in items
    ]
    return "".join(rows)


hero = (
    '<div class="hero">'
    "<div>"
    "<h1>Which Expresso customers are about to leave, and why?</h1>"
    "<p>Churn analytics and prediction for Expresso customers in Senegal. "
    "Explore what the data shows, then score customers yourself.</p>"
    "</div>"
    "<div>"
    '<div class="ring-wrap"><div class="ring"></div>'
    f'<div class="ring-label"><b>{churn_pct}</b><span>of customers<br>churned</span></div></div>'
    '<div class="ring-key">'
    f'<span><i class="dot" style="background:var(--churn)"></i>Churned {CHURNED_CUSTOMERS:,}</span>'
    f'<span><i class="dot" style="background:var(--stay)"></i>Stayed {stay_pct}</span>'
    "</div>"
    "</div>"
    "</div>"
)

facts = (
    '<div class="facts">'
    f'<div class="fact"><b>{TOTAL_CUSTOMERS:,}</b><span>customers analysed</span></div>'
    f'<div class="fact"><b>{N_FEATURES}</b><span>engineered features</span></div>'
    f'<div class="fact"><b>{MODEL_NAME}</b><span>production model</span></div>'
    f'<div class="fact"><b>{MODEL_AUC}</b><span>validation ROC-AUC</span></div>'
    "</div>"
)

modules = (
    '<div class="modules">'
    "<div><h3>Understand the churn</h3>"
    '<div class="sub">See where customers are leaving and what drives it.</div>'
    f'{module_list(UNDERSTAND, "understand")}</div>'
    "<div><h3>Act on it</h3>"
    '<div class="sub">Score customers and share the results.</div>'
    f'{module_list(ACT, "act")}</div>'
    "</div>"
)

team = (
    '<div class="team-title">Team Spartans</div>'
    '<div class="team-sub">Capstone project: Expresso Customer Churn Prediction and Retention Analytics</div>'
    '<div class="chips">' + "".join(f'<span class="chip">{n}</span>' for n in TEAM) + "</div>"
)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown(hero, unsafe_allow_html=True)
st.markdown(facts, unsafe_allow_html=True)
st.markdown(modules, unsafe_allow_html=True)
st.markdown(
    '<div class="hint">Pick a module from the sidebar to begin.</div>',
    unsafe_allow_html=True,
)
st.markdown(team, unsafe_allow_html=True)

st.caption("Spartans | Expresso Customer Churn Capstone Project | 2026")
