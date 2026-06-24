"""Gallery-aesthetic CSS for the Gradio interface."""

STYLESHEET = r"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400;1,500&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --ivory: #FAF7F2;
    --ivory-mid: #F0EBE3;
    --ivory-deep: #E5DFD5;
    --charcoal: #1C1917;
    --charcoal-soft: #292524;
    --charcoal-mid: #3D3835;
    --warm-stone: #A8A29E;
    --warm-sand: #D6D3D1;
    --accent-copper: #C2733F;
    --accent-copper-light: #D4956A;
    --text-ink: #1C1917;
    --text-body: #44403C;
    --text-caption: #78716C;
    --text-ghost: #A8A29E;
    --font-display: 'Cormorant Garamond', Georgia, serif;
    --font-body: 'Outfit', system-ui, sans-serif;
    --radius: 6px;
    --transition: 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: var(--ivory) !important;
    color: var(--text-ink) !important;
    font-family: var(--font-body) !important;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 960px !important;
    margin: 0 auto !important;
    padding: 0 2rem !important;
}

/* Grain overlay */
.grain-overlay {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    pointer-events: none;
    z-index: 9999;
    opacity: 0.03;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 256px 256px;
}

/* Masthead */
.masthead {
    text-align: center;
    padding: 4rem 1rem 3rem;
    position: relative;
}
.masthead::after {
    content: "";
    display: block;
    width: 48px;
    height: 1.5px;
    background: var(--accent-copper);
    margin: 2rem auto 0;
}
.masthead-eyebrow {
    font-family: var(--font-body);
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent-copper);
    margin-bottom: 1rem;
    display: block;
}
.masthead-title {
    font-family: var(--font-display);
    font-size: 3.6rem;
    font-weight: 400;
    font-style: italic;
    letter-spacing: -0.01em;
    line-height: 1.05;
    color: var(--text-ink);
    margin: 0 0 1rem;
}
.masthead-desc {
    font-family: var(--font-body);
    font-size: 0.92rem;
    font-weight: 300;
    color: var(--text-caption);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.7;
}

.divider {
    border: none;
    height: 1px;
    background: var(--ivory-deep);
    margin: 0.5rem 0 2rem;
}

/* Pipeline */
.pipeline-wrap {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0;
    margin: 0 0 2.5rem;
    border: 1px solid var(--ivory-deep);
    border-radius: var(--radius);
    overflow: hidden;
    background: white;
}
.pipeline-step {
    padding: 1.4rem 1rem;
    text-align: center;
    border-right: 1px solid var(--ivory-deep);
    transition: background var(--transition);
}
.pipeline-step:last-child { border-right: none; }
.pipeline-step:hover { background: var(--ivory-mid); }
.pipeline-num {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent-copper);
    margin-bottom: 0.3rem;
    line-height: 1;
}
.pipeline-name {
    font-family: var(--font-body);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-body);
    margin-bottom: 0.35rem;
}
.pipeline-detail {
    font-size: 0.7rem;
    color: var(--text-ghost);
    line-height: 1.4;
    font-weight: 300;
}

/* Section headers */
.section-head {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1.2rem;
}
.section-head-line {
    flex: 1;
    height: 1px;
    background: var(--ivory-deep);
}
.section-head-text {
    font-family: var(--font-body);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-caption);
    white-space: nowrap;
}

/* Uploads */
.upload-card {
    background: white !important;
    border: 1px solid var(--ivory-deep) !important;
    border-radius: var(--radius) !important;
    transition: border-color var(--transition), box-shadow var(--transition) !important;
}
.upload-card:hover {
    border-color: var(--accent-copper-light) !important;
    box-shadow: 0 2px 16px rgba(194, 115, 63, 0.06) !important;
}
.upload-card > div,
.upload-card .upload-area {
    background: white !important;
    border: none !important;
}
.upload-card label span {
    font-family: var(--font-body) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: var(--text-caption) !important;
}

/* Accordion */
.controls-accordion {
    background: white !important;
    border: 1px solid var(--ivory-deep) !important;
    border-radius: var(--radius) !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1.5rem !important;
}
.controls-accordion > .label-wrap {
    background: transparent !important;
    padding: 0.9rem 1.2rem !important;
}
.controls-accordion > .label-wrap span {
    font-family: var(--font-body) !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: var(--text-caption) !important;
}

.gradio-slider label span,
.gradio-checkbox label span,
.gradio-checkbox label {
    font-family: var(--font-body) !important;
    color: var(--text-body) !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
}
input[type=range] { accent-color: var(--accent-copper) !important; }
.gradio-slider input[type="number"] {
    background: var(--ivory) !important;
    border: 1px solid var(--ivory-deep) !important;
    color: var(--text-ink) !important;
    border-radius: var(--radius) !important;
}

/* Button */
#generate-btn {
    background: var(--charcoal) !important;
    color: var(--ivory) !important;
    font-family: var(--font-body) !important;
    font-weight: 500 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    padding: 1rem 3rem !important;
    border: none !important;
    border-radius: var(--radius) !important;
    cursor: pointer !important;
    transition: all var(--transition) !important;
    box-shadow: 0 2px 8px rgba(28, 25, 23, 0.15) !important;
    margin: 0 auto !important;
    display: block !important;
    min-width: 260px !important;
}
#generate-btn:hover {
    background: var(--charcoal-mid) !important;
    box-shadow: 0 4px 20px rgba(28, 25, 23, 0.25) !important;
    transform: translateY(-1px) !important;
}

/* Output frame */
.output-frame {
    background: white !important;
    border: 1px solid var(--ivory-deep) !important;
    border-radius: var(--radius) !important;
    padding: 1.2rem !important;
    margin-top: 0.5rem !important;
    box-shadow: 0 1px 8px rgba(28, 25, 23, 0.04) !important;
    transition: box-shadow var(--transition) !important;
}
.output-frame:hover {
    box-shadow: 0 4px 24px rgba(28, 25, 23, 0.08) !important;
}
.output-frame img { border-radius: 3px !important; }
.output-frame label span {
    font-family: var(--font-display) !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    font-style: italic !important;
    color: var(--text-body) !important;
}

/* Tech bar */
.tech-bar {
    display: flex;
    justify-content: center;
    gap: 2rem;
    padding: 1rem 0;
    margin: 2rem 0 0.5rem;
    border-top: 1px solid var(--ivory-deep);
    border-bottom: 1px solid var(--ivory-deep);
}
.tech-item { text-align: center; }
.tech-value {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-ink);
}
.tech-label {
    font-size: 0.6rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-ghost);
    margin-top: 0.25rem;
}

.footer {
    text-align: center;
    padding: 2.5rem 1rem 2rem;
    margin-top: 1.5rem;
}
.footer-line {
    font-family: var(--font-display);
    font-size: 0.9rem;
    font-style: italic;
    color: var(--text-ghost);
    margin-bottom: 0.5rem;
}
.footer-meta {
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--warm-sand);
}

/* Gradio overrides */
.gr-box { background: white !important; border-color: var(--ivory-deep) !important; }
.gr-input { background: var(--ivory) !important; border-color: var(--ivory-deep) !important; color: var(--text-ink) !important; }
.gradio-accordion { background: white !important; border: 1px solid var(--ivory-deep) !important; border-radius: var(--radius) !important; }
.gr-check-radio { border-color: var(--warm-stone) !important; }
.gr-check-radio.selected { background: var(--accent-copper) !important; border-color: var(--accent-copper) !important; }

@media (max-width: 768px) {
    .masthead-title { font-size: 2.4rem; }
    .pipeline-wrap { grid-template-columns: repeat(2, 1fr); }
    .pipeline-step:nth-child(2) { border-right: none; }
    .tech-bar { flex-wrap: wrap; gap: 1.2rem; }
}
@media (max-width: 480px) {
    .pipeline-wrap { grid-template-columns: 1fr; }
    .pipeline-step { border-right: none; border-bottom: 1px solid var(--ivory-deep); }
    .pipeline-step:last-child { border-bottom: none; }
}
"""
