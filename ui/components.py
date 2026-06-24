"""Reusable HTML fragments for the Gradio interface."""


GRAIN_OVERLAY = '<div class="grain-overlay"></div>'

MASTHEAD = """
<div class="masthead">
    <span class="masthead-eyebrow">Gatys et al. &middot; VGG-19 &middot; L-BFGS Optimization</span>
    <h1 class="masthead-title">Neural Style Transfer</h1>
    <p class="masthead-desc">
        Upload a photograph and a style reference. The algorithm deconstructs
        artistic texture, color, and brushwork — then reconstructs your image
        as if painted in that style.
    </p>
</div>
"""

PIPELINE = """
<div class="pipeline-wrap">
    <div class="pipeline-step">
        <div class="pipeline-num">I</div>
        <div class="pipeline-name">Encode</div>
        <div class="pipeline-detail">VGG-19 extracts features<br>across 5 relu layers</div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">II</div>
        <div class="pipeline-name">Match</div>
        <div class="pipeline-detail">Gram matrices capture<br>style statistics</div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">III</div>
        <div class="pipeline-name">Optimize</div>
        <div class="pipeline-detail">L-BFGS minimizes<br>content + style loss</div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-num">IV</div>
        <div class="pipeline-name">Refine</div>
        <div class="pipeline-detail">TV smoothing, sharpening<br>&amp; color correction</div>
    </div>
</div>
"""

TECH_BAR = """
<div class="tech-bar">
    <div class="tech-item"><div class="tech-value">VGG-19</div><div class="tech-label">Backbone</div></div>
    <div class="tech-item"><div class="tech-value">5</div><div class="tech-label">Style Layers</div></div>
    <div class="tech-item"><div class="tech-value">L-BFGS</div><div class="tech-label">Optimizer</div></div>
    <div class="tech-item"><div class="tech-value">Gram</div><div class="tech-label">Style Metric</div></div>
    <div class="tech-item"><div class="tech-value">TV</div><div class="tech-label">Regularizer</div></div>
</div>
"""

FOOTER = """
<div class="footer">
    <div class="footer-line">Every image is a new canvas.</div>
    <div class="footer-meta">Built with PyTorch &amp; Gradio &middot; Gatys Neural Style Transfer</div>
</div>
"""

DIVIDER = '<hr class="divider">'


def section_header(text: str) -> str:
    """Return an HTML section divider with centred label."""
    return f"""
    <div class="section-head">
        <div class="section-head-line"></div>
        <span class="section-head-text">{text}</span>
        <div class="section-head-line"></div>
    </div>"""
