"""Gradio theme — warm neutral palette inspired by art galleries."""

import gradio as gr


def build_theme() -> gr.themes.Base:
    """Create a custom Gradio theme with copper accents on ivory."""
    return gr.themes.Base(
        primary_hue=gr.themes.Color(
            c50="#FDF8F3", c100="#FAF0E6", c200="#F0E0CC", c300="#E0C9A8",
            c400="#D4A574", c500="#C2733F", c600="#A85C30", c700="#8B4826",
            c800="#6E381E", c900="#522A17", c950="#3A1D10",
        ),
        secondary_hue=gr.themes.Color(
            c50="#FAFAF9", c100="#F5F5F4", c200="#E7E5E4", c300="#D6D3D1",
            c400="#A8A29E", c500="#78716C", c600="#57534E", c700="#44403C",
            c800="#292524", c900="#1C1917", c950="#0C0A09",
        ),
        neutral_hue=gr.themes.Color(
            c50="#FAF7F2", c100="#F0EBE3", c200="#E5DFD5", c300="#D6D3D1",
            c400="#A8A29E", c500="#78716C", c600="#57534E", c700="#44403C",
            c800="#292524", c900="#1C1917", c950="#0C0A09",
        ),
        font=[gr.themes.GoogleFont("Outfit"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_color="*neutral_200",
        block_border_width="1px",
        block_label_text_color="*secondary_500",
        block_title_text_color="*secondary_700",
        input_background_fill="*neutral_50",
        input_border_color="*neutral_200",
        button_primary_background_fill="*secondary_900",
        button_primary_text_color="*neutral_50",
        button_primary_background_fill_hover="*secondary_700",
        slider_color="*primary_500",
        checkbox_background_color_selected="*primary_500",
        shadow_drop="0 1px 4px rgba(0,0,0,0.04)",
        shadow_drop_lg="0 4px 16px rgba(0,0,0,0.06)",
    )
