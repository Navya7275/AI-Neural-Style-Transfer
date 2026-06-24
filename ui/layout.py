"""Gradio Blocks layout — assembles widgets and wires them to the pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from ui.css import STYLESHEET
from ui.theme import build_theme
from ui.components import (
    GRAIN_OVERLAY,
    MASTHEAD,
    PIPELINE,
    TECH_BAR,
    FOOTER,
    DIVIDER,
    section_header,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def build_demo(stylize_fn: Callable) -> gr.Blocks:
    """Construct the full Gradio interface, decoupled from the backend."""

    with gr.Blocks(
        css=STYLESHEET,
        theme=build_theme(),
        title="Neural Style Transfer Studio",
    ) as demo:

        gr.HTML(GRAIN_OVERLAY)
        gr.HTML(MASTHEAD)
        gr.HTML(DIVIDER)
        gr.HTML(PIPELINE)

        gr.HTML(section_header("Source Images"))

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                content_input = gr.Image(
                    type="filepath",
                    label="Content — Your Photograph",
                    elem_classes="upload-card",
                    height=280,
                )
            with gr.Column(scale=1):
                style_input = gr.Image(
                    type="filepath",
                    label="Style — Artistic Reference",
                    elem_classes="upload-card",
                    height=280,
                )

        with gr.Accordion("Parameters", open=False, elem_classes="controls-accordion"):
            with gr.Row():
                style_strength = gr.Slider(
                    minimum=1, maximum=10, value=5, step=0.5,
                    label="Style Intensity",
                    info="1 = subtle wash · 10 = full stylization",
                )
                content_preservation = gr.Slider(
                    minimum=1, maximum=10, value=6, step=0.5,
                    label="Content Fidelity",
                    info="1 = loose interpretation · 10 = photographic",
                )
            with gr.Row():
                smoothing = gr.Slider(
                    minimum=0, maximum=10, value=2, step=0.5,
                    label="Smoothing",
                    info="Reduces noise and artifacts",
                )
                steps = gr.Slider(
                    minimum=100, maximum=600, value=400, step=50,
                    label="Iterations",
                    info="Higher = finer result, longer processing",
                )
            with gr.Row():
                resolution = gr.Slider(
                    minimum=256, maximum=768, value=512, step=64,
                    label="Resolution",
                    info="Output size (longest edge in px)",
                )
                contrast = gr.Slider(
                    minimum=0.9, maximum=1.3, value=1.05, step=0.05,
                    label="Contrast",
                    info="Post-processing contrast adjustment",
                )
            with gr.Row():
                color_match = gr.Checkbox(value=True, label="Histogram color matching")
                sharpening = gr.Checkbox(value=True, label="Output sharpening")

        gr.HTML('<div style="height: 0.5rem"></div>')
        run_btn = gr.Button("Generate", elem_id="generate-btn")

        gr.HTML(section_header("Result"))
        output_img = gr.Image(
            label="Stylized Output",
            elem_classes="output-frame",
            height=500,
        )

        gr.HTML(TECH_BAR)
        gr.HTML(FOOTER)

        run_btn.click(
            fn=stylize_fn,
            inputs=[
                content_input, style_input, style_strength, content_preservation,
                smoothing, steps, color_match, sharpening, contrast, resolution,
            ],
            outputs=output_img,
        )

    return demo
