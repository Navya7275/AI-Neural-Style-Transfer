"""Neural Style Transfer Studio — Gradio application entry point."""

from nst import stylize
from ui import build_demo

demo = build_demo(stylize)

if __name__ == "__main__":
    demo.launch()
