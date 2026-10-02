"""Builds the Jinja environment"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
    undefined=StrictUndefined,
)

def render(name: str, **variables) -> str:
    
    return env.get_template(name).render(**variables).strip()