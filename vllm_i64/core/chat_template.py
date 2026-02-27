"""
vllm-i64 :: Chat Template

Apply chat templates to messages for conversational models.
Loads Jinja2 templates from checkpoint directories.

INL - 2025
"""

import os
from typing import List, Dict, Optional


class ChatTemplate:
    """
    Chat template renderer.

    Loads a Jinja2 template (like the one in pacific-prime-chat)
    and renders messages into a prompt string.
    """

    def __init__(self, template_str: str):
        from jinja2 import Template
        self.template = Template(template_str)

    def apply(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Render messages into a prompt string.

        Args:
            messages: [{"role": "user", "content": "..."}, ...]
            add_generation_prompt: append assistant turn marker

        Returns:
            formatted prompt string
        """
        return self.template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt,
        )

    @staticmethod
    def from_file(path: str) -> "ChatTemplate":
        """Load template from a .jinja file."""
        with open(path, "r", encoding="utf-8") as f:
            return ChatTemplate(f.read())


def load_chat_template(model_name: str) -> Optional[ChatTemplate]:
    """
    Load chat template for a registered model.

    Looks for chat_template.jinja next to config.json.
    """
    from vllm_i64.core.registry import get_model_entry

    entry = get_model_entry(model_name)
    if not entry.config_path:
        return None

    config_dir = os.path.dirname(entry.config_path)

    # Look for chat_template.jinja
    for name in ["chat_template.jinja", "chat_template.j2", "template.jinja"]:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            print(f"  chat_template: {path}")
            return ChatTemplate.from_file(path)

    # Try parent directory
    parent_dir = os.path.dirname(config_dir)
    for name in ["chat_template.jinja", "chat_template.j2"]:
        path = os.path.join(parent_dir, name)
        if os.path.exists(path):
            print(f"  chat_template: {path}")
            return ChatTemplate.from_file(path)

    return None
