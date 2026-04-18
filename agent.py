"""Agente conversacional: crítico de cine y recomendador usando Strands Agents."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from strands import Agent

SYSTEM_PROMPT = """Eres un crítico de cine experto, apasionado y con conocimientos profundos sobre \
dirección, guion, cinematografía, música, montaje y actuaciones.

Tu tono es culto pero accesible: explicas con claridad y buen gusto, sin snobismo ni pedantería.

Antes de recomendar películas concretas, haz siempre un par de preguntas precisas para entender \
el estado de ánimo del usuario y sus gustos recientes (por ejemplo: qué ha visto últimamente, \
qué ritmo o género busca, si prefiere algo más ligero o más exigente).

Responde siempre en español.

Cuando recomiendes una película, justifica la elección: explica por qué es una buena obra \
cinematográfica (ideas de dirección, coherencia dramática, innovación visual, trabajo de reparto, \
u otros méritos concretos), sin spoilers innecesarios."""


def _build_model() -> Any | None:
    """Construye el modelo según LLM_PROVIDER; None usa el proveedor por defecto del SDK (Bedrock)."""
    raw = (os.getenv("LLM_PROVIDER") or "bedrock").strip().lower()
    if raw in ("bedrock", "aws", ""):
        return None

    if raw == "openai":
        try:
            from strands.models import OpenAIModel
        except ImportError as exc:
            raise SystemExit(
                "Falta el extra 'openai' del SDK. Instala, por ejemplo:\n"
                '  pip install "strands-agents[openai] @ git+https://github.com/strands-agents/sdk-python.git"'
            ) from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY no está definida en el entorno o en .env")

        model_id = (os.getenv("MODEL_ID") or "").strip() or "gpt-4o"
        client_args: dict[str, str] = {"api_key": api_key}
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        if base_url:
            client_args["base_url"] = base_url

        return OpenAIModel(model_id=model_id, client_args=client_args)

    if raw == "anthropic":
        try:
            from strands.models import AnthropicModel
        except ImportError as exc:
            raise SystemExit(
                "Falta el extra 'anthropic' del SDK. Instala, por ejemplo:\n"
                '  pip install "strands-agents[anthropic] @ git+https://github.com/strands-agents/sdk-python.git"'
            ) from exc

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit("ANTHROPIC_API_KEY no está definida en el entorno o en .env")

        model_id = (os.getenv("MODEL_ID") or "").strip() or "claude-sonnet-4-20250514"
        return AnthropicModel(
            client_args={"api_key": api_key},
            model_id=model_id,
            max_tokens=4096,
        )

    raise SystemExit(
        f"LLM_PROVIDER='{raw}' no es válido. Usa: bedrock, openai o anthropic."
    )


def _build_agent() -> Agent:
    load_dotenv()
    model = _build_model()
    if model is not None:
        return Agent(system_prompt=SYSTEM_PROMPT, model=model)

    # Bedrock (por defecto): credenciales AWS habituales (p. ej. AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    # y región; opcionalmente fija MODEL_ID y región vía variables de entorno.
    model_id = (os.getenv("MODEL_ID") or "").strip()
    if model_id:
        from strands.models import BedrockModel

        region = (
            (os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "").strip()
            or "us-west-2"
        )
        return Agent(
            system_prompt=SYSTEM_PROMPT,
            model=BedrockModel(
                model_id=model_id,
                region_name=region,
                temperature=0.4,
            ),
        )

    return Agent(system_prompt=SYSTEM_PROMPT)


def main() -> None:
    print(
        "Crítico de cine (Strands Agents). Escribe tu mensaje. "
        "Comandos de salida: salir, exit, quit.\n",
        flush=True,
    )
    agent = _build_agent()

    while True:
        try:
            user_text = input("Tú: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta pronto.", flush=True)
            break

        if not user_text:
            continue
        lowered = user_text.lower()
        if lowered in ("salir", "exit", "quit"):
            print("Hasta pronto.", flush=True)
            break

        agent(user_text)


if __name__ == "__main__":
    main()
