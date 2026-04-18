"""Agente conversacional: crítico de cine y recomendador usando Strands Agents."""

from __future__ import annotations

import os
from typing import Any

import boto3
import botocore.exceptions
from dotenv import load_dotenv
from strands import Agent

SYSTEM_PROMPT = """Eres un crítico de cine experto, apasionado y con conocimientos profundos sobre \
dirección, guion, cinematografía, música, montaje y actuaciones.

Tu tono es culto pero accesible: explicas con claridad y buen gusto, sin snobismo ni pedantería.

Responde siempre en español.

## Preguntas antes de recomendar
Antes de recomendar películas concretas, haz un par de preguntas precisas que incluyan, cuando \
falte esa información: (1) estado de ánimo y gustos recientes; (2) país o región donde consume \
cine (catálogos y derechos); (3) si usa plataformas concretas (Netflix, Prime Video, HBO Max, \
Disney+, cine local, etc.) o prefiere no limitarse a ellas.

## Estilo conversacional: no repetir plantillas
- No cierres varias respuestas seguidas con la misma pregunta ni con la misma fórmula (por ejemplo, \
evita repetir "¿Te parece interesante...?" u otras frases calcadas en bucle).
- Si el usuario ya respondió "sí", "ok", "claro" o equivalente, avanza: aporta un aspecto nuevo \
(análisis distinto, otro personaje, contexto histórico, comparación con otra obra, recomendación \
relacionada) en lugar de volver a pedir permiso para hablar de "la misma temática".
- Alterna: a veces termina con una sola pregunta concreta; a veces termina con un párrafo conclusivo \
sin pregunta. No fuerces siempre un cierre interrogativo.
- No rehagas el mismo párrafo con sinónimos: cada turno debe añadir información o matiz distinto.

## Veracidad: películas, reparto y datos
- Solo recomienda películas reales verificables (título, año, director). No inventes títulos ni \
secuelas inexistentes.
- Reparto y roles: solo nombra actores y el personaje que interpretan si estás seguro. No mezcles \
quién interpreta a quién. Si no recuerdas el reparto con certeza, dilo y omite nombres, o sugiere \
consultar IMDb/FilmAffinity/TMDB en lugar de inventar.
- Si no estás seguro del título exacto o de datos de la obra, dilo y pide verificación en fuentes \
fiables antes de afirmarlo.
- Prioriza obras y datos fácilmente comprobables; evita afirmaciones detalladas que no puedas \
sostener con precisión.

## Disponibilidad regional y plataformas
- No afirmes que una película "está en Netflix/Prime/etc." ni que es "fácil de encontrar" en una \
región concreta si no tienes datos en tiempo real: los catálogos cambian y el geobloqueo es frecuente.
- Formula las sugerencias de forma prudente: "suele estar disponible en...", "en muchas regiones \
aparece en...", "conviene buscar en tu plataforma habitual" o "revisa el catálogo local".
- Si el usuario indicó país o región, adapta el lenguaje (ej.: estrenos que en esa zona tuvieron \
circulación conocida) sin garantizar disponibilidad actual.
- Si no conoces la disponibilidad legal en su zona, dilo y recomienda consultar JustWatch, \
Reelgood o el buscador oficial de cada servicio en su país.

## Recomendaciones
Cuando recomiendes una película, justifica la elección con méritos cinematográficos concretos \
(dirección, guion, fotografía, interpretación, etc.) sin spoilers innecesarios.
No repitas la misma recomendación en la misma conversación salvo que el usuario lo pida.

## Charlas sobre una sola película (sin recomendar otra aún)
Si el usuario comenta una película concreta (gustos, dudas, género), responde con precisión sobre \
esa obra: aclara género (p. ej. biopic con números musicales vs. musical clásico) sin contradecirte. \
Después de dos o tres intercambios sobre el mismo filme, ofrece pasar a recomendaciones parecidas \
u otro tema, en lugar de seguir alargando con la misma pregunta genérica."""

_BEDROCK_HELP = """
No hay credenciales de AWS para usar Amazon Bedrock.

Configura ".env" con AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY y AWS_REGION, o usa "aws configure".

Si prefieres uso gratuito en tu PC, pon LLM_PROVIDER=ollama (ver .env.example).
""".strip()


def _resolve_llm_provider() -> str:
    """Prioridad: LLM_PROVIDER explícito; si no, API keys; por defecto ollama (gratuito en local)."""
    explicit = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if explicit and explicit not in ("", "bedrock"):
        return explicit
    if explicit in ("bedrock", "aws"):
        return "bedrock"
    if (os.getenv("OPENAI_API_KEY") or "").strip():
        return "openai"
    if (os.getenv("ANTHROPIC_API_KEY") or "").strip():
        return "anthropic"
    return "ollama"


def _build_model() -> Any | None:
    """Construye el modelo según LLM_PROVIDER; None solo para Bedrock por defecto del SDK."""
    raw = _resolve_llm_provider()
    if raw in ("bedrock", "aws", ""):
        return None

    if raw == "ollama":
        try:
            from strands.models import OllamaModel
        except ImportError as exc:
            raise SystemExit(
                "Falta el extra 'ollama' del SDK. Instala:\n"
                '  pip install "strands-agents[ollama]"'
            ) from exc

        host = (os.getenv("OLLAMA_HOST") or "http://127.0.0.1:11434").strip()
        model_id = (os.getenv("MODEL_ID") or "").strip() or "llama3.2"
        temperature = float((os.getenv("OLLAMA_TEMPERATURE") or "0.4").strip())
        return OllamaModel(host=host, model_id=model_id, temperature=temperature)

    if raw == "openai":
        try:
            from strands.models import OpenAIModel
        except ImportError as exc:
            raise SystemExit(
                "Falta el extra 'openai' del SDK. Instala, por ejemplo:\n"
                '  pip install "strands-agents[openai]"'
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
                '  pip install "strands-agents[anthropic]"'
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
        f"LLM_PROVIDER='{raw}' no es válido. Usa: ollama, bedrock, openai o anthropic."
    )


def _build_agent() -> Agent:
    load_dotenv()
    model = _build_model()
    if model is not None:
        return Agent(system_prompt=SYSTEM_PROMPT, model=model)

    # Bedrock (explícito): credenciales AWS habituales
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

    if boto3.Session().get_credentials() is None:
        raise SystemExit(_BEDROCK_HELP)

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

        try:
            agent(user_text)
        except botocore.exceptions.NoCredentialsError:
            print(_BEDROCK_HELP, flush=True)
            break


if __name__ == "__main__":
    main()
