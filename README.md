# Agente crítico de cine

Agente conversacional en **español** que actúa como crítico de cine y recomendador de películas, construido con el [SDK de Strands Agents](https://github.com/strands-agents/sdk-python) para Python.

## Características

- Recomendaciones y análisis con tono culto y accesible.
- Pregunta por estado de ánimo, gustos, **región** y **plataformas** antes de recomendar cuando hace falta.
- Instrucciones en el *system prompt* para **no inventar películas**, ser prudente con **disponibilidad regional** y **evitar respuestas repetitivas**.
- Conversación por terminal; el historial se mantiene en la misma ejecución.
- Varios proveedores de modelo: **Ollama** (local y gratuito por defecto), OpenAI, Anthropic y Amazon Bedrock.

## Requisitos

- **Python 3.10 o superior**
- Para **Ollama** (recomendado en local): [instalar Ollama](https://ollama.com) y descargar un modelo, por ejemplo:

  ```bash
  ollama pull llama3.2
  ```

## Instalación

```powershell
cd "ruta\al\proyecto"
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

En Linux o macOS, activa el entorno con `source venv/bin/activate`.

### Instalar el SDK desde Git (opcional)

Si prefieres la última versión desde el repositorio (requiere **Git** en el PATH):

```powershell
pip install git+https://github.com/strands-agents/sdk-python.git
pip install "strands-agents[ollama]"
pip install python-dotenv
```

## Configuración

1. Copia el archivo de ejemplo y créalo como `.env`:

   ```powershell
   copy .env.example .env
   ```

2. Edita `.env` según el proveedor que vayas a usar (no subas `.env` al repositorio; ya está en `.gitignore`).

### Proveedores de modelo

| Modo | Descripción breve |
|------|---------------------|
| **ollama** (por defecto) | Modelo en tu PC; sin API de pago. Ajusta `MODEL_ID` y `OLLAMA_HOST` si hace falta. |
| **openai** | Requiere `OPENAI_API_KEY` y `pip install "strands-agents[openai]"`. |
| **anthropic** | Requiere `ANTHROPIC_API_KEY` y `pip install "strands-agents[anthropic]"`. |
| **bedrock** | AWS con acceso a Bedrock; define `LLM_PROVIDER=bedrock` y credenciales `AWS_*`. |

La variable `LLM_PROVIDER` puede dejarse vacía: si solo tienes `OPENAI_API_KEY` o `ANTHROPIC_API_KEY`, se usará ese proveedor; si no, **Ollama**.

## Ejecución

```powershell
.\venv\Scripts\Activate.ps1
python agent.py
```

Comandos para salir: `salir`, `exit` o `quit`.

## Estructura del proyecto

```
.
├── agent.py           # Agente Strands, system prompt y bucle conversacional
├── requirements.txt   # Dependencias
├── .env.example       # Plantilla de variables de entorno
├── .gitignore
└── .github/           # Plantillas de issues y PR en español
```

## Documentación y soporte

- [Strands Agents — documentación](https://strandsagents.com/)
- Para **propuestas** o **errores**, usa las plantillas al crear un [issue](https://github.com/foxdeyvid88/Agente-critico-de-cine/issues/new/choose) en este repositorio.

## Licencia

El código del proyecto se ofrece según la licencia que indique el autor del repositorio. El SDK `strands-agents` tiene su propia licencia en el [repositorio oficial](https://github.com/strands-agents/sdk-python).
