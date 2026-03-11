# OmniParser: Refactored (Experimental)

This repository is a refactored version of [Microsoft/OmniParser](https://github.com/microsoft/OmniParser)
designed for cleaner code organisation and easier integration into downstream enterprise applications.

**Note on Development:** The entirety of this refactoring was performed with the assistance of **GitHub Copilot**.

**Disclaimer:** This version has only been tested within a specific internal environment and has not been
validated for general public use. It should be considered experimental. Because testing has been limited
to internal infrastructure, users may encounter issues in different environments.

## Original Project References

- Project Page: https://microsoft.github.io/OmniParser/
- Paper: https://arxiv.org/abs/2408.00203
- Models: https://huggingface.co/microsoft/OmniParser-v2.0

---

## Architecture Overview

Three components work together:

```
Windows Host (target machine)
        ↑  screenshots / actions
Agent UI  ←→  OmniParser Server  ←→  Core Parser (util/)
(omnitool/gradio/)   (omniparserserver/)   YOLO + OCR + Caption
```

| Component | Path | Purpose |
|---|---|---|
| Core parser | `util/` | YOLO detection, OCR, caption generation |
| Parser server | `omnitool/omniparserserver/` | FastAPI `POST /parse/` endpoint |
| Agent UI | `omnitool/gradio/` | Gradio interface + multi-agent orchestration |
| Legacy agents | `omnitool/gradio_legacy/` | Old implementation — reference only |

---

## Key Changes from Upstream

- **Modular architecture** — dedicated packages for utilities, server, and UI; absolute imports; PEP 8 throughout.
- **PaddleOCR GPU backend** — optional remote GPU-accelerated OCR server; local EasyOCR retained as default.
- **Expanded LLM support** — Azure OpenAI (managed identities / service principals), plus 9 total providers.
- **No VM dependency** — omnibox removed; the target environment does not support virtualisation.

---

## Requirements & Installation

**Prerequisites:** Python 3.12, [conda](https://www.anaconda.com/download/success)

```bash
# 1. Create and activate environment
conda create -n omni python=3.12
conda activate omni

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Download model weights
for folder in icon_caption icon_detect; do
    huggingface-cli download microsoft/OmniParser-v2.0 \
        --local-dir weights --repo-type model --include "$folder/*"
done
mv weights/icon_caption weights/icon_caption_florence
```

Expected weight layout:
```
weights/
├── icon_detect/
│   └── model.pt
└── icon_caption_florence/
```

---

## Running the Components

### 1. OmniParser Server

```bash
conda activate omni
python -m omnitool.omniparserserver
```

Key CLI options:

| Flag | Default | Description |
|---|---|---|
| `--som_model_path` | `weights/icon_detect/model.pt` | YOLO model weights |
| `--caption_model_name` | `florence2` | `florence2` or `blip2` |
| `--caption_model_path` | `weights/icon_caption_florence` | Caption model path |
| `--BOX_TRESHOLD` | `0.05` | YOLO confidence threshold |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Listen port |

### 2. Agent Gradio UI

```bash
conda activate omni
python omnitool/gradio/ui/gradio/app.py \
    --omniparser_server_url http://localhost:8000 \
    --windows_host_url http://localhost:5000
```

### 3. (Optional) Remote PaddleOCR GPU Server

Set `PADDLEOCR_URL` to point at a running PaddleOCR 3.x API server.
See `paddleocr_api_example.py` for the expected request/response schema.

---

## Configuration

Settings are resolved in priority order: **CLI args → YAML config file → environment variables**.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `GROQ_API_KEY` | Groq API key | — |
| `DASHSCOPE_API_KEY` | DashScope (Qwen) API key | — |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL | — |
| `OMNIPARSER_URL` | OmniParser server URL | `http://localhost:8000` |
| `WINDOWS_HOST_URL` | Windows host service URL | `http://localhost:5000` |
| `PADDLEOCR_URL` | Remote PaddleOCR API URL | `http://localhost:8001` |
| `GTA1_URL` | GTA1 server URL | `http://localhost:8002` |
| `CLOUD_ML_REGION` | Google Cloud region (Vertex AI) | `us-central1` |

### YAML Config File

```bash
python omnitool/gradio/ui/gradio/app.py --config-file config.yaml
```

Example `config.yaml`:
```yaml
omniparser_url: http://my-gpu-server:8000
windows_host_url: http://localhost:5000
run_folder: ./runs
```

---

## Supported Models & Agents

### Agent Types

| Agent | Description |
|---|---|
| **OmniAgent** | VLM + OmniParser SOM image → JSON action → coordinate resolution |
| **AnthropicAgent** | Anthropic computer-use API with OmniParser grounding |
| **GTAAgent** | GTA1 server-based agent |

### LLM Models

| Model ID | Providers |
|---|---|
| `gpt-4o` | openai |
| `gpt-4.1` | openai, azure |
| `gpt-5.1-codex` | openai, azure |
| `gpt-5.2` | openai, azure |
| `o1` | openai |
| `o3-mini` | openai |
| `claude-3-5-sonnet` | anthropic, bedrock, vertex |
| `qwen2.5vl` | dashscope |
| `deepseek-r1` | groq |
| `deepseek-V3.2` | azure |

### OCR Backends

| Backend | Mode | Notes |
|---|---|---|
| EasyOCR | Local CPU/GPU | Default |
| PaddleOCR | Local CPU/GPU | Higher throughput |
| PaddleOCR API | Remote GPU server | Set `PADDLEOCR_URL` |

---

## Testing

```bash
conda activate omni
pytest omnitool/gradio/tests/
```