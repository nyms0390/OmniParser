# OmniParser Gradio Refactored - Architecture & Implementation Guide

This directory contains the **refactored OmniParser Gradio application** built according to the comprehensive architecture plan. The refactoring prioritizes:

- **Clear separation of concerns**: Config → Services → Clients → Core → UI
- **Dependency injection**: All components receive dependencies via constructor
- **No code duplication**: Shared logic extracted to services and core modules
- **Environment-based configuration**: CLI args > YAML > env vars
- **Comprehensive testing**: Mocked external services, unit test coverage
- **Type hints**: Full type annotations for IDE support and documentation

## Directory Structure

```
gradio_refactored/
├── config/                          # Application configuration (no dependencies)
│   ├── __init__.py
│   ├── constants.py                 # Constants (timeouts, limits, etc)
│   ├── models.py                    # MODEL_CONFIG: model→agent/pricing mapping
│   └── settings.py                  # Settings loader: CLI > YAML > env
│
├── services/                        # Business logic (depends on config only)
│   ├── __init__.py
│   ├── state.py                     # Session/chat/auth/file/agent state
│   ├── auth.py                      # API key validation (env vars only)
│   └── file_handler.py              # File upload/detection/cleanup
│
├── clients/                         # External service clients
│   ├── __init__.py
│   ├── base.py                      # BaseLLMClient abstract interface
│   ├── llm/
│   │   ├── __init__.py              # get_llm_client factory
│   │   ├── openai.py                # OpenAI + Qwen (DashScope compatible)
│   │   ├── groq.py                  # Groq (R1 only, locked per requirements)
│   │   └── anthropic.py             # Anthropic + Bedrock + Vertex
│   └── services/
│       └── omniparser.py            # OmniParser HTTP client
│
├── core/                            # Core agent/execution logic
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py              # Agent factory
│   │   ├── base.py                  # BaseAgent abstract class
│   │   ├── vlm.py                   # VLMAgent (consolidated + no duplication)
│   │   ├── anthropic.py             # AnthropicAgent (Claude)
│   │   └── factory.py               # create_agent with DI
│   ├── executors/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseExecutor abstract
│   │   └── tool_executor.py         # ToolExecutor
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseTool, ToolResult, ToolCollection
│   │   └── collection.py            # ToolCollection manager
│   └── orchestrator.py              # **ALL orchestration logic here**
│                                    # Plan-execute-observe loop
│                                    # Ledger updates, trajectory saving
│
├── ui/
│   ├── gradio/
│   │   ├── __init__.py
│   │   ├── app.py                   # **MONOLITHIC app** with helper functions
│   │   └── components/              # Modular helper components
│   │       ├── __init__.py
│   │       ├── chat.py              # Chat formatting
│   │       ├── file_viewer.py       # File upload/viewing
│   │       └── settings.py          # Settings UI
│   └── streamlit/
│       ├── __init__.py
│       └── app.py                   # (To be refactored, reuses services/core)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                  # Pytest fixtures with mocks
    └── test_core_components.py      # Sample unit tests
```

## Architecture Decisions (Locked)

### 1. **Orchestration Location**
- ✅ **All orchestration in `core/orchestrator.py`**
- Agents are stateless: only plan based on input
- Orchestrator manages plan-execute-observe loop
- Orchestrator handles orchestrated variant logic (ledger, trajectory)

### 2. **UI Architecture**
- ✅ **Monolithic `ui/gradio/app.py`** with helper functions
- Components in `ui/gradio/components/` for modular callbacks
- Streamlit reuses same services/config/core layers

### 3. **State Management**
- ✅ **`services/state.py`** manages session lifecycle and file paths
- Session ID = timestamp-based unique identifier
- Per-session run folder created in `<run_folder>/<YYYYMMDD_HHMMSS>/`
- No persistent session storage (simple mode)

### 4. **Cost Calculation**
- ✅ **Cost formula stays in agents** (varies by provider)
- **Pricing metadata in `config/models.py`**
  - OpenAI/Groq/Qwen: `token_type: "total"`, unified token pricing
  - Anthropic: `token_type: "separate"`, input/output pricing
- Each agent's `_calculate_cost()` method implements provider logic

### 5. **Configuration Priority**
- ✅ **CLI args > YAML file > environment variables**
- `config/settings.py` implements priority order
- No multiple profiles (single configuration per launch)

### 6. **Authentication**
- ✅ **API keys from environment variables only**
- No file storage (secure by design)
- Providers map to env vars: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.

### 7. **APIProvider Enum**
- ✅ **Extended to include all providers**
  ```python
  class APIProvider(StrEnum):
      OPENAI = "openai"
      ANTHROPIC = "anthropic"
      BEDROCK = "bedrock"
      VERTEX = "vertex"
      GROQ = "groq"
      DASHSCOPE = "dashscope"
  ```

### 8. **Groq Model Lock**
- ✅ **Groq locked to R1 model** (`deepseek-r1-distill-llama-70b`)
- Future extensibility: can parameterize when other models needed

### 9. **Testing Strategy**
- ✅ **Comprehensive mocks for external services**
  - MockOmniParserClient, MockLLMClients (all providers)
  - MockWindowsHost for future tool tests
- **No integration tests in this layer** (require running servers)

## How to Use This Refactored Code

### 1. Installation

```bash
# Install required packages
pip install openai anthropic groq gradio pyyaml

# For development/testing
pip install pytest pytest-mock
```

### 2. Configuration

**Option A: Environment Variables**
```bash
export OPENAI_API_KEY="sk-..."
export OMNIPARSER_URL="http://localhost:8000"
export WINDOWS_HOST_URL="http://localhost:8006"
```

**Option B: YAML Config File**
```yaml
# config.yaml
openai_api_key: "sk-..."
anthropic_api_key: "sk-ant-..."
omniparser_url: "http://localhost:8000"
windows_host_url: "http://localhost:8006"
run_folder: "./runs"
```

**Option C: CLI Arguments**
```bash
python ui/gradio/app.py \
  --run_folder ./runs \
  --omniparser_server_url http://localhost:8000 \
  --windows_host_url http://localhost:8006 \
  --config-file config.yaml
```

### 3. Running Gradio App

```bash
# Basic
python ui/gradio/app.py

# With custom config
python ui/gradio/app.py --config-file ./config.yaml

# With CLI args
python ui/gradio/app.py --run_folder /data/runs --omniparser_server_url http://192.168.1.100:8000
```

### 4. Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_core_components.py -v

# Exclude slow tests
pytest tests/ -v -m "not slow"

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Model Configuration Reference

All models defined in `config/models.py`:

| Model | Agent | Client | Provider | Pricing |
|-------|-------|--------|----------|---------|
| omniparser + gpt-4o | VLMAgent | openai | openai | $2.50/1M tokens |
| omniparser + o1 | VLMAgent | openai | openai | $15.00/1M tokens |
| omniparser + o3-mini | VLMAgent | openai | openai | $1.10/1M tokens |
| omniparser + R1 | VLMAgent | groq | groq | $0.99/1M tokens |
| omniparser + qwen2.5vl | VLMAgent | openai | dashscope | $2.20/1M tokens |
| claude-3-5-sonnet | AnthropicAgent | anthropic | anthropic/bedrock/vertex | Input: $3.00, Output: $15.00 per 1M |

**Orchestrated variants** (same models + "-orchestrated" suffix):
- Use same agent/client but trigger orchestration logic in `core/orchestrator.py`
- Trajectory saved to JSON for audit trail
- Ledger updated with each step

## Key Design Patterns

### 1. Factory Pattern (Model → Agent)

```python
from core import create_agent

agent = create_agent(
    model_name="omniparser + gpt-4o",
    state=state,
    tools_collection=tools,
    save_folder=Path("./outputs"),
)
```

### 2. Dependency Injection

```python
# Agent receives all dependencies in constructor
agent = VLMAgent(
    model_name="omniparser + gpt-4o",
    llm_client=llm_client,  # Injected
    state=state,            # Injected
    tools_collection=tools, # Injected
    save_folder=save_folder,
)
```

### 3. Configuration Inheritance

```python
# Get settings with priority chain
settings = get_settings(args)
# Will use: CLI args → YAML → env vars → defaults

# Get model config
config = get_model_config("omniparser + gpt-4o")
# Returns dict with pricing, max_tokens, temperature, etc
```

### 4. Service Layer

```python
from services import AppState, FileHandler, validate_api_key

# State management
state = AppState(run_folder=Path("./runs"))
state.chat.add_message("user", "Hello")
state.files.add_file(Path("file.txt"))

# File operations
FileHandler.upload_file(src, dest_folder, "filename.txt")
FileHandler.detect_new_files(folder)

# Auth validation
is_valid, error = validate_api_key(APIProvider.OPENAI)
```

## Migration from Old Code

### Old Structure → New Structure

| Old | New | Location |
|-----|-----|----------|
| app.py, app_new.py, app_streamlit.py | app.py + components/ | ui/gradio/ |
| anthropic_agent.py, vlm_agent.py, vlm_agent_with_orchestrator.py | vlm.py, anthropic.py | core/agents/ |
| loop.py | orchestrator.py | core/ |
| agent/llm_utils/*.py | clients/llm/ | clients/ |
| anthropic_executor.py | tool_executor.py | core/executors/ |
| tools/* | tools/ | core/ |
| (scattered constants) | constants.py | config/ |
| (auth logic in apps) | auth.py | services/ |
| (file handling in apps) | file_handler.py | services/ |
| (state in apps) | state.py | services/ |

### Breaking Changes
- ❗ All agents require DI (no global state, no direct instantiation)
- ❗ LLM clients have unified `generate()` interface
- ❗ No hardcoded `OUTPUT_DIR` - always use `save_folder` parameter
- ❗ Config must be explicitly loaded with `get_settings(args)`
- ❗ Orchestration removed from agents - use `SamplingOrchestrator`

## Next Steps for Completion

1. **Complete Gradio app.py** - Add tool implementations (click, type, etc)
2. **Implement Streamlit app** - Reuse services/config/core layers
3. **Add more test coverage** - Tool execution, orchestrator loop, edge cases
4. **Documentation** - Docstrings, type hints, examples
5. **Integration tests** - With real omniparser server and Windows host
6. **Performance optimization** - Token/cost tracking, caching, async improvements
7. **Error handling** - Comprehensive exception handling and recovery
8. **Logging** - Structured logging for debugging and auditing

## Contact & Questions

For questions about the refactoring architecture, refer to the decision log at top of this file or check inline code comments.
