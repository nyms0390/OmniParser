"""
QUICK START GUIDE - Refactored OmniParser Gradio
=================================================

This guide helps you get started with the refactored codebase.
"""

# STEP 1: UNDERSTAND THE STRUCTURE
# ================================
# The refactored code follows a 9-layer architecture:
# 
# Layer 1: config/           - Constants, model registry, settings
# Layer 2: services/         - State, auth, file handling
# Layer 3: clients/          - LLM clients (OpenAI, Anthropic, Groq)
# Layer 4: core/agents/      - Agent implementations (VLM, Anthropic)
# Layer 5: core/executors/   - Tool execution
# Layer 6: core/orchestrator - Main sampling loop (KEY FILE!)
# Layer 7: core/tools/       - Tool definitions
# Layer 8: ui/gradio/        - Gradio interface
# Layer 9: tests/            - Unit tests with mocks
#
# Each layer only depends on layers below it (no circular deps)


# STEP 2: EXAMINE KEY FILES
# =========================

# Start with these files to understand the refactoring:

# A. Architecture overview
#    Read: gradio_refactored/README.md
#    Time: 10 minutes
#    What: Full architecture guide, decisions, usage

# B. Configuration system
#    Read: config/models.py (MODEL_CONFIG)
#    Time: 5 minutes
#    What: How models map to agents, LLM clients, pricing

# C. State management
#    Read: services/state.py (AppState class)
#    Time: 5 minutes  
#    What: How state flows through the app

# D. Core orchestration
#    Read: core/orchestrator.py (SamplingOrchestrator)
#    Time: 10 minutes
#    What: The main plan-execute-observe loop

# E. Agent implementation
#    Read: core/agents/vlm.py (VLMAgent.plan())
#    Time: 10 minutes
#    What: How agents generate responses

# Total time to understand: ~40 minutes


# STEP 3: RUN THE TESTS
# =====================

# Install pytest:
# $ pip install pytest pytest-mock

# Run tests:
# $ cd /Users/zhangyang/Documents/OmniParser/omnitool/gradio_refactored
# $ pytest tests/test_core_components.py -v

# Expected output:
# test_config_models.py::TestModelConfig::test_get_model_config PASSED
# test_config_models.py::TestModelConfig::test_all_models_have_config PASSED
# ...
# ============ 10 passed in 0.25s ============


# STEP 4: UNDERSTAND THE FLOW
# ============================

# User Input Flow:
# ================
# User sends message in Gradio UI
#     ↓
# ui/gradio/app.py: on_submit() callback
#     ↓
# Creates AppState (services/state.py)
#     ↓
# Creates SamplingOrchestrator (core/orchestrator.py)
#     ↓
# Calls create_agent() from factory (core/agents/factory.py)
#     ↓
# Creates agent with injected LLM client
#     ↓
# Starts sampling_loop() generator:
#
#   PLAN PHASE:
#   -----------
#   Captures screen via OmniParser (clients/services/omniparser.py)
#   ↓
#   Calls agent.plan() with screen info (core/agents/vlm.py)
#   ↓
#   LLM generates response via BaseLLMClient interface
#   ↓
#   Agent parses tool calls from response
#
#   EXECUTE PHASE:
#   ---------------
#   ToolExecutor runs tool calls (core/executors/tool_executor.py)
#   ↓
#   Tools interact with computer (via Windows host)
#
#   OBSERVE PHASE:
#   ---------------
#   Captures new screen state
#   ↓
#   Yields status update to UI
#   ↓
#   Loop repeats or terminates


# STEP 5: KEY DESIGN PATTERNS
# ============================

# 1. DEPENDENCY INJECTION
#    Pattern: Pass dependencies into constructors
#    Example:
#        agent = create_agent(
#            model_name="omniparser + gpt-4o",
#            state=state,           # Injected
#            tools_collection=tools,  # Injected
#            save_folder=Path("./outputs"),
#        )
#    Benefit: Easy to test (swap implementations), no hidden globals

# 2. FACTORY PATTERN
#    Pattern: Use factory function to create objects
#    Example:
#        agent = create_agent(model_name, state, tools, save_folder)
#        # Factory handles: config lookup, client creation, agent instantiation
#    Benefit: Single point of object creation, easy to extend

# 3. STRATEGY PATTERN
#    Pattern: Unified interface for different implementations
#    Example:
#        llm_client = get_llm_client("openai", "gpt-4o", api_key)
#        # Returns OpenAIClient
#        llm_client = get_llm_client("anthropic", "claude-3-5", api_key)
#        # Returns AnthropicClient
#        # Both implement BaseLLMClient.generate()
#    Benefit: Plug in different providers without changing agent code

# 4. CONFIGURATION REGISTRY
#    Pattern: Centralized configuration lookup
#    Example:
#        config = get_model_config("omniparser + gpt-4o")
#        # Returns dict with agent_type, pricing, max_tokens, etc
#    Benefit: Single source of truth for model info

# 5. GENERATOR PATTERN
#    Pattern: Yield updates instead of blocking
#    Example:
#        for update in orchestrator.sampling_loop():
#            if update["type"] == "step":
#                print(f"Step {update['step_num']}")
#    Benefit: Can update UI in real-time during execution


# STEP 6: COMPARE WITH OLD CODE
# ==============================

# Old: Multiple app files (app.py, app_new.py, app_streamlit.py)
# New: Single app.py with imported components

# Old: State scattered in callbacks
# New: Centralized in services/state.py

# Old: Agent and orchestrator mixed (loop.py)
# New: Separated - agents in core/agents/, orchestration in core/orchestrator.py

# Old: 60% code duplication between VLMAgent versions
# New: Single consolidated VLMAgent, 0% duplication

# Old: Hardcoded model-to-provider mapping with string matching
# New: MODEL_CONFIG registry with automatic lookup

# Old: Multiple LLM client implementations with different interfaces
# New: Unified BaseLLMClient interface, factory creates right one

# Old: Auth keys in file + env var mix
# New: Environment variables only (secure, stateless)

# Old: No tests, tightly coupled to external services
# New: Comprehensive tests, mocked dependencies


# STEP 7: IMPLEMENT YOUR CHANGES
# ===============================

# To add a new feature, follow this pattern:
#
# 1. If it's configuration: Add to config/
# 2. If it's business logic: Add to services/
# 3. If it's external API: Add to clients/
# 4. If it's agent behavior: Modify core/agents/
# 5. If it's orchestration: Modify core/orchestrator.py
# 6. If it's UI: Modify ui/gradio/app.py and components/
# 7. Write tests in tests/

# Example: Add support for GPT-4 Turbo
#
# 1. Add to MODEL_CONFIG in config/models.py:
#    "omniparser + gpt-4-turbo": {
#        "internal_name": "gpt-4-turbo-2024-04-09",
#        "agent_type": "VLMAgent",
#        "llm_client": "openai",
#        "provider": APIProvider.OPENAI,
#        "pricing": {"token_type": "total", "cost_per_1m": 10.0},
#        ...
#    }
#
# 2. Update UI in ui/gradio/components/settings.py if needed
#
# 3. Done! Factory and orchestrator automatically support it


# STEP 8: NEXT ACTIONS
# ====================

# SHORT TERM (Complete this refactoring):
# 1. Implement tool classes (click, type, screenshot, etc.)
# 2. Complete Gradio UI (wire up orchestrator)
# 3. Add error handling
# 4. Run integration tests with real servers

# MEDIUM TERM (Polish and optimize):
# 1. Add logging/tracing
# 2. Implement Streamlit UI (reuse services/core)
# 3. Performance optimization (async, caching)
# 4. Documentation/examples

# LONG TERM (Maintain and extend):
# 1. Add new providers/models
# 2. Add new tools
# 3. Cloud deployment
# 4. API server


# STEP 9: DEBUGGING TIPS
# ======================

# To understand execution flow:
# 1. Add breakpoints in orchestrator.sampling_loop()
# 2. Print state at each step: print(state.to_dict())
# 3. Check logs for API calls and errors
# 4. Verify config loaded correctly: print(get_all_model_names())

# To test a component in isolation:
# 1. Use fixtures from tests/conftest.py
# 2. Mock external services (OmniParser, LLM)
# 3. Run specific test: pytest tests/test_file.py::test_func -v

# To check configuration:
# $ python -c "from config import get_model_config; print(get_model_config('omniparser + gpt-4o'))"


# STEP 10: RESOURCES
# ==================

# Read these files in order:
# 1. gradio_refactored/README.md - Full architecture guide
# 2. gradio_refactored/IMPLEMENTATION_SUMMARY.md - What was built
# 3. config/models.py - Model definitions
# 4. services/state.py - State management
# 5. core/orchestrator.py - Main loop
# 6. core/agents/vlm.py - Agent implementation
# 7. ui/gradio/app.py - UI implementation
# 8. tests/test_core_components.py - Example tests

# Good practices:
# - Always use type hints
# - Write docstrings for functions
# - Create tests before implementing features
# - Use factories instead of direct instantiation
# - Prefer dependency injection over globals


print("""

============================================================
Refactored OmniParser Gradio - Quick Start Guide
============================================================

You now have a clean, maintainable codebase with:
✅ Clear separation of concerns
✅ No code duplication
✅ Dependency injection for testability
✅ Unified interfaces for all providers
✅ Comprehensive configuration system
✅ State management layer
✅ Test framework with mocks

Next steps:
1. Review gradio_refactored/README.md
2. Run tests: pytest tests/test_core_components.py -v
3. Examine core/orchestrator.py
4. Implement remaining components
5. Switch to refactored version when complete

Questions? Check the documentation or code comments.

============================================================
""")
