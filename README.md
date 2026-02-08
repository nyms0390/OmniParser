# OmniParser: Refactored (Experimental)

This repository is a refactored version of [Microsoft/OmniParser](https://github.com/microsoft/OmniParser) designed for cleaner code organization and easier integration into downstream enterprise applications.

**Note on Development:** The entirety of this refactoring was performed with the assistance of **GitHub Copilot**. 

**Disclaimer:** This version has only been tested within a specific internal environment and has not been validated for general public use. It should be considered experimental.

## Original Project References
- Project Page: https://microsoft.github.io/OmniParser/
- Paper: https://arxiv.org/abs/2408.00203
- Models: https://huggingface.co/microsoft/OmniParser-v2.0

## Key Changes and Improvements

### Architecture and Structure
- Reorganized into a modular structure with dedicated packages for utilities, Gradio UI, and the inference server.
- Standardized on absolute imports and PEP 8 compliance for better IDE support.
- Implemented lazy initialization for services to reduce startup overhead.

### PaddleOCR GPU Integration
- Added support for a GPU-accelerated PaddleOCR server backend.
- Optimized for low-latency processing, moving OCR tasks from CPU-bound to GPU-accelerated infrastructure.
- Retains fallback support for local EasyOCR.

### Enterprise LLM Support
- Integrated Azure OpenAI provider support to accommodate environments using managed identities or service principals instead of direct API keys.
- Abstracted LLM clients to allow seamless switching between Azure, OpenAI, and other providers via environment variables.

### Production Components
- FastAPI server for structured UI parsing requests.
- Optimized Gradio interface for multi-agent orchestration.

## Testing and Environment Constraints
- **No Virtualization:** This refactor discards "OmniBox" and other VM-based solutions as the target internal hardware does not support virtualization.
- **Internal Validation:** This code is functional in restricted corporate network environments. Because testing has been limited to internal infrastructure, users may encounter issues in different environments.