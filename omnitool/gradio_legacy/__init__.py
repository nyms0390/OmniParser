"""Legacy Gradio application module - DEPRECATED.

.. deprecated:: 0.2.0
    This module is deprecated and will be removed in version 1.0.0.
    Use :mod:`omnitool.gradio` instead for all new development.

This module contains older implementations of the OmniParser Gradio interface.
While functional, it is no longer actively maintained and has been superseded
by the refactored :mod:`omnitool.gradio` module which provides:

- Better separation of concerns with 11-layer architecture
- Improved code organization and maintainability
- Enhanced type hints (100% coverage)
- Professional design patterns (Factory, Strategy, Singleton)
- Unified configuration management
- Comprehensive documentation

**Migration Path**:

Old (Deprecated)::

    from omnitool.gradio_legacy.app import create_app
    from omnitool.gradio_legacy.agent import AnthropicActor
    app = create_app()

New (Recommended)::

    from omnitool.gradio.ui.gradio.app import GradioApp
    from omnitool.gradio.core.agents import AnthropicAgent
    from omnitool.gradio.services import AppState
    app = GradioApp(settings)

**See Also**:
    - Migration guide: See MIGRATION_GUIDE.md in project root
    - Refactored module: :mod:`omnitool.gradio`
    - Architecture documentation: See PROJECT_ANALYSIS.md

**Timeline**:
    - v0.2.0 (current): Marked deprecated, warnings enabled
    - v0.3.0: Limited support, major refactoring recommended
    - v1.0.0: Module removed, migration required

"""

import warnings

warnings.warn(
    (
        "The 'omnitool.gradio_legacy' module is deprecated and will be removed "
        "in version 1.0.0. Please migrate to 'omnitool.gradio' for improved "
        "architecture, maintainability, and features. "
        "See MIGRATION_GUIDE.md for detailed migration steps."
    ),
    DeprecationWarning,
    stacklevel=2,
)

__deprecated__ = True
__migration_path__ = "omnitool.gradio"
__removal_version__ = "1.0.0"

__all__ = [
    "app",
    "app_new",
    "app_streamlit",
    "agent",
    "executor",
    "tools",
    "loop",
]
