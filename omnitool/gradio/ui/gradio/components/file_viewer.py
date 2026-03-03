"""
File viewer UI components.
"""

from pathlib import Path

from omnitool.gradio.app import FileHandler


def render_file_list(files: list[Path]) -> str:
    """Render list of files as HTML.
    
    Args:
        files: List of file paths
        
    Returns:
        HTML string
    """
    if not files:
        return "<p>No files</p>"
    
    html_parts = ["<ul>"]
    for file_path in files:
        file_info = FileHandler.get_file_info(file_path)
        size_mb = file_info.get('size_mb', 0)
        html_parts.append(f"<li>{file_path.name} ({size_mb:.2f} MB)</li>")
    html_parts.append("</ul>")
    
    return "\n".join(html_parts)


def render_file_viewer(file_path: Path) -> str:
    """Render file viewer HTML.
    
    Args:
        file_path: Path to file
        
    Returns:
        HTML string for displaying file
    """
    return FileHandler.render_file_html(file_path)


__all__ = [
    "render_file_list",
    "render_file_viewer",
]
