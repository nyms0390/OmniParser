"""
File handling service for uploads, storage, and file operations.
"""

import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class FileHandler:
    """Handles file uploads, storage, and management."""
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        '.txt', '.md',  # Text files
        '.pdf',  # Documents
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp',  # Images
        '.mp4', '.avi', '.mov',  # Videos
    }
    
    # File type categories
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
    DOCUMENT_EXTENSIONS = {'.txt', '.md', '.pdf'}
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov'}
    
    @staticmethod
    def is_allowed_file(filename: str) -> bool:
        """Check if file type is allowed.
        
        Args:
            filename: Name of the file
            
        Returns:
            True if file type is allowed
        """
        _, ext = os.path.splitext(filename)
        return ext.lower() in FileHandler.ALLOWED_EXTENSIONS
    
    @staticmethod
    def get_file_category(filename: str) -> str:
        """Get category of file.
        
        Args:
            filename: Name of the file
            
        Returns:
            Category: 'image', 'document', 'video', or 'unknown'
        """
        _, ext = os.path.splitext(filename)
        ext = ext.lower()
        
        if ext in FileHandler.IMAGE_EXTENSIONS:
            return "image"
        elif ext in FileHandler.DOCUMENT_EXTENSIONS:
            return "document"
        elif ext in FileHandler.VIDEO_EXTENSIONS:
            return "video"
        else:
            return "unknown"
    
    @staticmethod
    def upload_file(source_path: str, destination_folder: Path, original_filename: str) -> Tuple[bool, str, Optional[Path]]:
        """Upload/copy file to destination folder.
        
        Args:
            source_path: Source file path
            destination_folder: Destination folder path
            original_filename: Original filename to preserve
            
        Returns:
            (success, message, destination_path)
        """
        # Validate source file exists
        if not os.path.isfile(source_path):
            return False, f"Source file not found: {source_path}", None
        
        # Check if file type is allowed
        if not FileHandler.is_allowed_file(original_filename):
            return False, f"File type not allowed: {original_filename}", None
        
        # Create destination directory if needed
        destination_folder.mkdir(parents=True, exist_ok=True)
        
        # Create destination path
        destination_path = destination_folder / original_filename
        
        try:
            # Copy file
            with open(source_path, 'rb') as src:
                with open(destination_path, 'wb') as dst:
                    dst.write(src.read())
            
            return True, f"File uploaded: {original_filename}", destination_path
        except Exception as e:
            return False, f"Failed to upload file: {str(e)}", None
    
    @staticmethod
    def detect_new_files(folder: Path) -> List[Path]:
        """Detect new files in folder.
        
        Args:
            folder: Folder to scan
            
        Returns:
            List of file paths
        """
        if not folder.exists():
            return []
        
        files = []
        for item in folder.rglob('*'):
            if item.is_file():
                files.append(item)
        
        return sorted(files)
    
    @staticmethod
    def clear_session_files(session_folder: Path) -> Tuple[bool, str]:
        """Clear all files in session folder.
        
        Args:
            session_folder: Session folder to clear
            
        Returns:
            (success, message)
        """
        try:
            if not session_folder.exists():
                return True, "Session folder does not exist"
            
            # Remove all files in folder
            for item in session_folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir() and item.name != '.':  # Don't delete subdirectories
                    pass
            
            return True, "Session files cleared"
        except Exception as e:
            return False, f"Failed to clear session files: {str(e)}"
    
    @staticmethod
    def get_file_size_mb(file_path: Path) -> float:
        """Get file size in MB.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in MB
        """
        if not file_path.exists():
            return 0.0
        return file_path.stat().st_size / (1024 * 1024)
    
    @staticmethod
    def get_file_info(file_path: Path) -> dict:
        """Get detailed file information.
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with file info
        """
        if not file_path.exists():
            return {}
        
        stat = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return {
            "name": file_path.name,
            "path": str(file_path),
            "size_mb": stat.st_size / (1024 * 1024),
            "size_bytes": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "mime_type": mime_type or "unknown",
            "category": FileHandler.get_file_category(file_path.name),
        }
    
    @staticmethod
    def render_file_html(file_path: Path) -> str:
        """Render HTML preview for file based on type.
        
        Args:
            file_path: Path to file to render
            
        Returns:
            HTML string
        """
        if not file_path.exists():
            return "<p>File not found</p>"
        
        file_info = FileHandler.get_file_info(file_path)
        category = file_info.get("category", "unknown")
        name = file_info.get("name", "Unknown")
        
        html_parts = [f"<h3>File: {name}</h3>"]
        
        try:
            if category == "image":
                # Render image
                html_parts.append(f'<img src="file://{file_path}" style="max-width:100%; max-height:600px;"/>')
            
            elif category == "document":
                if str(file_path).endswith('.txt') or str(file_path).endswith('.md'):
                    # Render text content
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    html_parts.append(f"<pre>{content[:5000]}...</pre>" if len(content) > 5000 else f"<pre>{content}</pre>")
                else:
                    html_parts.append(f"<p>PDF files cannot be displayed inline. Download to view: {name}</p>")
            
            elif category == "video":
                html_parts.append('<video width="320" height="240" controls>')
                html_parts.append(f'  <source src="file://{file_path}">')
                html_parts.append('</video>')
            
            else:
                html_parts.append(f"<p>Cannot preview file type: {category}</p>")
        
        except Exception as e:
            html_parts.append(f"<p>Error rendering file: {str(e)}</p>")
        
        return "\n".join(html_parts)
