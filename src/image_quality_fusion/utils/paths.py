"""
Centralized path utilities for the Image Quality Fusion project.
Provides robust, portable path resolution that works regardless of:
- Where the project is cloned
- Which directory scripts are run from
- Package installation status
"""
import os
import sys
from pathlib import Path
from typing import Optional


def find_project_root(start_path: Optional[Path] = None) -> Path:
    """
    Find the project root directory by looking for key marker files.
    
    This approach is more robust than Path(__file__).parent.parent.parent
    because it searches upward until it finds the project root markers.
    
    Args:
        start_path: Starting path to search from (defaults to current file's directory)
        
    Returns:
        Path to the project root directory
        
    Raises:
        FileNotFoundError: If project root cannot be found
    """
    if start_path is None:
        start_path = Path(__file__).parent
    
    # Marker files/directories that indicate project root
    root_markers = [
        'src',
        'README.md',
        'pyproject.toml',
        '.git',
        'setup.py'
    ]
    
    current_path = start_path.resolve()
    
    # Search upward through parent directories
    for _ in range(10):  # Limit search depth to prevent infinite loops
        # Check if any root markers exist in current directory
        if any((current_path / marker).exists() for marker in root_markers):
            # Verify this looks like our project by checking for src/image_quality_fusion
            if (current_path / 'src' / 'image_quality_fusion').exists():
                return current_path
        
        # Move up one directory
        parent = current_path.parent
        if parent == current_path:  # Reached filesystem root
            break
        current_path = parent
    
    raise FileNotFoundError(
        f"Could not find project root starting from {start_path}. "
        f"Make sure you're running from within the image-quality-fusion project."
    )


def get_project_root() -> Path:
    """
    Get the project root directory.
    Cached for performance.
    """
    if not hasattr(get_project_root, '_cached_root'):
        get_project_root._cached_root = find_project_root()
    return get_project_root._cached_root


def get_src_dir() -> Path:
    """Get the src directory path."""
    return get_project_root() / 'src'


def get_scripts_dir() -> Path:
    """Get the scripts directory path."""
    return get_project_root() / 'scripts'


def get_datasets_dir() -> Path:
    """Get the datasets directory path."""
    return get_project_root() / 'datasets'


def get_outputs_dir() -> Path:
    """Get the outputs directory path."""
    return get_project_root() / 'outputs'


def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_project_root() / 'configs'


def get_docs_dir() -> Path:
    """Get the docs directory path."""
    return get_project_root() / 'docs'


def ensure_project_imports():
    """
    Ensure the project src directory is in Python path for imports.
    This replaces manual sys.path manipulation throughout the codebase.
    """
    src_path = str(get_src_dir())
    if src_path not in sys.path:
        sys.path.insert(0, src_path)


def resolve_path_from_project_root(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.
    
    Args:
        relative_path: Path relative to project root (e.g., 'datasets/demo/annotations.csv')
        
    Returns:
        Absolute Path object
    """
    return get_project_root() / relative_path


def validate_project_structure():
    """
    Validate that the project has the expected structure.
    Useful for debugging path issues.
    
    Returns:
        bool: True if structure is valid
        
    Raises:
        FileNotFoundError: If critical directories are missing
    """
    root = get_project_root()
    
    critical_paths = [
        root / 'src' / 'image_quality_fusion',
        root / 'scripts',
        root / 'README.md'
    ]
    
    missing_paths = [p for p in critical_paths if not p.exists()]
    
    if missing_paths:
        raise FileNotFoundError(
            f"Project structure validation failed. Missing: {missing_paths}"
        )
    
    return True


def get_relative_path_from_root(absolute_path: Path) -> str:
    """
    Get a path relative to the project root from an absolute path.
    
    Args:
        absolute_path: Absolute path to convert
        
    Returns:
        String path relative to project root
    """
    root = get_project_root()
    try:
        return str(absolute_path.relative_to(root))
    except ValueError:
        # Path is not relative to project root
        return str(absolute_path)


def safe_chdir_to_project_root():
    """
    Safely change working directory to project root.
    Returns the previous working directory for restoration.
    """
    original_cwd = Path.cwd()
    project_root = get_project_root()
    os.chdir(project_root)
    return original_cwd


# Convenience functions for common paths
def model_path(experiment_name: str = "fixed_run") -> Path:
    """Get path to a trained model."""
    return get_outputs_dir() / experiment_name / "model_best.pth"


def config_path(config_name: str = "default") -> Path:
    """Get path to a configuration file."""
    return get_configs_dir() / f"{config_name}.yaml"


def dataset_path(dataset_name: str = "demo") -> Path:
    """Get path to a dataset directory."""
    return get_datasets_dir() / dataset_name