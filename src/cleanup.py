# src/cleanup.py
"""
Cross-platform cleanup utility for checkpoints and cache.
Supports Windows, Linux, and macOS.
"""
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional


def _is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform.startswith('win')


def _is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform.startswith('darwin')


def _is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform.startswith('linux')


def get_platform_name() -> str:
    """Return platform name: 'Windows', 'Linux', or 'macOS'."""
    if _is_windows():
        return "Windows"
    elif _is_macos():
        return "macOS"
    elif _is_linux():
        return "Linux"
    else:
        return "Unknown"


def safe_remove_dir(path: str | Path) -> tuple[bool, Optional[str]]:
    """
    Safely remove a directory and all its contents (cross-platform).
    
    Args:
        path: Directory path to remove
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        path = Path(path)
        if not path.exists():
            return True, None
        
        if path.is_dir():
            shutil.rmtree(str(path), ignore_errors=True)
            
            # On Windows, sometimes files are still locked; retry with different approach
            if _is_windows() and path.exists():
                try:
                    import stat
                    for root, dirs, files in os.walk(str(path), topdown=False):
                        for name in files:
                            file_path = os.path.join(root, name)
                            try:
                                os.chmod(file_path, stat.S_IWRITE)
                                os.remove(file_path)
                            except Exception:
                                pass
                        for name in dirs:
                            dir_path = os.path.join(root, name)
                            try:
                                os.rmdir(dir_path)
                            except Exception:
                                pass
                    if path.exists():
                        os.rmdir(str(path))
                except Exception as e:
                    return False, str(e)
        else:
            os.remove(str(path))
        
        return True, None
    except Exception as e:
        return False, str(e)


def safe_remove_file(path: str | Path) -> tuple[bool, Optional[str]]:
    """
    Safely remove a single file (cross-platform).
    
    Args:
        path: File path to remove
        
    Returns:
        (success: bool, error_message: Optional[str])
    """
    try:
        path = Path(path)
        if path.exists() and path.is_file():
            if _is_windows():
                try:
                    import stat
                    os.chmod(str(path), stat.S_IWRITE)
                except Exception:
                    pass
            os.remove(str(path))
        return True, None
    except Exception as e:
        return False, str(e)


def get_cleanup_targets() -> List[str]:
    """
    Get list of common checkpoint and cache directories.
    
    Returns:
        List of directory paths that should be cleaned
    """
    targets = [
        'model/Bayesian',           # Main checkpoint location
        'model/Bayesian/logs',      # Training logs
        'model/Testing',            # Testing models
        'model/Validation',         # Validation models
        'runs',                     # Historical runs directory
        '__pycache__',              # Python cache (root)
        '.pytest_cache',            # Pytest cache
        '.cache',                   # Generic cache
    ]
    
    # Add recursive __pycache__ search (for nested packages)
    root_path = Path.cwd()
    for pycache_dir in root_path.rglob('__pycache__'):
        if str(pycache_dir) not in targets:
            targets.append(str(pycache_dir))
    
    return targets


def cleanup_checkpoints_and_cache(
    dry_run: bool = False,
    verbose: bool = True,
    keep_dirs: Optional[List[str]] = None
) -> dict:
    """
    Delete all checkpoints and cache directories (cross-platform).
    
    This function safely removes:
    - model/Bayesian and subfolders (checkpoints)
    - model/Testing and model/Validation
    - runs/ (historical run data)
    - __pycache__ directories (Python cache)
    - .pytest_cache and .cache directories
    
    Args:
        dry_run: If True, report what would be deleted without deleting
        verbose: If True, print progress messages
        keep_dirs: Optional list of directory names to skip deletion
        
    Returns:
        Dictionary with cleanup statistics:
        {
            'platform': str,
            'deleted_dirs': List[str],
            'failed_dirs': List[tuple[str, str]],
            'dry_run': bool,
            'total_deleted': int,
            'total_failed': int,
        }
    """
    if keep_dirs is None:
        keep_dirs = []
    
    stats = {
        'platform': get_platform_name(),
        'deleted_dirs': [],
        'failed_dirs': [],
        'dry_run': dry_run,
        'total_deleted': 0,
        'total_failed': 0,
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  CLEANUP: Checkpoints & Cache (Platform: {stats['platform']})")
        print(f"{'='*70}")
        mode = "DRY RUN" if dry_run else "EXECUTING"
        print(f"  Mode: {mode}")
        print(f"  {'='*70}\n")
    
    targets = get_cleanup_targets()
    
    for target in targets:
        # Skip if in keep_dirs list
        if any(keep in target for keep in keep_dirs):
            if verbose:
                print(f"  [SKIP] {target} (in keep list)")
            continue
        
        target_path = Path(target)
        
        if not target_path.exists():
            if verbose:
                print(f"  [OK]   {target} (not found)")
            continue
        
        if verbose:
            print(f"  [...]  Processing: {target}")
        
        if target_path.is_dir():
            if dry_run:
                if verbose:
                    print(f"    [DRY RUN] Would delete directory: {target}")
                stats['deleted_dirs'].append(target)
                stats['total_deleted'] += 1
            else:
                success, error = safe_remove_dir(target)
                if success:
                    if verbose:
                        print(f"    [OK]   Deleted directory: {target}")
                    stats['deleted_dirs'].append(target)
                    stats['total_deleted'] += 1
                else:
                    if verbose:
                        print(f"    [FAIL] Failed to delete: {target}")
                        print(f"      Error: {error}")
                    stats['failed_dirs'].append((target, error or "Unknown error"))
                    stats['total_failed'] += 1
        else:
            if dry_run:
                if verbose:
                    print(f"    [DRY RUN] Would delete file: {target}")
                stats['deleted_dirs'].append(target)
                stats['total_deleted'] += 1
            else:
                success, error = safe_remove_file(target)
                if success:
                    if verbose:
                        print(f"    [OK]   Deleted file: {target}")
                    stats['deleted_dirs'].append(target)
                    stats['total_deleted'] += 1
                else:
                    if verbose:
                        print(f"    [FAIL] Failed to delete: {target}")
                        print(f"      Error: {error}")
                    stats['failed_dirs'].append((target, error or "Unknown error"))
                    stats['total_failed'] += 1
    
    # Clear PyTorch/CUDA cache
    try:
        import torch
        torch.cuda.empty_cache()
        if verbose:
            print(f"  [OK]   Cleared PyTorch CUDA cache")
    except Exception as e:
        if verbose:
            print(f"  [---] Could not clear CUDA cache: {e}")
    
    # Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  CLEANUP SUMMARY")
        print(f"  {'='*70}")
        print(f"  Total Deleted:  {stats['total_deleted']}")
        print(f"  Total Failed:   {stats['total_failed']}")
        print(f"  Platform:       {stats['platform']}")
        print(f"{'='*70}\n")
    
    return stats


if __name__ == "__main__":
    # Test the cleanup function
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup checkpoints and cache')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no deletion)')
    parser.add_argument('--keep', type=str, default='', help='Comma-separated dirs to keep')
    
    args = parser.parse_args()
    keep_list = [k.strip() for k in args.keep.split(',')] if args.keep else []
    
    stats = cleanup_checkpoints_and_cache(
        dry_run=args.dry_run,
        verbose=True,
        keep_dirs=keep_list
    )
    
    sys.exit(0 if stats['total_failed'] == 0 else 1)
