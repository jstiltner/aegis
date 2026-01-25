#!/usr/bin/env python3
"""Script to rename agent_runtime imports to aegis."""

import os
import re
from pathlib import Path


def rename_imports_in_file(filepath: Path) -> bool:
    """Rename imports in a single file."""
    try:
        content = filepath.read_text(encoding='utf-8')
        original = content
        
        # Replace import statements
        content = re.sub(r'from agent_runtime\.', 'from aegis.', content)
        content = re.sub(r'import agent_runtime\.', 'import aegis.', content)
        content = re.sub(r'import agent_runtime\b', 'import aegis', content)
        content = re.sub(r'"agent_runtime\.', '"aegis.', content)
        content = re.sub(r"'agent_runtime\.", "'aegis.", content)
        
        if content != original:
            filepath.write_text(content, encoding='utf-8')
            print(f"Updated: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to rename all imports."""
    # Process src/aegis directory
    src_dir = Path("src/aegis")
    if src_dir.exists():
        for filepath in src_dir.rglob("*.py"):
            rename_imports_in_file(filepath)
    
    # Process tests directory
    tests_dir = Path("tests")
    if tests_dir.exists():
        for filepath in tests_dir.rglob("*.py"):
            rename_imports_in_file(filepath)
    
    # Process examples directory
    examples_dir = Path("examples")
    if examples_dir.exists():
        for filepath in examples_dir.rglob("*.py"):
            rename_imports_in_file(filepath)
    
    # Process docs directory
    docs_dir = Path("docs")
    if docs_dir.exists():
        for filepath in docs_dir.rglob("*.md"):
            rename_imports_in_file(filepath)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
