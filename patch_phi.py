import os
import sys
import site
import shutil
from pathlib import Path

def find_phi_utils():
    # Check common installation locations
    possible_paths = [
        # pip install location
        Path(site.getsitepackages()[0]) / "phi" / "utils.py",
        # conda install location
        Path(sys.prefix) / "lib" / "python3.11" / "site-packages" / "phi" / "utils.py",
        # user install location
        Path.home() / ".local" / "lib" / "python3.11" / "site-packages" / "phi" / "utils.py"
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("Could not find phi/utils.py. Please ensure phi-ai is installed.")

def apply_patch():
    # Find utils.py
    utils_path = find_phi_utils()
    backup_path = utils_path.parent / "utils.py.backup"

    # Create backup
    print(f"Creating backup at {backup_path}")
    shutil.copy2(utils_path, backup_path)

    # Read existing file
    with open(utils_path, 'r') as f:
        content = f.read()

    # Add import if missing
    if 'import inspect' not in content:
        content = 'import inspect\n' + content

    # New get_method_sig implementation
    new_method = '''
def get_method_sig(method):
    """Get method signature information."""
    try:
        # Use getfullargspec instead of deprecated getargspec
        argspec = inspect.getfullargspec(method)
        # Convert to format expected by rest of phi library
        return {
            'args': argspec.args,
            'varargs': argspec.varargs,
            'keywords': argspec.varkw,
            'defaults': argspec.defaults or ()
        }
    except Exception:
        return {
            'args': [],
            'varargs': None, 
            'keywords': None,
            'defaults': ()
        }
'''

    # Replace old implementation
    import re
    pattern = r'def get_method_sig\([^)]*\):.*?(?=\n\S)'
    new_content = re.sub(pattern, new_method.strip(), content, flags=re.DOTALL)

    # Write updated file
    print(f"Updating {utils_path}")
    with open(utils_path, 'w') as f:
        f.write(new_content)

    print("Patch applied successfully!")
    print(f"Backup saved at: {backup_path}")

if __name__ == "__main__":
    try:
        apply_patch()
    except Exception as e:
        print(f"Error applying patch: {e}")
        sys.exit(1)