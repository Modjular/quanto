
import h5py
import sys
import os

def print_structure(name, obj):
    """
    Recursively print the structure of the HDF5 file.
    """
    indent = "  " * name.count("/")
    base_name = os.path.basename(name)
    
    size_str = "-"
    if isinstance(obj, h5py.Dataset):
        try:
            size_bytes = obj.id.get_storage_size()
            size_str = f"{size_bytes:,}"
        except Exception:
            size_str = "?"
            
    # Pad left 40 chars
    prefix = size_str.ljust(8)
    
    if isinstance(obj, h5py.Group):
        print(f"{prefix} {indent}📂 {base_name}/")
        for key, val in obj.attrs.items():
            print(f"{' ' * 40} {indent}  - [Attr] {key}: {val}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix} {indent}📄 {base_name} {obj.shape} {obj.dtype}")
        for key, val in obj.attrs.items():
            print(f"{' ' * 40} {indent}  - [Attr] {key}: {val}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_ilp.py <path_to_ilp_file>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    print(f"Inspecting: {file_path}")
    print("=" * 40)
    
    try:
        with h5py.File(file_path, 'r') as f:
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
