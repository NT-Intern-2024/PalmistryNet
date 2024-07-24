import os
import sys

def convert_filenames_to_lowercase(directory):
    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            # Get the full path of the file
            file_path = os.path.join(dirpath, filename)
            
            # Convert filename to lowercase
            new_filename = filename.lower()
            
            # Check if the new filename is different from the original one
            if new_filename != filename:
                # Rename the file to lowercase
                os.rename(file_path, os.path.join(dirpath, new_filename))
                print(f"Renamed: {filename} -> {new_filename}")

# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python script.py <directory_path>")
#         sys.exit(1)
        
#     directory_path = sys.argv[1]
#     if not os.path.isdir(directory_path):
#         print("Invalid directory path.")
#         sys.exit(1)
        
#     convert_filenames_to_lowercase(directory_path)
#     print("All filenames converted to lowercase.")
