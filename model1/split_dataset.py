import os
import shutil

source_directory = "../data/DHR_audio"

# Define target directories for each category
target_directories = {
    'H': '../data/H_audio',
    'D': '../data/D_audio',
    'R': '../data/R_audio',
}

# Iterate through each file in the source directory
for filename in os.listdir(source_directory):
    for key in target_directories.keys():
        if key in filename:
            src_file_path = os.path.join(source_directory, filename)
            dest_file_path = os.path.join(target_directories[key], filename)
            shutil.copy(src_file_path, dest_file_path)
            print(f"File {filename}")
            break  # Stop checking once the first match is found and file is copied

print("Files have been sorted and copied to their respective directories.")
