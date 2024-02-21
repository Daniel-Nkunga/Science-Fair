import os

def rename_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for filename in files:
        if filename.endswith('.csv'):
            # Remove all 'L's from the filename
            new_filename = filename.replace('L', '')
            
            # Construct the full path for both old and new filenames
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")

# Replace 'folder_path' with the path to your folder containing the files
folder_path = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\LipsBrows')
rename_files(folder_path)
