import os

def delete_files_with_L(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Iterate through each file
    for filename in files:
        # Check if 'L' is in the filename
        if 'L' in filename:
            # Construct the full path for the file
            file_path = os.path.join(folder_path, filename)
            
            # Delete the file
            os.remove(file_path)
            print(f"Deleted {filename}")

# Replace 'folder_path' with the path to your folder containing the files
folder_path = r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\AllShifted'
delete_files_with_L(folder_path)
