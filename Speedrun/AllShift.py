import os
import csv

def find_offset_for_landmark4(input_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            try:
                index_4 = row.index('4')
                x4, y4 = int(row[index_4 + 1]), int(row[index_4 + 2])
                return 150 - x4, 150 - y4
            except ValueError:
                pass
    return None, None

def refocus_landmarks(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name)
            offset_x, offset_y = find_offset_for_landmark4(input_file_path)
            if offset_x is not None and offset_y is not None:
                with open(input_file_path, 'r') as infile, open(output_file_path, 'w', newline='') as outfile:
                    reader = csv.reader(infile)
                    writer = csv.writer(outfile)
                    for row in reader:
                        new_row = []
                        for i, val in enumerate(row):
                            if i % 3 == 1:  # x-coordinate
                                try:
                                    new_row.append(str(int(val) + offset_x))
                                except ValueError:
                                    new_row.append(val)
                            elif i % 3 == 2:  # y-coordinate
                                try:
                                    new_row.append(str(int(val) + offset_y))
                                except ValueError:
                                    new_row.append(val)
                            else:
                                new_row.append(val)
                        writer.writerow(new_row)

def main():
    input_folder = (r'C:\Users\danie\Desktop\Coding Spring 2024\Science-Fair\Speedrun\All') # Update this to your input folder containing CSV files
    output_folder = "output_folder"  # Update this to your output folder

    refocus_landmarks(input_folder, output_folder)

if __name__ == "__main__":
    main()
