import pandas as pd
import yaml
import os

<<<<<<< HEAD
def find_task_bounds_by_line_number(filepath, line_number):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    if not lines or line_number > len(lines) or line_number < 1:
        raise ValueError(f"Line number {line_number} is out of range for file {filepath}")

    target_line = lines[line_number - 1]
    target_indentation = len(target_line) - len(target_line.lstrip())

    start = line_number - 1
    while start > 0 and (len(lines[start - 1]) - len(lines[start - 1].lstrip())) >= target_indentation:
        start -= 1
    while start > 0 and (len(lines[start - 1].strip()) != 0 and (len(lines[start - 1]) - len(lines[start - 1].lstrip())) <= target_indentation):
        start -= 1

    end = line_number
    while end < len(lines) and (len(lines[end]) - len(lines[end].lstrip())) >= target_indentation:
        end += 1

    return start, end

=======
>>>>>>> c4dbdb406a4fd8cbd05f61098038c07b554ea71c
def print_yaml_section(filepath, start, end):
    with open(filepath, 'r') as file:
        lines = file.readlines()[start:end]
    filtered_lines = [line for i, line in enumerate(lines) if not (line.strip() == '---' and i != 0)]
    content_str = ''.join(filtered_lines)
    try:
        yaml_content = yaml.safe_load(content_str)
        print(yaml.dump(yaml_content, sort_keys=False))
    except yaml.YAMLError as e:
        print(f"Error parsing YAML content: {e}")

def find_and_print_task_by_line(filepath, line_number):
    start, end = find_task_bounds_by_line_number(filepath, line_number)
    print(f"Task containing line {line_number} (considering task name) spans from lines {start+1} to {end}:")
    print_yaml_section(filepath, start, end)

def process_tasks_from_csv(csv_file_path, filepath_column_name, line_number_column_name):
    df = pd.read_csv(csv_file_path)
    
    for index, row in df.iterrows():
        filepath = row[filepath_column_name]
        line_number = int(row[line_number_column_name])
        
        absolute_path = os.path.abspath(filepath)
        
        if not os.path.exists(absolute_path):
            print(f"File not found: {absolute_path}")
            continue
        
        try:
            print(f"Processing {absolute_path} at line {line_number}")
            find_and_print_task_by_line(absolute_path, line_number)
        except ValueError as e:
            print(f"Warning: {e}. Skipping this entry.")
            continue  # Skip to the next iteration in the loop


<<<<<<< HEAD
=======

>>>>>>> c4dbdb406a4fd8cbd05f61098038c07b554ea71c
csv_file_path = 'GLITCH-ansible.csv'  # The path to your CSV file
filepath_column_name = 'Filepath'  # The name of the column containing the file paths
line_number_column_name = 'Line_Number'  # The name of the column containing the line numbers

# Run the process
process_tasks_from_csv(csv_file_path, filepath_column_name, line_number_column_name)
