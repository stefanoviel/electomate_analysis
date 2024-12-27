import os

def read_file_content(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading {file_path}: {str(e)}\n"

def combine_python_files():
    # List of files to process
    files = [
        'src/visualization.py',
        'src/query_for_party.py',
        'src/main.py',
        'src/gpt_interface.py',
        'src/download_pdf.py',
        'src/data_processing.py',
        'src/create_index.py',
        'src/config.py'
    ]
    
    # Create output string
    output = "# Combined Python Scripts\n\n"
    
    # Process each file
    for file_path in files:
        output += f"\n{'='*80}\n"
        output += f"# File: {file_path}\n"
        output += f"{'='*80}\n\n"
        output += read_file_content(file_path)
        output += "\n\n"
    
    # Write to output file
    with open('combined_scripts.txt', 'w', encoding='utf-8') as f:
        f.write(output)

if __name__ == "__main__":
    combine_python_files()
    print("Files have been combined into 'combined_scripts.txt'")