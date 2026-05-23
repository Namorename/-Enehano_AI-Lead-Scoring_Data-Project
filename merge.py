import os
import glob

def combine_python_files(directory=".", output_filename="combined_output.py"):
    # Get the name of the current script to avoid including it in the output
    current_script = os.path.basename(__file__)
    
    # Create a search pattern for all Python files in the directory
    search_pattern = os.path.join(directory, "*.py")
    py_files = glob.glob(search_pattern)
    
    # Open the output file in write mode
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for file_path in py_files:
            filename = os.path.basename(file_path)
            
            # Skip the output file itself and the script running the code
            if filename == output_filename or filename == current_script:
                continue
                
            # Write the header separator for each file
            outfile.write(f"\n# {'='*50}\n")
            outfile.write(f"# HEADER: {filename}\n")
            outfile.write(f"# {'='*50}\n\n")
            
            # Read the content of the current python file and append it
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    outfile.write("\n")
            except Exception as e:
                # Log an error if a file cannot be read
                print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    # Define the target directory ('.' represents the current folder)
    target_dir = ""
    output_file = "combined_output.py"
    
    # Run the function
    combine_python_files(target_dir, output_file)
    print(f"Success! All files combined into '{output_file}'.")