import pandas as pd
import json

# Define the file paths
input_file_path = 'test1.xlsx'  # Change this to your input file path
output_json_path = 'data.json'  # Change this to your desired output file path

def convert_excel_to_json(excel_file_path, json_file_path):
    # Read the Excel file
    df = pd.read_excel(excel_file_path)
    # Convert the DataFrame to a JSON string
    json_str = df.to_json(orient='records', indent=4)
    # Write the JSON string to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)
    print(f"Excel file '{excel_file_path}' has been converted to JSON and saved as '{json_file_path}'.")

def convert_csv_to_json(csv_file_path, json_file_path):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    # Convert the DataFrame to a JSON string
    json_str = df.to_json(orient='records', indent=4)
    # Write the JSON string to a file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_str)
    print(f"CSV file '{csv_file_path}' has been converted to JSON and saved as '{json_file_path}'.")

if __name__ == "__main__":
    # Specify the file type: 'excel' or 'csv'
    file_type = 'excel'  # Change this to 'csv' if you are converting a CSV file

    if file_type == 'excel':
        convert_excel_to_json(input_file_path, output_json_path)
    elif file_type == 'csv':
        convert_csv_to_json(input_file_path, output_json_path)
    else:
        print("Invalid file type. Please specify 'excel' or 'csv'.")
