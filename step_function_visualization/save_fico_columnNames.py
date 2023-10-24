import csv

# Define the input and output file paths
input_csv_file = 'fico_bin_first_5_rows.csv'  # Replace with your input CSV file path
output_csv_file = 'fico_bin_columnNames.csv'  # Replace with your desired output CSV file path

# Read the input CSV file
with open(input_csv_file, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    # Read the first row to get the column names
    column_names = next(reader)

# Write the column names to the output CSV file
with open(output_csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the column names as a single row
    writer.writerow(column_names)

print(f'Column names from {input_csv_file} have been exported to {output_csv_file}.')
