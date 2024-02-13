import csv

# Function to parse the output file and extract values
def parse_output_file(output_file):
    with open(output_file, 'r') as file:
        values = []
        for line in file:
            stripped_line = line.strip()
            # print(stripped_line)
            values.append(stripped_line)
            # print(values)

        # values = [line.strip() for line in file]
    return values

# # Function to replace zero values in CSV with parsed values
def replace_zero_values(csv_file, column_index, values):
    # Open CSV file for reading and writing
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        # print(rows)

    # Replace zero values in specified column with parsed values
    for row in rows:
        if row[column_index] == '0':  # Assuming zero values are represented as strings
            try:
                replacement_value = values.pop(0)  # Get the next parsed value
                row[column_index] = replacement_value
            except IndexError:
                break  # Stop if there are no more parsed values

    # # Write modified rows back to CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# Specify the paths to the output file and CSV file
output_file = '/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/patternlogistic10k.txt'
csv_file = '/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/10k/10k1/metrics_data_logistic.csv'

# Parse the output file to obtain values
parsed_values = parse_output_file(output_file)

# Replace zero values in the specified column of the CSV file with parsed values
column_index = 2  # Specify the index of the column containing zero values (0-based index)
replace_zero_values(csv_file, column_index, parsed_values)
