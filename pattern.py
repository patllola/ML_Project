import re

# Define a regular expression pattern to match "Mem usage" lines
pattern = r'\d+\s+(\d+\.\d+)\s+MiB'

# Open the file and read its contents
with open('/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/svm5k.txt', 'r') as file:
    content = file.read()

matches = re.findall(pattern, content)

mem_usage_values = [float(match) for match in matches]
# print(mem_usage_values)
# int_mem_usage_values = [int(value) for value in mem_usage_values]

# Iterate over the list of floating-point numbers
for i in range(2, len(mem_usage_values), 4):
    print(mem_usage_values[i])




