import os
import csv
import regex, re
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the regex pattern to extract values

# pattern = regex.compile(r'''\s*\d+\s*([\d.]+)\s*MiB\s*[\d,.]+\s*MiB\s*prediction\s*=\s*model\.predict\(img\)''', regex.VERBOSE)
pattern = r'\d+\s+(\d+\.\d+)\s+MiB'


# Function to extract values from a file using regex
def extract_values_from_file(file_path):
    print("Entered 2")
    values = []
    with open(file_path, 'r') as file:
        for line in file:
            match = regex.match(pattern, line)
            if match:
                value = match.group(1)
                values.append(value)
    return values

# Function to append values to an existing CSV file under a specific column
def append_to_csv(values, csv_file, column_name):
    with open(csv_file, 'r', newline='', encoding='UTF-8') as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames
        # headers=head[0].split('\t')
        print(headers)

    if column_name not in headers:
        print(f"Column '{column_name}' does not exist in the CSV file.")
        return

    with open(csv_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        for row in values:
            row_data = {column: '' for column in headers}
            row_data[column_name] = row
            writer.writerow(row_data)

# Define the event handler for file changes
class FileHandler(FileSystemEventHandler):
    print("entered 1")
    def on_created(self, event):
        if event.is_directory:
            print("it is a directory")
            return
        new_file = event.src_path
        if new_file.endswith('.txt'):
            print(f"New txt file detected: {new_file}")
            values = extract_values_from_file(new_file)
            if values:
                print("Extracted values:", values)
                append_to_csv(values, 'csv', 'Memory')
            else:
                print(values)

# Define the folder to watch
folder_to_watch = r'/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/filewatcher/'
csv="/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/5k/5k3/"
# Start the file watcher
if __name__ == "__main__":
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_to_watch, recursive=False)
    observer.start()

    print(f"Watching folder: {folder_to_watch}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()