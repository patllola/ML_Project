import os

folder_path = '/Users/sandeepreddy/Desktop/Image_detection_CNN/Input_data/Negative/'  # Change this to the path of your 
counter = 1

for filename in os.listdir(folder_path):
    if filename.startswith('00' or '0' or '01' or '21' or '23'or '22' or '24' or '25' or '26' or '27' or '28' or '29'):

        new_filename = str(counter) + '_' + filename
        
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)
        print(new_filename)

        os.rename(old_filepath, new_filepath)
        
      
        counter += 1
