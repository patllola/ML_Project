import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage import io, color, feature
from skimage.transform import resize
import psutil
import time
import joblib
import numpy as np
import os
import csv
from memory_profiler import profile
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import psutil
# import time
# import matplotlib.pyplot as plt

# Load the saved model
model = joblib.load("/Users/sandeepreddy/Desktop/Differentmodels/models5k/best_svm_model_split_data.joblib")

# model = joblib.load("/Users/sandeepreddy/Desktop/Differentmodels/models40k/logisticregression.pkl")


desktop_path = "/Users/sandeepreddy/Desktop/results"

input_path = '/Users/sandeepreddy/Desktop/testsample/'  # Replace with the path to your image or folder


def preprocess_image(image_path):
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
    img = resize(image, output_shape=(64,64))
    # Flatten and normalize test images
    img = img.reshape(1, -1)
    return img

@profile
def calling_decorators(image_path):
    img = preprocess_image(image_path)
    prediction =  model.predict(img)
    return prediction    

def predict_single_image(image_path):
    # start_time = time.time()
    # psutil.cpu_percent(1)
    # initial_memory_usage =psutil.virtual_memory()[2]
    # prediction = calling_decorators(image_path)
    img = preprocess_image(image_path)
    prediction =  model.predict(img)
    print(image_path,prediction)
    # final_cpu_usage = psutil.cpu_percent(1)
    # end_time = time.time()
    # final_memory_usage = psutil.virtual_memory()[4]
    # mem_diff=0
    
    return prediction[0]







def main(input_path, output_csv_path):
    if os.path.isfile(input_path):
        print("Input should be a directory containing images.")
        return
        
    elif os.path.isdir(input_path):
        num_iterations = 10
        batch_size = 100
        all_predictions = []
        all_runtimes = []
        total_cpu =[]
        

        for _ in range(num_iterations):
            batch_predictions = []
            start_time = time.time()
            cpu_usage_start = psutil.cpu_percent(1)
            print(f"CPU Usage (Before): {cpu_usage_start}%")
            
            for i, filename in enumerate(os.listdir(input_path)):
                if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                    image_path = os.path.join(input_path, filename)
                    predicted_class = predict_single_image(image_path)
                    batch_predictions.append((filename, predicted_class))
                    
                    
                    if (i + 1) % batch_size == 0:
                        break
            
            end_time = time.time()
            cpu_usage_end = psutil.cpu_percent(1)
            print(f"CPU Usage (after): {cpu_usage_end}%")
            batch_runtime = end_time - start_time
            print(f"Batch Runtime: {batch_runtime} seconds")
            total_cpu_usage= cpu_usage_end
            print(f"CPU Usage (total): {total_cpu_usage}%")
            
            all_predictions.append(batch_predictions)
            all_runtimes.append(batch_runtime)
            total_cpu.append(total_cpu_usage)

            print(f"Iteration {len(all_runtimes)}:")
            print("--------------------------------------")
        
            write_results_to_csv(output_csv_path, all_predictions, batch_runtime, total_cpu_usage)
        
    else:
        print("Invalid input path.")


def write_results_to_csv(output_csv_path, all_predictions,batch_runtime,total_cpu_usage):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Batch Index', 'Image Name', 'Predicted Class'])
        
        for i, batch_predictions in enumerate(all_predictions):
            for filename, predicted_class in batch_predictions:
                csv_writer.writerow([i+1, filename, predicted_class])
                
            csv_writer.writerow(["Total batch time:", batch_runtime])
            csv_writer.writerow(["Total cpu time:",total_cpu_usage])

output_csv_path = os.path.join(desktop_path, 'metrics_data_1dcnn_1version1.csv')
input_path = '/Users/sandeepreddy/Desktop/test/'
main(input_path, output_csv_path)


output_csv_path = os.path.join(desktop_path, 'metrics_data_svm.csv')
main(input_path,output_csv_path)