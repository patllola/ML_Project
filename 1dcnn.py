import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import psutil
import time
import matplotlib.pyplot as plt
import csv

# Load the saved model
model = tf.keras.models.load_model("/Users/sandeepreddy/Desktop/Differentmodels/models5k/cnn2D_image_classification_model.h5")


desktop_path = "/Users/sandeepreddy/Desktop/Images/Images"
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0  # Normalize pixel values to [0, 1]
    return img

def predict_single_image(image_path):
    start_time = time.time()
    psutil.cpu_percent(1)
    initial_memory_usage =psutil.virtual_memory()[2]
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    final_cpu_usage = psutil.cpu_percent(1)
    end_time = time.time()
    final_memory_usage = psutil.virtual_memory()[4]
    
    if prediction[0][0] > 0.5:
        return "Positive", end_time-start_time, final_memory_usage-initial_memory_usage, final_cpu_usage
    else:
        return "Negative", end_time-start_time, final_memory_usage-initial_memory_usage, final_cpu_usage


    

def predict_images_in_folder(folder_path):
    predictions = {}
    cpu_usage_list = []
    memory_usage_list = []
    runtime_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference  = predict_single_image(image_path)
            predictions[filename] = predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference
            
            cpu_usage_list.append(cpu_usage_during_inference)
            memory_usage_list.append(memory_usage_during_inference)
            runtime_list.append(runtime)
            
    return predictions, cpu_usage_list, memory_usage_list, runtime_list



def main(input_path, output_csv_path):
    if os.path.isfile(input_path):  # Check if the input is a file
          
        predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference  = predict_single_image(input_path)
        
        create_graphs(cpu_usage_during_inference, memory_usage_during_inference, runtime, input_path, output_csv_path)

        
        print(f"The predicted class for the image is: {predicted_class}")        
        print("Total Runtime of the program is:", runtime,"seconds")
        print("CPU Usage during the program is:", cpu_usage_during_inference,"%")
        print("Memory Usage during the program is:", memory_usage_during_inference,"KB")
        
        
       
        
    elif os.path.isdir(input_path): # Check if the input is a directory
        
        
        folder_predictions, cpu_usage_list, memory_usage_list, runtime_list = predict_images_in_folder(input_path)
        
        create_graphs(cpu_usage_list, memory_usage_list, runtime_list, folder_predictions.keys(), output_csv_path)

        
        for filename, predicted_class in folder_predictions.items(): 
            
            print(f"Image {filename}: {predicted_class[0]}\nTime: {predicted_class[1]} seconds\nMemory usage: {predicted_class[2]}\nCPU usage: {predicted_class[3]}%")
            print(f"Image {filename}: {predicted_class}")
            
            

            
    else:
        print("Invalid input path.")



def write_to_csv(output_csv_path, image_names, cpu_usage_list, memory_usage_list, runtime_list):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write header
        csv_writer.writerow(['Image Name', 'CPU Usage (%)', 'Memory Usage (KB)', 'Runtime (seconds)'])

        # Write data rows
        for i, (image_name, cpu, memory, runtime) in enumerate(zip(image_names, cpu_usage_list, memory_usage_list, runtime_list)):
            # if runtime<0 and memory<0 and cpu<0:
            #     runtime,memory,cpu=0,0,0
            # runtime = round(runtime, 5)
            # csv_writer.writerow([image_name, cpu, memory, runtime])
                
            csv_writer.writerow([image_name, cpu, memory, runtime])
            
            

def create_graphs(cpu_usage_list, memory_usage_list, runtime_list, image_names, output_csv_path):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']


    # Plot CPU Usage
    axs[0].plot(cpu_usage_list, marker='o', label='CPU Usage', color=colors[0])
    # axs[0].hist(cpu_usage_list, bins=20, color='skyblue', edgecolor='black', label='CPU Usage')
    
    # for i, (cpu, image_name) in enumerate(zip(cpu_usage_list, image_names)):
    #     axs[0].text(i, cpu, f'{cpu:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
    axs[0].set_title('Consolidated CPU Usage during Inference')
    axs[0].set_xlabel('Image Index')
    axs[0].set_ylabel('CPU Usage (%)')
    # axs[0].set_xticks(range(len(image_names)))
    # axs[0].set_xticklabels(image_names, rotation=45, ha="right")
    axs[0].legend()

    # Plot Memory Usage
    axs[1].plot(memory_usage_list, marker='o', label='Memory Usage', color=colors[1])
    # axs[1].hist(memory_usage_list, bins=20, color='orange', edgecolor='black', label='Memory Usage')

    # for i, (memory, image_name) in enumerate(zip(memory_usage_list, image_names)):
    #     axs[1].text(i, memory, f'{memory:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
    axs[1].set_title('Consolidated Memory Usage during Inference')
    axs[1].set_xlabel('Image Index')
    axs[1].set_ylabel('Memory Usage (KB)')
    # axs[1].set_xticks(range(len(image_names)))
    # axs[1].set_xticklabels(image_names, rotation=45, ha="right")
    axs[1].legend()

    # Plot Runtime
    axs[2].plot(runtime_list, marker='o', label='Runtime', color=colors[2])
    # axs[2].hist(runtime_list, bins=20, color='green', edgecolor='black', label='Runtime')

    # for i, (runtime, image_name) in enumerate(zip(runtime_list, image_names)):
    #     axs[2].text(i, runtime, f'{runtime:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
    axs[2].set_title('Consolidated Runtime of Inference')
    axs[2].set_xlabel('Image Index')
    axs[2].set_ylabel('Runtime (seconds)')
    # axs[2].set_xticks(range(len(image_names)))
    # axs[2].set_xticklabels(image_names, rotation=45, ha="right")
    axs[2].legend()
    
    #box plot
    data = [cpu_usage_list, memory_usage_list, runtime_list]
    boxplots = axs[3].boxplot(data, patch_artist=True, widths=0.6, showfliers=False)

    # Customizing the box plot and adding quartile annotations
    for i, bplot in enumerate(boxplots['boxes']):
        # Get the data for each box (CPU, Memory, Runtime)
        dataset = data[i]

        # Calculate quartiles
        quartiles = np.percentile(dataset, [25, 50, 75])

        # Annotate Quartiles
        axs[3].text(i+1.1, quartiles[0], f'Q1: {quartiles[0]:.2f}', va='center', ha='right', fontsize=10, color='blue')
        axs[3].text(i+1.1, quartiles[1], f'Q2: {quartiles[1]:.2f}', va='center', ha='right', fontsize=10, color='green')
        axs[3].text(i+1.1, quartiles[2], f'Q3: {quartiles[2]:.2f}', va='center', ha='right', fontsize=10, color='red')

    axs[3].set_xticklabels(['CPU Usage (%)', 'Memory Usage (KB)', 'Runtime (seconds)'], fontsize=12)
    axs[3].set_title('Box Plot of CPU, Memory, and Runtime', fontsize=14)
    axs[3].set_ylabel('Values', fontsize=12)
    axs[3].set_ylabel("box plot for different algorithms")
    
    write_to_csv(output_csv_path, image_names, cpu_usage_list, memory_usage_list, runtime_list)

    
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, 'local1D_CNN_metrics.png'),dpi=80)
    plt.show()

output_csv_path = os.path.join(desktop_path, 'metrics_data_1dcnn.csv')
input_path = '/Users/sandeepreddy/Desktop/testsample/'  # Replace with the path to your image or folder
main(input_path, output_csv_path)