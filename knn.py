<<<<<<< HEAD
=======
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import StandardScaler
# from skimage import io, color, feature
# from skimage.transform import resize
# import psutil
# import time
# import joblib
# import os
# import csv
# import numpy as np

# # Load the saved model
# # model = tf.keras.models.load_model("/root/inferences/modelfile/knn_model.joblib")
# model = joblib.load("/Users/sandeepreddy/Desktop/Differentmodels/models5k/knn_model.joblib")


# desktop_path = "/Users/sandeepreddy/Desktop/Images"

# def preprocess_image(image_path):
    
#     image = io.imread(image_path)
#     if len(image.shape) == 3 and image.shape[-1] == 4:
#         image = image[:, :, :3]
#     img = resize(image, output_shape=(227,227))
#     img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
#     img_gray = color.rgb2gray(img)
#     hog_features = feature.hog(img_gray, block_norm='L2-Hys', pixels_per_cell=(16, 16))
#     img = hog_features.reshape(1, -1)
    
#     return img

# def predict_single_image(image_path):
#     start_time = time.time()
#     psutil.cpu_percent(interval=None)
#     initial_memory_usage =psutil.virtual_memory().used
#     img = preprocess_image(image_path)
#     prediction = model.predict(img)
#     final_cpu_usage = psutil.cpu_percent(interval=None)
#     end_time = time.time()
#     final_memory_usage = psutil.virtual_memory().used
#     print(image_path,prediction)
    
#     return prediction[0], end_time-start_time, final_memory_usage-initial_memory_usage, final_cpu_usage


    

# def predict_images_in_folder(folder_path):
#     predictions = {}
#     cpu_usage_list = []
#     memory_usage_list = []
#     runtime_list = []
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
#             image_path = os.path.join(folder_path, filename)
#             predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference  = predict_single_image(image_path)
#             predictions[filename] = predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference
            
#             cpu_usage_list.append(cpu_usage_during_inference)
#             memory_usage_list.append(memory_usage_during_inference)
#             runtime_list.append(runtime)
            
#     return predictions, cpu_usage_list, memory_usage_list, runtime_list



# def main(input_path, output_csv_path):
#     if os.path.isfile(input_path):  # Check if the input is a file
          
#         predicted_class, runtime, memory_usage_during_inference, cpu_usage_during_inference  = predict_single_image(input_path)
        
#         create_graphs(cpu_usage_during_inference, memory_usage_during_inference, runtime, input_path)

        
#         print(f"The predicted class for the image is: {predicted_class}")        
#         print("Total Runtime of the program is:", runtime,"seconds")
#         print("CPU Usage during the program is:", cpu_usage_during_inference,"%")
#         print("Memory Usage during the program is:", memory_usage_during_inference,"KB")
        
        
       
        
#     elif os.path.isdir(input_path): # Check if the input is a directory
        
        
#         folder_predictions, cpu_usage_list, memory_usage_list, runtime_list = predict_images_in_folder(input_path)
        
#         create_graphs(cpu_usage_list, memory_usage_list, runtime_list, folder_predictions.keys(), output_csv_path)

        
#         for filename, predicted_class in folder_predictions.items(): 
            
#             print(f"Image {filename}: {predicted_class[0]}\nTime: {predicted_class[1]} seconds\nMemory usage: {predicted_class[2]}\nCPU usage: {predicted_class[3]}%")
#             print(f"Image {filename}: {predicted_class}")
            
            

            
#     else:
#         print("Invalid input path.")


# def write_to_csv(output_csv_path, image_names, cpu_usage_list, memory_usage_list, runtime_list):
#     with open(output_csv_path, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)

#         # Write header
#         csv_writer.writerow(['Image Name', 'CPU Usage (%)', 'Memory Usage (KB)', 'Runtime (seconds)'])

#         # Write data rows
#         for i, (image_name, cpu, memory, runtime) in enumerate(zip(image_names, cpu_usage_list, memory_usage_list, runtime_list)):
#             csv_writer.writerow([image_name, cpu, memory, runtime])

# def create_graphs(cpu_usage_list, memory_usage_list, runtime_list, image_names, output_csv_path):
#     fig, axs = plt.subplots(4, 1, figsize=(10, 20))

#     # Plot CPU Usage
#     axs[0].plot(cpu_usage_list, marker='o', label='CPU Usage')
#     for i, (cpu, image_name) in enumerate(zip(cpu_usage_list, image_names)):
#         axs[0].text(i, cpu, f'{cpu:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
#     axs[0].set_title('Consolidated CPU Usage during Inference')
#     axs[0].set_xlabel('Image Index')
#     axs[0].set_ylabel('CPU Usage (%)')
#     axs[0].set_xticks(range(len(image_names)))
#     axs[0].set_xticklabels(image_names, rotation=45, ha="right")
#     axs[0].legend()

#     # Plot Memory Usage
#     axs[1].plot(memory_usage_list, marker='o', label='Memory Usage')
#     for i, (memory, image_name) in enumerate(zip(memory_usage_list, image_names)):
#         axs[1].text(i, memory, f'{memory:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
#     axs[1].set_title('Consolidated Memory Usage during Inference')
#     axs[1].set_xlabel('Image Index')
#     axs[1].set_ylabel('Memory Usage (KB)')
#     axs[1].set_xticks(range(len(image_names)))
#     axs[1].set_xticklabels(image_names, rotation=45, ha="right")
#     axs[1].legend()

#     # Plot Runtime
#     axs[2].plot(runtime_list, marker='o', label='Runtime')
#     for i, (runtime, image_name) in enumerate(zip(runtime_list, image_names)):
#         axs[2].text(i, runtime, f'{runtime:.2f}\n{image_name}', ha='center', va='bottom', fontsize=8)
#     axs[2].set_title('Consolidated Runtime of Inference')
#     axs[2].set_xlabel('Image Index')
#     axs[2].set_ylabel('Runtime (seconds)')
#     axs[2].set_xticks(range(len(image_names)))
#     axs[2].set_xticklabels(image_names, rotation=45, ha="right")
#     axs[2].legend()
    
#     #box plot
#     data = [cpu_usage_list, memory_usage_list, runtime_list]
#     boxplots = axs[3].boxplot(data, patch_artist=True, widths=0.6, showfliers=False)

#     # Customizing the box plot and adding quartile annotations
#     for i, bplot in enumerate(boxplots['boxes']):
#         # Get the data for each box (CPU, Memory, Runtime)
#         dataset = data[i]

#         # Calculate quartiles
#         quartiles = np.percentile(dataset, [25, 50, 75])

#         # Annotate Quartiles
#         axs[3].text(i+1.1, quartiles[0], f'Q1: {quartiles[0]:.2f}', va='center', ha='right', fontsize=10, color='blue')
#         axs[3].text(i+1.1, quartiles[1], f'Q2: {quartiles[1]:.2f}', va='center', ha='right', fontsize=10, color='green')
#         axs[3].text(i+1.1, quartiles[2], f'Q3: {quartiles[2]:.2f}', va='center', ha='right', fontsize=10, color='red')

#     axs[3].set_xticklabels(['CPU Usage (%)', 'Memory Usage (KB)', 'Runtime (seconds)'], fontsize=12)
#     axs[3].set_title('Box Plot of CPU, Memory, and Runtime', fontsize=14)
#     axs[3].set_ylabel('Values', fontsize=12)
    
#     write_to_csv(output_csv_path, image_names, cpu_usage_list, memory_usage_list, runtime_list)

#     plt.tight_layout()
#     plt.savefig(os.path.join(desktop_path, 'localKNN_metrics.png'),dpi=80)
#     plt.show()


# output_csv_path = os.path.join(desktop_path, 'metrics_data_knn.csv')
# input_path = '/Users/sandeepreddy/Desktop/test/'  # Replace with the path to your image or folder
# main(input_path, output_csv_path)


####################################################################################################################

#I have written for Beaglebone

>>>>>>> 2fdcaa0639ad2851901cfaf53545fa43bba40cde
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from skimage import io, color, feature
from skimage.transform import resize
import psutil
import time
import joblib
import os
import csv
import numpy as np
from memory_profiler import profile
# Load the saved model
# model = tf.keras.models.load_model("/root/inferences/modelfile/knn_model.joblib")
<<<<<<< HEAD
model = joblib.load("/Users/sandeepreddy/Desktop/Differentmodels/models40k/knn_model.joblib")


desktop_path = "/Users/sandeepreddy/Desktop/results/"
=======
model = joblib.load("/Users/sandeepreddy/Desktop/Differentmodels/models5k/knn_model.joblib")


desktop_path = "/Users/sandeepreddy/Desktop/cloud/ML_Project/ML_Project/40k/40k1/"
>>>>>>> 2fdcaa0639ad2851901cfaf53545fa43bba40cde

def preprocess_image(image_path):
    
    image = io.imread(image_path)
    if len(image.shape) == 3 and image.shape[-1] == 4:
        image = image[:, :, :3]
    img = resize(image, output_shape=(227,227))
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img_gray = color.rgb2gray(img)
    hog_features = feature.hog(img_gray, block_norm='L2-Hys', pixels_per_cell=(16, 16))
    img = hog_features.reshape(1, -1)
    
    return img


<<<<<<< HEAD
# @profile
# def calling_decorators(image_path):
#     img = preprocess_image(image_path)
#     prediction = model.predict(img)
#     return prediction 

def predict_single_image(image_path):
    # start_time = time.time()
    # psutil.cpu_percent(1)
    # initial_memory_usage =psutil.virtual_memory()[2]
    # prediction=calling_decorators(image_path)
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    # final_cpu_usage = psutil.cpu_percent(1)
    # end_time = time.time()
    # final_memory_usage = psutil.virtual_memory()[4]
    print(image_path,prediction)
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
        
=======
@profile
def calling_decorators(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    return prediction 

def predict_single_image(image_path):
    start_time = time.time()
    psutil.cpu_percent(1)
    # initial_memory_usage =psutil.virtual_memory()[2]
    prediction=calling_decorators(image_path)
    final_cpu_usage = psutil.cpu_percent(1)
    end_time = time.time()
    # final_memory_usage = psutil.virtual_memory()[4]
    print(image_path,prediction)
    mem_diff=0
    
    return prediction[0], end_time-start_time, mem_diff, final_cpu_usage


    

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
        
        create_graphs(cpu_usage_during_inference, memory_usage_during_inference, runtime, input_path)

        
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
            
            

            
>>>>>>> 2fdcaa0639ad2851901cfaf53545fa43bba40cde
    else:
        print("Invalid input path.")


<<<<<<< HEAD
def write_results_to_csv(output_csv_path, all_predictions,batch_runtime,total_cpu_usage):
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Batch Index', 'Image Name', 'Predicted Class'])
        
        for i, batch_predictions in enumerate(all_predictions):
            for filename, predicted_class in batch_predictions:
                csv_writer.writerow([i+1, filename, predicted_class])
                
            csv_writer.writerow(["Total batch time:", batch_runtime])
            csv_writer.writerow(["Total cpu time:",total_cpu_usage])

output_csv_path = os.path.join(desktop_path, 'mac_data_knn.csv')
input_path = '/Users/sandeepreddy/Desktop/testsample/'
=======
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
            csv_writer.writerow([image_name, cpu, memory, runtime])

def create_graphs(cpu_usage_list, memory_usage_list, runtime_list, image_names, output_csv_path):
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    
    colors = ['blue', 'orange', 'green', 'red', 'purple']


    # Plot CPU Usage
    axs[0].plot(cpu_usage_list, marker='o', label='CPU Usage', color=colors[0])
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
    
    write_to_csv(output_csv_path, image_names, cpu_usage_list, memory_usage_list, runtime_list)

    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, 'localKNN_metrics.png'),dpi=80)
    plt.show()


output_csv_path = os.path.join(desktop_path, 'metrics_data_knn.csv')
input_path = '/Users/sandeepreddy/Desktop/testsample/'  # Replace with the path to your image or folder
>>>>>>> 2fdcaa0639ad2851901cfaf53545fa43bba40cde
main(input_path, output_csv_path)