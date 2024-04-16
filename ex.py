import pandas as pd
import time

cpu_data = pd.read_csv("/Users/sandeepreddy/Desktop/cpu_time_knn_results.csv")

    # Set the start and end times of the experiment
stime_when_you_start_experiment = "10:54:08"  
etime_when_you_end_experiment = "10:54:19"     

# Filter CPU data for before and during the experiment
before_experiment = cpu_data[cpu_data['time_stamp'] < stime_when_you_start_experiment]
during_experiment = cpu_data[(cpu_data['time_stamp'] >= stime_when_you_start_experiment) & 
                                (cpu_data['time_stamp'] <= etime_when_you_end_experiment)]

# Calculate average CPU usage before and during the experiment
avg_cpu_before_experiment = before_experiment['cpu_percent'].mean()
avg_cpu_during_experiment = during_experiment['cpu_percent'].mean()

print("Average CPU percent before experiment:", avg_cpu_before_experiment)
print("Average CPU percent during experiment:", avg_cpu_during_experiment)