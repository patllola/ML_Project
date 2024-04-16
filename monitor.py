import psutil
import time
import pandas as pd

loads = []
try:
    while True: 
        cpu_percent = psutil.cpu_percent(interval=1)
        sampling_time = time.strftime("%I:%M:%S")
        loads.append((sampling_time, cpu_percent))

except KeyboardInterrupt:
    print("Experiment interrupted. Saving results.")
    loads_df = pd.DataFrame(loads, columns=['time_stamp', 'cpu_percent'])
    loads_df.to_csv("/Users/sandeepreddy/Desktop/cpu_time_knn_results.csv", index=False)

    # Calculate average CPU usage before and during experiment
    # stime_when_you_start_experiment = "11:01:56"  # Set the start time of your experiment
    # etime_when_you_end_experiment = "11:02:13"     # Set the end time of your experiment

    # before_experiment = loads_df[loads_df['time_stamp'] < stime_when_you_start_experiment]
    # during_experiment = loads_df[(loads_df['time_stamp'] >= stime_when_you_start_experiment) & 
    #                              (loads_df['time_stamp'] <= etime_when_you_end_experiment)]

    # avg_cpu_before_experiment = before_experiment['cpu_percent'].mean()
    # avg_cpu_during_experiment = during_experiment['cpu_percent'].mean()

    # print("Average CPU percent before experiment:", avg_cpu_before_experiment)
    # print("Average CPU percent during experiment:", avg_cpu_during_experiment)
