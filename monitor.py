import psutil
import time
import pandas as pd
import matplotlib.pyplot as plt

loads = []
try:
    while True: 
        cpu_percent = psutil.cpu_percent(interval=1)
        sampling_time = time.strftime("%I:%M:%S")
        loads.append((sampling_time, cpu_percent))
        # print(loads)

        loads_df = pd.DataFrame(loads).rename(columns={0: 'time', 1: 'cpu_percent'})
except KeyboardInterrupt:
    if len(loads_df) > 0:
        # average_time = loads_df['time'].mean()
        average_cpu_percent = loads_df['cpu_percent'].mean()
        # print("Average time:", average_time)
        print("Average CPU percent:", average_cpu_percent)
    else:
        print("No data collected.")


    
    # plt.fill_between(range(loads_df.shape[0]), loads_df.cpu_percent)
    # plt.xlabel('Time')
    # plt.ylabel('CPU Percent')
    # plt.title('CPU Usage Over Time')
    # plt.show()

 

 