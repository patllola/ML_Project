import psutil
import time
import pandas as pd
import matplotlib.pyplot as plt

loads = []
while True: 
	cpu_percent = psutil.cpu_percent(interval=5)
	sampling_time = time.strftime("%I:%M:%S")
	loads.append((sampling_time, cpu_percent))
 
 
	loads_df = pd.DataFrame(loads).rename(columns = {0:'time', 1:'cpu_percent'})
 
	plt.fill_between(range(loads_df.shape[0]), loads_df.cpu_percent)

 