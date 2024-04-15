import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = '/Users/sandeepreddy/Desktop/graph_results.xlsx'
data = pd.read_excel(file_path)

algorithm_names = data.iloc[0, :5].values.tolist()
cpu_data = data.iloc[1:, :5]
cpu_data.columns = algorithm_names
runtime_data = data.iloc[1:, 6:11]
runtime_data.columns = algorithm_names
memory_data = data.iloc[1:, 12:17]
memory_data.columns = algorithm_names


cpu_data = cpu_data.apply(pd.to_numeric, errors='coerce')
runtime_data = runtime_data.apply(pd.to_numeric, errors='coerce')
memory_data = memory_data.apply(pd.to_numeric, errors='coerce')


sns.set(style="whitegrid")

fig, axes = plt.subplots(3, 1, figsize=(10, 18))

sns.boxplot(data=cpu_data, ax=axes[0])
axes[0].set_title('CPU Usage by Algorithm')
axes[0].set_ylabel('CPU Usage (%)')
axes[0].set_xlabel('Algorithm')

sns.boxplot(data=runtime_data, ax=axes[1])
axes[1].set_title('Runtime by Algorithm')
axes[1].set_ylabel('Runtime (seconds)')
axes[1].set_xlabel('Algorithm')

sns.boxplot(data=memory_data, ax=axes[2])
axes[2].set_title('Memory Usage by Algorithm')
axes[2].set_ylabel('Memory Usage (MB)')
axes[2].set_xlabel('Algorithm')

# Display the plots
plt.tight_layout()

plt.savefig("/Users/sandeepreddy/Desktop/graph.png")
plt.show()