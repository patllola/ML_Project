import pandas as pd
import matplotlib.pyplot as plt

# Load your data
data_full = pd.read_excel('/Users/sandeepreddy/Desktop/mac_book_results.xlsx')


print(data_full)
# Extract the relevant CPU usage columns for BeagleBone
dcnn_cpu = data_full.iloc[3:54, [0,1]].copy()
# print(dcnn_cpu)
cnn_cpu = data_full.iloc[3:38, [3,4]].copy()
knn_cpu = data_full.iloc[3:206, [6,7]].copy()
logistic_cpu = data_full.iloc[3:34, [9,10]].copy()
svm_cpu = data_full.iloc[3:60, [12,13]].copy()

dcnn_cpu.columns = ['1d CNN TIME', 'DCNN CPU Usage (%)']
cnn_cpu.columns = ['CNN TIME', 'CNN CPU Usage (%)']
knn_cpu.columns = ['knn TIME', 'KNN CPU Usage (%)']
logistic_cpu.columns = ['logistic TIME', 'Logistic CPU Usage (%)']
svm_cpu.columns = ['svm TIME', 'SVM CPU Usage (%)']

dcnn_cpu['DCNN CPU Usage (%)'] = pd.to_numeric(dcnn_cpu['DCNN CPU Usage (%)'], errors='coerce')
cnn_cpu['CNN CPU Usage (%)'] = pd.to_numeric(cnn_cpu['CNN CPU Usage (%)'], errors='coerce')
knn_cpu['KNN CPU Usage (%)'] = pd.to_numeric(knn_cpu['KNN CPU Usage (%)'], errors='coerce')
logistic_cpu['Logistic CPU Usage (%)'] = pd.to_numeric(logistic_cpu['Logistic CPU Usage (%)'], errors='coerce')
svm_cpu['SVM CPU Usage (%)'] = pd.to_numeric(svm_cpu['SVM CPU Usage (%)'], errors='coerce')

# dcnn_cpu['1d CNN TIME'] = pd.to_numeric(dcnn_cpu['1d CNN TIME'], errors='coerce')
# cnn_cpu['CNN TIME'] = pd.to_numeric(cnn_cpu['CNN TIME'], errors='coerce')
# knn_cpu['knn TIME'] = pd.to_numeric(knn_cpu['knn TIME'], errors='coerce')
# logistic_cpu['logistic TIME'] = pd.to_numeric(logistic_cpu['logistic TIME'], errors='coerce')
# svm_cpu['svm TIME'] = pd.to_numeric(svm_cpu['svm TIME'], errors='coerce')

cpu_usage_data=pd.concat([
    dcnn_cpu['DCNN CPU Usage (%)'],
    cnn_cpu['CNN CPU Usage (%)'],
    knn_cpu['KNN CPU Usage (%)'],
    logistic_cpu['Logistic CPU Usage (%)'],
    svm_cpu['SVM CPU Usage (%)']
], axis=1)


# TIME_usage_data=pd.concat([
#     dcnn_cpu['1d CNN TIME'],
#     cnn_cpu['CNN TIME'],
#     knn_cpu['knn TIME'],
#     logistic_cpu['logistic TIME'],
#     svm_cpu['svm TIME']
# ], axis=1)


cpu_usage_data.boxplot()
plt.title('Macbook Usage (%) for Different Algorithms')
plt.ylabel('Macbook cpu Usage (%)')
plt.xlabel('Algorithms')
plt.xticks([1, 2, 3, 4, 5], ['DCNN', 'CNN', 'knn', 'LR', 'SVM'])

# TIME_usage_data.boxplot()
# plt.title('Macbook Usage (%) for Different Algorithms')
# plt.ylabel('Macbook Run TIME Usage Usage (%)')
# plt.xlabel('Algorithms')
# plt.xticks([1, 2, 3, 4, 5], ['DCNN', 'CNN', 'KNN', 'LR', 'SVM'])
# Save the plot to a file
# plt.savefig('macbook_cpu_usage.png')
plt.show()