import sys
import csv
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

# Call the script with two arguments, the first one being the path to the input CSV file, and the second one the output path

if(len(sys.argv) < 3):
    exit(0)





input = sys.argv[1]
output = sys.argv[2]
header = None
sample = []
time_series_1 = []
time_series_2 = []

with open(input) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if not header:
            header = row
        else:
            sample.append(row[0])
            time_series_1.append(int(row[1]))
            time_series_2.append(int(row[2]))
    print("Read input")


nyq = 0.5 * 360
low = 0.4 / nyq
high = 45 / nyq
sos = signal.butter(3, [low, high], analog=False, btype='band', output='sos')
filtered_time_series_1 = signal.sosfilt(sos, time_series_1)
filtered_time_series_2 = signal.sosfilt(sos, time_series_2)
print("applied filters")

with open(output, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header)
    for i in range(len(filtered_time_series_1)):
        writer.writerow([sample[i], filtered_time_series_1[i], filtered_time_series_2[i]])
    print("Wrote output")




t=np.linspace(0, 15, 5400)
x = filtered_time_series_1[0:5400]
