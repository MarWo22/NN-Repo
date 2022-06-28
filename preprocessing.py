import sys
import re
from scipy import signal
import csv
import random


# ----------------------------------------------------------------------------------------
#   Call the script as 'preprocessing.py "path_to_timestamp.csv" "path_to_annotations.txt"'
#   This script preprocesses the data and rebalences it
# ----------------------------------------------------------------------------------------

# Reads the timeseries from the .csv file
def read_data(dataPath):
    time_series_1 = []
    time_series_2 = []

    with open(dataPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            time_series_1.append(float(row[1]))
            time_series_2.append(float(row[2]))

    return time_series_1, time_series_2


# Reads the annotations from the .txt file
def read_annotations(annotationsPath):
    with open(annotationsPath, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (re.split(' +', line)[1:3] for line in stripped if line)
        lines = list(lines)

    return lines[2:]


# Applies a 3rd order butterworth filters with a 0.5 low cutoff and 45 high cutoff
def apply_filter(time_series):
    nyq = 0.5 * 360
    low = 0.4 / nyq
    high = 45 / nyq
    sos = signal.butter(3, [low, high], analog=False, btype='band', output='sos')
    return signal.sosfilt(sos, time_series)


# Creates a static 160 timestamp window of each heartbeat
def create_static_windows(annotations, time_series_1, time_series_2, time_since_last_beat):
    annotatedWindows = []
    for i, heartbeat in enumerate(annotations[1:]):
        low = int(heartbeat[0]) - 40
        high = int(heartbeat[0]) + 40
        time_series_1_window = normalize(time_series_1[low:high])
        time_series_2_window = normalize(time_series_2[low:high])
        annotatedWindows.append(
            [str(heartbeat[1]), str(time_since_last_beat[i])] + list(time_series_1_window) + list(time_series_2_window))
    return annotatedWindows


# Rebalances the data to make sure that the frequency of normal and pathological heartbeats are the same
def rebalance(annotatedWindows):
    normal_beats = []
    pathological_beats = []
    balanced_beats = []

    for window in annotatedWindows:
        if window[0] == 'N':
            normal_beats.append(window)
        else:
            pathological_beats.append(window)

    if normal_beats > pathological_beats:
        balanced_pathological_beats = pathological_beats.copy()
        for _ in range(len(normal_beats) - len(pathological_beats)):
            balanced_pathological_beats.append(random.choice(pathological_beats))
        balanced_beats = normal_beats + balanced_pathological_beats
    else:
        balanced_normal_beats = normal_beats.copy()
        for _ in range(len(pathological_beats) - len(normal_beats)):
            balanced_normal_beats.append(random.choice(normal_beats))
        balanced_beats = pathological_beats + balanced_normal_beats

    random.shuffle(balanced_beats)
    return balanced_beats


# Writes the output to a csv file
def write_output(balanced_dataset):
    with open("processed_data101.csv", "w", newline="") as outfile:
        writer = csv.writer(outfile)
        timestamp_headers = []
        for i in range(0, 160):
            timestamp_headers.append("timestamp_" + str(i))

        writer.writerow(['annotation', 'time_since_last_beat'] + timestamp_headers)
        for i in range(len(balanced_dataset)):
            writer.writerow(balanced_dataset[i])


def normalize(list):
    normalized_list = []
    max_val = max(list)
    min_val = min(list)

    for val in list:
        normalized_list.append((val - min_val) / (max_val - min_val))

    return normalized_list


def one_hot_encoding(dataset):
    encoder = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', '[', '!', ']', 'e', 'j', 'E', '/', 'f', 'x', 'Q', '|']
    for row in range(len(dataset)):
        for annotation in encoder:
            if dataset[row][0] == annotation:
                dataset[row][0] = encoder.index(annotation)
    return dataset


def get_time_since_last_beat(annotations):
    time_since_last_beat = []
    for i in range(1, len(annotations)):
        time_since_last_beat.append(int(annotations[i][0]) - int(annotations[i - 1][0]))
    return normalize(time_since_last_beat)


def main():
    if len(sys.argv) < 3:
        print("Not enough arguments")
        exit(0)

    dataPath = sys.argv[1]
    annotationsPath = sys.argv[2]
    time_series_1, time_series_2 = read_data(dataPath)
    annotations = read_annotations(annotationsPath)
    filtered_time_series_1 = apply_filter(time_series_1)
    filtered_time_series_2 = apply_filter(time_series_2)
    time_since_last_beat = get_time_since_last_beat(annotations)
    annotatedWindows = create_static_windows(annotations, filtered_time_series_1, filtered_time_series_2,
                                             time_since_last_beat)
    balanced_dataset = rebalance(annotatedWindows)
    encoded_dataset = one_hot_encoding(balanced_dataset)
    write_output(encoded_dataset)


if __name__ == "__main__":
    main()
