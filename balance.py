import re
import csv
import random


def getAnnotations(filePath):
    with open(filePath, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (re.split(' +', line)[1:3] for line in stripped if line)
        lines = list(lines)

    return lines[2:]


def getTimeseries(filePath):
    time_series_1 = []
    time_series_2 = []

    with open(filePath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            time_series_1.append(float(row[1]))
            time_series_2.append(float(row[2]))

    return time_series_1, time_series_2


def createStaticWindows(annotations, time_series_1, time_series_2):
    annotatedWindows = []

    for heartbeat in annotations:
        low = int(heartbeat[0]) - 40
        high = int(heartbeat[0]) + 40
        time_series_1_window = time_series_1[low:high]
        time_series_2_window = time_series_2[low:high]
        annotatedWindows.append([heartbeat[1], time_series_1_window, time_series_2_window])

    return annotatedWindows


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
        balanced_beats = normal_beats + balanced_normal_beats

    random.shuffle(balanced_beats)
    return balanced_beats


def main():
    annotations = getAnnotations("annotations.txt")
    time_series_1, time_series_2 = getTimeseries("100_filtered.csv")
    annotatedWindows = createStaticWindows(annotations, time_series_1, time_series_2)
    balanced_dataset = rebalance(annotatedWindows)
    print(balanced_dataset[0])


main()
