# load data
import pandas as pd
import numpy as np

df = pd.read_csv('relative_speed_progression_better.csv')

# get unique file ids from first column
ids = df['file'].unique()

# create empty dataframe
df2 = pd.DataFrame(columns=['user_id', 'segment_id', 'speed', 'time'])

# loop through ids
for idx, id in enumerate(ids):
    # calculate the average speed for each kilometer = segment
    speeds = []
    dist = eval(df['distance'][idx]) 
    time = eval(df['time'][idx]) 
    for d, t in zip(dist, time):
        if t == 0:
            speed = 0
        else:
            speed = (float(d) / float(t)) * 3.6
        speeds.append(speed)

    # calculate average speed per kilometer
    avg_speeds = []
    time_l = []
    distance = 0
    timing = 0
    for i in range(len(speeds)):
        distance += float(dist[i])
        timing += float(time[i])
        if distance >= 1000:
            avg_speeds.append(np.mean(speeds[:i+1]))
            time_l.append(timing)
            speeds = speeds[i+1:]
            timing = 0
            distance = distance - 1000

    # add to dataframe
    for i in range(len(avg_speeds)):
        df2 = df2.append({'user_id': id, 'segment_id': i, 'speed': avg_speeds[i], 'time': time_l[i]}, ignore_index=True)
        print("Added segment", i, "for user", idx, "out of", len(ids))

# save to csv
df2.to_csv('average_speeds_per_segment.csv', index=False)
print("DONE")