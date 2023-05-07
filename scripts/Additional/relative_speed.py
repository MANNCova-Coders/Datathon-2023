# load all files in big_data folder to one dataframe
import glob
import pandas as pd
import numpy as np
from math import atan, atan2, cos, sin, sqrt, radians, tan

path = r'big_data' # use your path
all_files = glob.glob(path + "/*.csv")

from geopy import distance
import pandas as pd
import math 

def vincenty_distance(lat1, lon1, h1, lat2, lon2, h2):
    # Constants
    f = 1/298.257223563
    a = 6378137
    b = 6356752.314245
    L = radians(abs(lon2 - lon1))

    # Reduced latitudes
    U1 = atan((1 - f) * tan(radians(lat1)))
    U2 = atan((1 - f) * tan(radians(lat2)))

    # Trigonometric functions
    sinU1 = sin(U1)
    cosU1 = cos(U1)
    sinU2 = sin(U2)
    cosU2 = cos(U2)
    sinL = sin(L)
    cosL = cos(L)

    # Initial bearing
    sigma = 0
    deltaSigma = 1e-12
    while abs(deltaSigma) > 1e-12:
        cosSigma = sinU1*sinU2 + cosU1*cosU2*cosL
        sinSigma = sqrt(sinU1**2 + (cosU1**2 * sinU2**2) - (2 * cosU1 * cosU2 * cosL))
        sigma_prev = sigma
        sigma = atan2(sinSigma, cosSigma)
        deltaSigma = sigma - sigma_prev

    # Distance calculation
    uSq = cosU1**2 * (a**2 - b**2) / b**2
    A = 1 + uSq/16384*(4096+uSq*(-768+uSq*(320-175*uSq)))
    B = uSq/1024 * (256+uSq*(-128+uSq*(74-47*uSq)))
    deltaSigmaSq = deltaSigma**2
    S = b*A*(sigma - B*sin(sigma))*(sigma + B*sin(sigma) + deltaSigmaSq/4*(A*cos(2*sigma)-(B/2)*sin(2*sigma)))

    # Elevation differences
    dh = h2 - h1

    # Final distance
    D = sqrt(S**2 + dh**2)
    return D


def relative_speed_progression(filename):
    # read in data
    df = pd.read_csv(filename)

    # calculate time difference between consecutive points and sort by time
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', inplace=True)
    df['time_diff'] = df['time'].diff().dt.total_seconds()
    df['time_diff'].fillna(0, inplace=True)
    times = df['time_diff'].tolist()
    

    # calculate distance between consecutive points
    distances = []
    time_diffs = []
    for i in range(len(df)-1):
        lat1, lon1, ele1 = df.loc[i, 'lat'], df.loc[i, 'lng'], df.loc[i, 'ele']
        lat2, lon2, ele2 = df.loc[i+1, 'lat'], df.loc[i+1, 'lng'], df.loc[i+1, 'ele']
        dist = vincenty_distance(lat1, lon1, ele1, lat2, lon2, ele2)
        distances.append(dist)
        time_diffs.append(times[i+1])

    print(filename, distances, time_diffs)

    return filename, distances, time_diffs


if __name__ == '__main__':
    dic = dict()
    for file in all_files:
        file, distances, time_diffs = relative_speed_progression(file)
        dic[file] = [distances, time_diffs]

    # to csv
    df = pd.DataFrame.from_dict(dic, orient='index')
    df.to_csv('relative_speed_progression_better.csv')