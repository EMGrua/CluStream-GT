#author emgrua 10/01/2019
import matplotlib.pyplot as plt
import numpy as np
import pylab
import pickle


# days - how many days long the time-series should be
#
# sample - how many time an hour data should be sampled
#
# group - what group should the curve be (1,2,3 are the possible values)
#
# noise_width - determine the width of the normal curve used to generate noise
#
# write - if write is a string, write the data on a .md file with the string as
#         name, otherwise do not write
#
def generateData(days, sample, group, noise_width, write = False):
    xlim = sample * 24 * days
    if group == 1:
        variance = np.random.normal(2.0,0.5)
        height = np.random.normal(1.0,0.5)
    elif group == 2:
        variance = np.random.normal(5.0,0.5)
        height = np.random.normal(5.0,0.5)
    else:
        variance = np.random.normal(8.0,0.5)
        height = np.random.normal(9.0,0.5)
    # define functions
    x = np.arange(0, xlim, 1.0)
    y = height * np.sin(x / variance)
    noise = np.random.normal(0, noise_width, len(y))
    y += noise
    if isinstance(write, basestring):
        # write the data out to a file
        sinedata = open(string+'.md', 'wb')
        pickle.dump(y, sinedata)
        sinedata.close()
    return y
#Debugging purposes
'''
#generateData(28, 1, 2, 0.1)

#print(test)

    # interactive mode on
    pylab.ion()

    # set the data limits
    plt.xlim(0, xlim)
    plt.ylim(-1, 1)

    # plot the first 200 points in the data
    plt.plot(x, y)
    # hold the plot until terminated
    while True:
        plt.pause(0.5)
'''
