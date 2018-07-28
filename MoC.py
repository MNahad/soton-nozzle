#!/usr/bin/env python
"""
1. Lookup table for Argon (gamma=5/3) with:
    Mach number M
    Mach angle mu in degrees
    Prandtl-Meyer function neu in degrees
from 1 <= M <= 3

2. Method of Characteristics table which produces geometry for
    minimum length 2D symmetric nozzle using 4 characteristics

Author: Mohammed Nawabuddin
"""


import numpy as np		# http://www.numpy.org/
import scipy.optimize as opt	# https://scipy.org/
import math

# Set initial gamma, initial theta and desired Mach number
gamma = 5/3
Mdes = 2.7
initialTheta = 0.4


def mu(mach):
"""
Returns mu from Mach
"""
    return math.asin(1 / mach) * (180 / math.pi)


def neu(mach):
"""
Returns neu from Mach
"""
    return ((((gamma+1)/(gamma-1))**0.5) *
            (math.atan2((((gamma-1)/(gamma+1))*((mach**2)-1))**0.5, 1)) -
            math.atan2(((mach**2)-1)**0.5, 1)) * \
            (180 / math.pi)


def _tanAlpha(prv, nxt):
"""
Called by coord()
"""
    return math.tan(((prv + nxt)/2) * (math.pi/180))


def coord(xUpp, yUpp, xLow, yLow, angUpp, angLow, angSum, angDiff, isSym):
"""
Used in calculating nozzle geometry
"""
    if isSym == 1:
        xp = xUpp - ((yUpp)/(_tanAlpha(angUpp, angDiff)))
        return [xp, 0]
    if isSym == 0:
        xp = (xUpp*_tanAlpha(angUpp, angDiff) - xLow*_tanAlpha(angLow, angSum) +
              yLow - yUpp) / \
              (_tanAlpha(angUpp, angDiff) - _tanAlpha(angLow, angSum))
        yp = yLow + (xp - xLow)*_tanAlpha(angLow, angSum)
        return [xp, yp]


def areaRatio(gamma, mach):
"""
Used for finding the area ratio at a set gamma and Mach number
"""
    return (((2/(gamma+1))*(1+(((gamma-1)/2)*(mach**2)))) **
            ((gamma+1)/(2*(gamma-1)))) / \
            mach


""" Create lookup table consisting of Mach numbers from 1.00 to
3.00, and their associated mu and neu angles. """

lookup = np.zeros((201, 3))

np.put(lookup, np.arange(0, 601, 3), np.arange(1.00, 3.01, 0.01))

for row in lookup:
    row[1] = mu(row[0])

for row in lookup:
    row[2] = neu(row[0])

# Save table
np.savetxt("outputs/lookup.csv", lookup, delimiter=',')

""" Create MoC table for fluid at the desired gamma and
at an exit velocity at the desired Mach number """

# Blank MoC table
MoC = np.zeros((18, 11))

# Add point IDs
for row in np.arange(4, 18):
    MoC[row, 0] = row - 3

# Add initial info at throat
MoC[3, 3] = neu(Mdes)/2
MoC[0, 3] = initialTheta
MoC[1, 3] = ((MoC[3, 3] - MoC[0, 3]) / 3) + MoC[0, 3]
MoC[2, 3] = 2 * ((MoC[3, 3] - MoC[0, 3]) / 3) + MoC[0, 3]

# Fill in neu and Riemann for throat
for row in np.arange(4):
    MoC[row, 4] = MoC[row, 3]
    MoC[row, 2] = MoC[row, 3] + MoC[row, 4]

# Fill other points by iterating over characteristic lines
outerWaveIter = 3
for charac in np.array([[17, 2], [15, 3], [12, 4], [8, 5]]):
    innerWaveIter = 4
    for row in np.arange(charac[0], charac[0]-charac[1], -1):
        MoC[row, 1] = MoC[outerWaveIter, 2]
        if innerWaveIter < 4:
            MoC[row, 2] = MoC[innerWaveIter, 2]
        MoC[row, 3] = (MoC[row, 2] - MoC[row, 1]) / 2
        MoC[row, 4] = (MoC[row, 2] + MoC[row, 1]) / 2
        innerWaveIter -= 1
    outerWaveIter -= 1

# Housekeeping
for row in np.array([8, 12, 15, 17]):
    MoC[row, 3] = (MoC[row - 1, 2] - MoC[row - 1, 1]) / 2
    MoC[row, 4] = (MoC[row - 1, 2] + MoC[row - 1, 1]) / 2

""" Find Mach from neu by searching for the optimum value using
the SciPy implementation of the van Wijngaarden-Deker-Brent
method. Find mu from Mach. Find theta+mu and theta-mu. """
for row in MoC:
    row[5] = opt.brentq(lambda x: neu(x) - row[4], 1.00, 3.00)
    row[6] = mu(row[5])
    row[7] = row[3]+row[6]
    row[8] = row[3]-row[6]

# Set initial x and y geometry for the throat to (0,1)
for row in np.arange(4):
    MoC[row, 10] = 1

""" For each point in the nozzle, define its two other
significant points and a flag for the type of point. Then find
the x and y coordinates of that point
and insert into table. """
for point in np.array([[100, 4, 0, 0],
                      [4, 5, 1, 1],
                      [5, 6, 2, 1],
                      [6, 7, 3, 1],
                      [7, 8, 3, 2],
                      [100, 9, 5, 0],
                      [9, 10, 6, 1],
                      [10, 11, 7, 1],
                      [11, 12, 8, 2],
                      [100, 13, 10, 0],
                      [13, 14, 11, 1],
                      [14, 15, 12, 2],
                      [100, 16, 14, 0],
                      [16, 17, 15, 2]]):
    if point[3] == 0:
        MoC[point[1], 9], MoC[point[1], 10] = \
         coord(MoC[point[2], 9], MoC[point[2], 10],
               0, 0,
               MoC[point[2], 8], 0,
               0, MoC[point[1], 8], 1)
    if point[3] == 1:
        MoC[point[1], 9], MoC[point[1], 10] = \
         coord(MoC[point[2], 9], MoC[point[2], 10],
               MoC[point[0], 9], MoC[point[0], 10],
               MoC[point[2], 8], MoC[point[0], 7],
               MoC[point[1], 7], MoC[point[1], 8], 0)
    if point[3] == 2:
        MoC[point[1], 9], MoC[point[1], 10] = \
         coord(MoC[point[2], 9], MoC[point[2], 10],
               MoC[point[0], 9], MoC[point[0], 10],
               MoC[point[2], 3], MoC[point[0], 7],
               MoC[point[0], 7], MoC[point[0], 3], 0)

# Save the MoC table
np.savetxt("outputs/MoC.csv", MoC, delimiter=',')

# Compare nozzle geometry with quasi-1D nozzle theory

AXAStar = np.array([areaRatio(gamma, MoC[row, 5]) for row in [8, 12, 15, 17]])
ratioDiff = np.array([MoC[row, 10] for row in [8, 12, 15, 17]]) - AXAStar

# Print tables

print("LOOKUP")
for row in lookup:
    print(row)
print("\n")
print("LEFT HALF OF MOC")
print(MoC[0:4, :5])
print(MoC[4:9, :5])
print(MoC[9:13, :5])
print(MoC[13:16, :5])
print(MoC[16:18, :5])
print("\n")
print("RIGHT HALF OF MOC")
print(MoC[0:4, 5:])
print(MoC[4:9, 5:])
print(MoC[9:13, 5:])
print(MoC[13:16, 5:])
print(MoC[16:18, 5:])
print("\n")
print("A/A* FOR MACH NUMBERS FOUND AT WALL POINTS")
for ratio in AXAStar:
    print(ratio)
print("\n")
print("DIFFERENCE BETWEEN MOC AREA RATIO AND QUASI-1D AREA RATIO")
for diff in ratioDiff:
    print(diff)
