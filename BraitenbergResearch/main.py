import math

import matplotlib.pyplot as plt

from Light import Light
from Heat import Heat
from Vehicle import Vehicle
from MGA import Microbial


# global parameters
NI = 4
NH = 10
NO = 2
geneSize = NI * NH + NH * NO + NH + NO + 2
popSize = 80
duration = 300
recombProb = 0.05
mutatProb = 0.8
demeSize = 12
generations = 80

def fitnessFunction(genotype):
    # # Create instances of Vehicle and Light
    vehicle = Vehicle(NI, NH, NO)

    vehicle.setController(genotype)
    light = Light()
    heat = Heat()

    # Initialize lists to store positions for plotting trajectory
    x_positions = [vehicle.xPos]
    y_positions = [vehicle.yPos]
    # simulation
    distance = 0
    for _ in range(duration):
        vehicle.update(light, heat)
        x_positions.append(vehicle.xPos)
        y_positions.append(vehicle.yPos)
        distance1 = vehicle.distanceLight(light)
        distance2 = vehicle.distanceHeat(heat)
        distance = distance + distance1 - distance2
    # print(distance)
    averageDistance = distance/duration
    distanceConsidered = averageDistance/ math.sqrt(200)
    if distanceConsidered > 1:
        fitness = 0
    else:
        fitness = 1 - distanceConsidered

    return fitness


#fitnessFunction()
microbial = Microbial(fitnessFunction, popSize, geneSize, recombProb, mutatProb, demeSize, generations)
microbial.run()
microbial.showFitness()
avgfit, bestfit, genotype = microbial.fitStats()
print(genotype)



vehicle = Vehicle(NI, NH, NO)

vehicle.setController(genotype)

vehicle.brain.print()
light = Light()
heat = Heat()

# Initialize lists to store positions for plotting trajectory
x_positions = [vehicle.xPos]
y_positions = [vehicle.yPos]
# simulation
distance = 0
for _ in range(duration):
    vehicle.update(light, heat)
    x_positions.append(vehicle.xPos)
    y_positions.append(vehicle.yPos)
    distance1 = vehicle.distanceLight(light)
    distance2 = vehicle.distanceHeat(heat)
    distance = distance + distance1 - distance2
print(distance/duration)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(light.x, light.y, 'yo', markersize=10, label='Light Source')
plt.plot(heat.x, heat.y, 'ko', markersize=10, label='Heat Source')
plt.plot(x_positions, y_positions, 'b-', label='Vehicle Trajectory')
plt.plot(x_positions[1], y_positions[1], 'ro', label='Start Position')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Vehicle Movement with Light Source')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()