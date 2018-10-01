import networkx as nx
import random as rd
from operator import itemgetter
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import math
from ant import Ant


# LISBOA PROPERTIES__________________________________________________________________________________________________

LISBOA_LOC = ["LX Factory", "Alfama", "Santos", " Rossio", "Marques de Pompal", "Campo de Ourique", "Instituto", "Estrela", "Cais de sodre", "Jardim Zoológico","Prazeres", "Airport", "Amoreiras", "Martim Moniz", "Intendente", "Belem", "Alto da ajuda", "Alges", "Almada", "Xabregas", "Colombo"]
coordinates = [(1115, 862), (1647, 711), (1347, 862), (1501, 739), (1389, 614), (1187, 751), (1506, 481), (1301, 753), (1455,865), (1192,360), (1204, 854), (1528, 75), (1283, 615), (1545, 767), (1541, 633) ,(849, 993), (927, 807), (607, 951), (1237, 1220), (1861, 261), (1017, 241)]



# CREATE MODEL__________________________________________________________________________________________________

def create_Model():
	x = []
	y = []
	for node in coordinates:
		x.append(node[0])
		y.append(node[1])
	n = len(x)
	distance_matrix = np.zeros((n,n))

	for i in range(n):
		for j in range(i,n):
			distance_matrix[i][j] = math.sqrt(((x[i]-x[j])**2)+((y[i]-y[j])**2))
			distance_matrix[j][i] = distance_matrix[i][j]

	return(n, x, y, distance_matrix)



# PLOT SOLUTION__________________________________________________________________________________________________

def plot_solution(x, y, solution_routes, starts):
	x = np.asarray(x)
	y = np.asarray(y)
	starts_x = x[starts]
	starts_y = y[starts]
	plt.xlabel('x coordinate')
	plt.ylabel('y coordinate')
	plt.title('Downscaling of the true Lisbon distances')
	plt.scatter(x=x, y=y, c='r', s=60, marker='D')
	plt.scatter(starts_x, starts_y, c='black', s=60, marker='D', label = 'Start nodes')
	colors = ['orange', 'green', 'red', 'mediumblue']
	labels = ['Ant 1', 'Ant 2', 'Ant 3', 'Ant 4']
	for i in range(len(solution_routes)):
		tour = solution_routes[i]
		tour_x = np.asarray(x)[tour]
		tour_y = np.asarray(y)[tour]
		plt.plot(tour_x, tour_y, c = colors[i], label = labels[i] )
		plt.legend()    

	for i, place in enumerate(LISBOA_LOC):
		plt.annotate(place, (x[i], y[i]))
	#plt.show()



# INNER FUNCTIONS FOR ACO ALGORITHM__________________________________________________________________________________________________

def Roulette(P):
    r = rd.uniform(0,1) 
    C = np.cumsum(P)
    j = np.where(r<=C)[0][0]
    
    return(j)



def tour_length(tour, D):
    n = len(tour)
    the_tour = tour.copy() 
    the_tour.append(tour[0])
    
    L = 0
    for i in range(n):
        L = L + D[the_tour[i],the_tour[i+1]]
    
    return(L)



def multi_parameters(n, number_ants):
	nc = int(round(n/number_ants))		# Number of customers for each car, accept from the last
	nc_last = int(n-(number_ants-1)*nc)	# Number of customers for last car
	nodes_per_ant = []
	for i in range(number_ants-1):
		nodes_per_ant.append(nc)
	nodes_per_ant.append(nc_last)	
	start_nodes = rd.sample(range(n),number_ants)
	
	return(nodes_per_ant, start_nodes)



def info_and_tours(best_costs, start_nodes, number_ants, best_solutions):
	B = np.sum(best_costs, axis = 0)
	LISBOA = np.asarray(LISBOA_LOC)
	last_locations = []
	best_tours = []
	be = 0
	for i in range(number_ants):
		best_tours.append(best_solutions[i].tour)
		last_locations.append(best_solutions[i].tour[-1])
		be += best_solutions[i].cost
	print('Last Coustomer: ', LISBOA[last_locations])
	print('Best solution found in iteration: ', np.argmin(B))
	print('Best cost value: ', B[np.argmin(B)])
	print('Start positions: ', LISBOA[start_nodes])
	print('Sum of best solutions cost values, to compare with above: ',be)
	return(best_tours)



def find_other_tours(ant, number_ants, ant_k):
	others = []
	for a in range(number_ants):
		if a != ant_k:
			others.append(ant[a].tour)
	others = [s for sublist in others for s in sublist]
	return(others)



# ACO, MAIN ALGORITHM__________________________________________________________________________________________________

def ACO(number_loc, number_cars, D, max_it, x, y):
	# Parameters
	n = number_loc
	Q = 1
	tau_0 = 10*Q/(n*np.mean(D))			# Initial Phromone
	alpha = 1							# Phromone Exponential Weight
	beta = 1							# Heuristic Exponential Weight
	rho = 0.05 							# Evaporation rate

	# Multi cars parameters
	number_ants = number_cars
	nodes_per_ant, start_nodes = multi_parameters(n, number_ants)

	# Initialization
	tau = np.ones((n,n))*tau_0			# Phromone Matrix
	eta = np.zeros((n,n))				# Heuristic Information Matrix
	for i in range(n):
		for j in range(n):
			if i != j:
				d_ij = D[i][j]
				eta[i][j] = 1/d_ij

	best_costs = np.zeros((number_ants, max_it))		# Array to hold best cost values
	
	# Best ant
	best_solutions = []
	for d in range(number_ants):
		a = Ant()
		a.cost = np.inf
		best_solutions.append(a)
	
	plt.ion()
	for it in range(max_it):
		# Create Ant colony 
		ant = []
		for a in range(number_ants):
			the_ant = Ant()
			the_ant.add_to_tour(start_nodes[a])
			ant.append(the_ant)

		# Move ants
		for k in range(number_ants):						
			other_ant_tours = find_other_tours(ant, number_ants, k)
			
			while len(ant[k].tour) < nodes_per_ant[k]:
				i = ant[k].tour[-1]
				P = (np.asarray(tau[i,:])**alpha)*(np.asarray(eta[i,:])**beta)
				P[ant[k].tour] = 0		    # Cannot go back to its own already visited nodes
				P[other_ant_tours] = 0		# Cannot go to nodes that other ants has been at
				P = P/np.sum(P)
				j = Roulette(P)         
				ant[k].add_to_tour(j)
			
			ant[k].cost = tour_length(ant[k].tour, D)
			
		cost_sum = 0
		best_cost_sum = 0
		for k in range(number_ants):
			cost_sum += ant[k].cost
			best_cost_sum += best_solutions[k].cost
		
		if cost_sum < best_cost_sum:
			for k in range(number_ants):
				best_solutions[k] = ant[k]

		# Update Phromones
		for k in range(number_ants):
			tour = ant[k].tour.copy()   # Erase copy() to add start node as end node
			tour.append(tour[0])

			for l in range(nodes_per_ant[k]):
				i = tour[l]
				j = tour[l+1]

				tau[i][j] = tau[i][j]+Q/ant[k].cost

		# Evaporation
		tau = (1-rho)*tau

		# Store Best Cost
		for k in range(number_ants):
			best_costs[k][it] = best_solutions[k].cost		

		# Show iteration information
		total_best_cost = np.sum(best_costs, axis = 0)
		print('Iteration', it, ': Best Cost = ', total_best_cost[it])

		# Plot best tours for each iteration, showing the progress
		best_it_tours = []
		for i in range(number_ants):
			best_it_tours.append(best_solutions[i].tour)
		fig = plt.figure(1, figsize=(14,7.5))
		plot_solution(x, y, best_it_tours, start_nodes)
		plt.pause(10**(-16))
		plt.clf()

	best_tours = info_and_tours(best_costs, start_nodes, number_ants, best_solutions)
	#plot_solution(x,y, best_tours)
	#plt.show()
	return(best_tours)



# MAIN PROGRAM__________________________________________________________________________________________________

def main(max_iters, num_ants):
	n, x, y, D = create_Model()
	T = ACO(n, num_ants, D, max_iters, x, y)
	print(T)


# SET PARAMETERS AND RUN__________________________________________________________________________________________________

max_iterations = 150
m = 4      				# Number of cars
number_of_ants = m


main(max_iterations, m)









