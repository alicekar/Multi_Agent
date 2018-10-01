# IMPORTS___________________________________________________________________________________________________

import networkx as nx
import random as rd
from operator import itemgetter
from scipy.misc import imread
import matplotlib.pyplot as plt
import numpy as np
import math
from ant import Ant
from Car import Car

# UBER DETERMINISTIC ENVIROMENT_______________________________________________________________________________

# Multiple and dynamic envimoment. Changable parameters:

# max_iterations -  Max interations for the optimization in ACO 
# midstep_size   - Parameter for plotting
# Number of cars - 2-4 cars 
# number_of_cust - number of orders
# prob_of_costumer - Probability of customer popping up in each time step

# Run by: main(max_iterations, number_of_ants, number_of_cust, prob_of_costumer)


# LISBOA PROPERTIES__________________________________________________________________________________________________

LISBOA_LOC = ["Camping", "Estádio José Alvalade", "Amadora", "Queluz","Caxias","Queijas", "LX Factory", "Alfama", "Santos", " Rossio", "Marques de Pompal", "Campo de Ourique", "Instituto", "Estrela", "Cais de sodre", "Jardim Zoológico","Prazeres", "Airport", "Amoreiras", "Martim Moniz", "Intendente", "Belem", "Alto da ajuda", "Alges", "Almada", "Xabregas", "Colombo"]
coordinates = [(863,618),(1279,156),(678,255),(334,218),(195,937),(311, 697), (1115, 862), (1647, 711), (1347, 862), (1501, 739), (1389, 614), (1187, 751), (1506, 481), (1301, 753), (1455,865), (1192,360), (1204, 854), (1528, 75), (1283, 615), (1545, 767), (1541, 633) ,(849, 993), (927, 807), (607, 951), (1237, 1220), (1861, 261), (1017, 241)]

# CREATE A MODEL______________________________________________________________________________________________

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


# ANT COLONY OPTIMIZATION_____________________________________________________________________________

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

    nc = int(round(n/number_ants))      # Number of customers for each car, accept from the last
    nc_last = int(n-(number_ants-1)*nc) # Number of customers for last car
    nodes_per_ant = []
    for i in range(number_ants-1):
        nodes_per_ant.append(nc)
    nodes_per_ant.append(nc_last)   

    return(nodes_per_ant)

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
    #print('Last Coustomer: ', last_locations)
    #print('Best solution found in iteration: ', np.argmin(B))
    #print('Best cost value: ', B[np.argmin(B)])
    #print('Start positions: ', LISBOA[start_nodes])
    #print('BE',be)

    return(best_tours)

def find_other_tours(ant, number_ants, ant_k):
    others = []
    for a in range(number_ants):
        if a != ant_k:
            others.append(ant[a].tour)
    others = [s for sublist in others for s in sublist]
    return(others)

def ACO(number_loc, number_cars, D, max_it, start_nodes, resturants, customers):
    # Parameters
    number_of_orders = len(resturants)
    n = number_loc
    Q = 1
    tau_0 = 10*Q/(n*np.mean(D))     # Initial Phromone
    alpha = 1                       # Phromone Exponential Weight
    beta = 1                        # Heuristic Exponential Weight
    rho = 0.05                      # Evaporation rate

    # Multi cars parameteres
    number_ants = number_cars

    nodes_per_ant = multi_parameters(number_of_orders, number_ants)

    print("Nodes per ant", nodes_per_ant)


    # Initialization
    tau = np.ones((n,n))*tau_0      # Phromone Matrix
    eta = np.zeros((n,n))           # Heuristic Information Matrix
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = D[i][j]
                eta[i][j] = 1/d_ij

    best_costs = np.zeros((number_ants, max_it))      # Array to hold best cost values
    
    # Best ant
    best_solutions = []
    for d in range(number_ants):
        a = Ant()
        a.cost = np.inf
        best_solutions.append(a)
    

    # Iterations
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
    
            while len(ant[k].tour) <= nodes_per_ant[k]*2:
                i = ant[k].tour[-1]
                P = (np.asarray(tau[i,:])**alpha)*(np.asarray(eta[i,:])**beta)
                # Not allowed to choose a node that have been visited before
                P[ant[k].tour] = 0
                # Not allowed to choose a next node that is not a resturant
                P[customers] = 0
                # Not allowed to choose a next node where other ants has been at
                P[other_ant_tours] = 0
                # Only allowed to choose a next node that is a resutrant that has an order
                for i in range(n):
                    if i not in resturants:
                        P[i] = 0

                P = P/np.sum(P)

                new_resturant = Roulette(P)

                i=0
                for resturant in resturants:
                    if resturant == new_resturant:
                        new_customer = customers[i]
                    i+=1
                

                ant[k].add_to_tour(new_resturant)
                ant[k].add_to_tour(new_customer)

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
            tour = ant[k].tour.copy()
            tour.append(tour[0])

            number_of_orders_for_this_ant = nodes_per_ant[k]

            for l in range(number_of_orders_for_this_ant):
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
        #print('Iteration', it, ': Best Cost = ', total_best_cost[it])
    
    best_tours = info_and_tours(best_costs, start_nodes, number_ants, best_solutions)

    return(best_solutions)

    #return(best_tours)


# PLOTTING_____________________________________________________________________________________________________

def plot_solution(x, y, solution_routes, resturants, customers, start_nodes, D, stepsize, new_order_time, new_order_customers, new_order_resturants):
    img = plt.imread("map.png")                     # Read in map of Lisbon
    implot = plt.imshow(img)
    
    x_resturants = np.asarray(x)[resturants]   
    y_resturants = np.asarray(y)[resturants]

    x_customers = np.asarray(x)[customers]
    y_customers = np.asarray(y)[customers]

    x_start = np.asarray(x)[start_nodes]
    y_start = np.asarray(y)[start_nodes]

    # Plot start nodes, restaurants and customers in different colors
    plt.scatter(x=x_start, y=y_start, c='black', s=60, marker='>', label = 'Start nodes')
    plt.scatter(x=x_resturants, y=y_resturants, c='coral', s=60, marker ='D', label = 'Resturants')
    plt.scatter(x=x_customers, y=y_customers, c = 'mediumblue', s=60, marker = 'D', label = 'Customers')   
    plt.legend()    


    # Adjust the routs to equal length for the plotting
    all_adjusted_moves = adjust_for_plot(solution_routes, D, stepsize)
    colors = ['fuchsia', 'red', 'darkviolet', 'blue']
    
    # Figure properties
    plt.title('Lisbon map')
    plt.xlabel('x coordinate')
    plt.ylabel('y coordinate')
    plt.pause(8)


    if len(all_adjusted_moves) == 4:
        route1, route2 = all_adjusted_moves[0], all_adjusted_moves[1]
        route3, route4 = all_adjusted_moves[2], all_adjusted_moves[3]

        x1, y1, x2, y2 = route1[0], route1[1], route2[0], route2[1]
        x3, y3, x4, y4 = route3[0], route3[1], route4[0], route4[1]

        # Plot all routs for all cars as times goes on
        for i in range(len(x1)):
            if len(new_order_time) != 0 and new_order_time[0] == i:
            # A new order has popped up! Replan and give out new route.

                x_customer = np.asarray(x)[new_order_customers[0]]
                y_customer = np.asarray(y)[new_order_customers[0]]

                x_resturant = np.asarray(x)[new_order_resturants[0]]
                y_resturant = np.asarray(y)[new_order_resturants[0]]

                new_order_time.pop(0)
                new_order_customers.pop(0)
                new_order_resturants.pop(0)

                text = plt.text(30,40, "New order has been made, REPLANNING.....", fontsize = 30, bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})

                plt.scatter(x=x_customer, y=y_customer, c='g', s = 80, marker = '*', label ='New orders')
                plt.scatter(x=x_resturant, y=y_resturant, c='y', s = 80, marker = '*', label ='New orders')

                arrow_c = plt.arrow(x=x_customer, y=y_customer+220, dx=0, dy=-100,  color='mediumblue',fc = "k", ec="k",head_width=40, head_length=80 )
                arrow_r = plt.arrow(x=x_resturant, y=y_resturant+220, dx=0, dy=-100,  color='coral',fc = "k", ec="k",head_width=40, head_length=80 )


                plt.pause(2)

                plt.scatter(x=x_resturant, y=y_resturant, c='coral', s=100, marker ='D', label = 'Resturants')
                plt.scatter(x=x_customer, y=y_customer, c = 'mediumblue', s=100, marker = 'D', label = 'Customers')   


                text.remove()
                arrow_c.remove()
                arrow_r.remove()


            x_val1, y_val1 = x1[i], y1[i]
            x_val2, y_val2 = x2[i], y2[i]
            x_val3, y_val3 = x3[i], y3[i]
            x_val4, y_val4 = x4[i], y4[i]
            plt.scatter(x_val1, y_val1, c = colors[0], s = 10, marker = 'h')
            plt.scatter(x_val2, y_val2, c = colors[1], s = 10, marker = 'h')
            plt.scatter(x_val3, y_val3, c = colors[2], s = 10, marker = 'h')
            plt.scatter(x_val4, y_val4, c = colors[3], s = 10, marker = 'h')
            plt.pause(0.05)

        plt.show()


    elif len(all_adjusted_moves) == 3:
        route1, route2 = all_adjusted_moves[0], all_adjusted_moves[1]
        route3 = all_adjusted_moves[2]

        x1, y1, x2, y2 = route1[0], route1[1], route2[0], route2[1]
        x3, y3 = route3[0], route3[1]

        # Plot all routs for all cars as times goes on
        for i in range(len(x1)):
            

            if len(new_order_time) != 0 and new_order_time[0] == i:
                # A new order has popped up! Replan and give out new route.

                x_customer = np.asarray(x)[new_order_customers[0]]
                y_customer = np.asarray(y)[new_order_customers[0]]

                x_resturant = np.asarray(x)[new_order_resturants[0]]
                y_resturant = np.asarray(y)[new_order_resturants[0]]

                new_order_time.pop(0)
                new_order_customers.pop(0)
                new_order_resturants.pop(0)

                text = plt.text(30,40, "New order has been made, REPLANNING.....", fontsize = 30, bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})

                plt.scatter(x=x_customer, y=y_customer, c='g', s = 80, marker = '*', label ='New orders')
                plt.scatter(x=x_resturant, y=y_resturant, c='y', s = 80, marker = '*', label ='New orders')

                arrow_c = plt.arrow(x=x_customer, y=y_customer+220, dx=0, dy=-100,  color='mediumblue',fc = "k", ec="k",head_width=40, head_length=80 )
                arrow_r = plt.arrow(x=x_resturant, y=y_resturant+220, dx=0, dy=-100,  color='coral',fc = "k", ec="k",head_width=40, head_length=80 )


                plt.pause(2)

                plt.scatter(x=x_resturant, y=y_resturant, c='coral', s=100, marker ='D', label = 'Resturants')
                plt.scatter(x=x_customer, y=y_customer, c = 'mediumblue', s=100, marker = 'D', label = 'Customers')   


                text.remove()
                arrow_c.remove()
                arrow_r.remove()


            x_val1, y_val1 = x1[i], y1[i]
            x_val2, y_val2 = x2[i], y2[i]
            x_val3, y_val3 = x3[i], y3[i]
            plt.scatter(x_val1, y_val1, c = colors[0], s = 60, marker = 'h')
            plt.scatter(x_val2, y_val2, c = colors[1], s = 60, marker = 'h')
            plt.scatter(x_val3, y_val3, c = colors[2], s = 60, marker = 'h')
            plt.pause(0.05)

        plt.show()
    

    elif len(all_adjusted_moves) == 2:
        route1, route2 = all_adjusted_moves[0], all_adjusted_moves[1]
        x1, y1, x2, y2 = route1[0], route1[1], route2[0], route2[1]

        # Plot all routs for all cars as times goes on
        for i in range(len(x1)):

            if len(new_order_time) != 0 and new_order_time[0] == i:
            # A new order has popped up! Replan and give out new route.

                x_customer = np.asarray(x)[new_order_customers[0]]
                y_customer = np.asarray(y)[new_order_customers[0]]

                x_resturant = np.asarray(x)[new_order_resturants[0]]
                y_resturant = np.asarray(y)[new_order_resturants[0]]

                new_order_time.pop(0)
                new_order_customers.pop(0)
                new_order_resturants.pop(0)

                text = plt.text(30,40, "New order has been made, REPLANNING.....", fontsize = 30, bbox={'facecolor':'grey', 'alpha':0.5, 'pad':10})

                plt.scatter(x=x_customer, y=y_customer, c='g', s = 80, marker = '*', label ='New orders')
                plt.scatter(x=x_resturant, y=y_resturant, c='y', s = 80, marker = '*', label ='New orders')

                arrow_c = plt.arrow(x=x_customer, y=y_customer+220, dx=0, dy=-100,  color='mediumblue',fc = "k", ec="k",head_width=40, head_length=80 )
                arrow_r = plt.arrow(x=x_resturant, y=y_resturant+220, dx=0, dy=-100,  color='coral',fc = "k", ec="k",head_width=40, head_length=80 )


                plt.pause(2)

                plt.scatter(x=x_resturant, y=y_resturant, c='coral', s=100, marker ='D', label = 'Resturants')
                plt.scatter(x=x_customer, y=y_customer, c = 'mediumblue', s=100, marker = 'D', label = 'Customers')   


                text.remove()
                arrow_c.remove()
                arrow_r.remove()

            x_val1, y_val1 = x1[i], y1[i]
            x_val2, y_val2 = x2[i], y2[i]
            plt.scatter(x_val1, y_val1, c = colors[0], s = 10, marker = 'h')
            plt.scatter(x_val2, y_val2, c = colors[1], s = 10, marker = 'h')
            plt.pause(0.05)

        plt.show()

# FIND ALL MIDSTEPS FOR ALL CARS AND ROUTES ___________________________________________________

def find_midsteps(solutions, D, stepsize):
    all_moves = []
    for i in range(len(solutions)):
        x_per_car = []
        y_per_car = []
        sol = solutions[i]
        midsteps = []
        for j in range(len(sol)-1):
            start = sol[j]
            end = sol[j+1]
            distance = D[start][end]
            x_steps, y_steps = calc_steps(start, end, distance, stepsize)
            x_per_car.append(x_steps)
            y_per_car.append(y_steps)

        x_car = [item for sublist in x_per_car for item in sublist]
        y_car = [item for sublist in y_per_car for item in sublist]
        all_moves.append((x_car,y_car))
     
    return(all_moves)

# ADJUST THE ROUTES FOR FACILITATING THE PLOTTING PROCESS ___________________________________________________

def adjust_for_plot(solutions, D, stepsize):
    all_moves = find_midsteps(solutions, D, stepsize)
    all_adjusted_moves = all_moves.copy()
    lengths = []

    for route in all_moves:
        l = len(route[0])
        lengths.append(l)
    max_l = lengths[np.argmax(np.asarray(lengths))]

    for r in range(len(all_moves)):
        route = all_moves[r]

        # If route shorther than the longest one, add end point until equal length
        if len(route[0]) < max_l:
            diff = max_l-len(route[0])
            last_x = route[0][-1]
            last_y = route[1][-1]
            for i in range(diff):
                all_adjusted_moves[r][0].append(last_x)
                all_adjusted_moves[r][1].append(last_y)

    return(all_adjusted_moves)

# CALCULATE MIDSTEPS BETWEEN RESTAURANT AND CUSTUMER___________________________________________________

def calc_steps(start, end, dist, step_size):
    start_node = coordinates[start]
    end_node = coordinates[end]
    nr_steps = int(round(dist/step_size))
    x_span = abs(end_node[0]-start_node[0])
    x_step = x_span/nr_steps
    y_span = abs(end_node[1]-start_node[1])
    y_step = y_span/nr_steps
    x_coordinates = []
    y_coordinates = []

    if end_node[0] >= start_node[0] and end_node[1] >= start_node[1]:
        lx = 0
        ly = 0
        for i in range(nr_steps):
            x = start_node[0] + lx
            x_coordinates.append(x)
            lx += x_step
            y = start_node[1] + ly
            y_coordinates.append(y)
            ly += y_step
        x_coordinates.append(end_node[0])
        y_coordinates.append(end_node[1])

    if end_node[0] < start_node[0] and end_node[1] >= start_node[1]:
        lx = 0
        ly = 0
        for i in range(nr_steps):
            x = start_node[0] - lx
            x_coordinates.append(x)
            lx += x_step
            y = start_node[1] + ly
            y_coordinates.append(y)
            ly += y_step
        x_coordinates.append(end_node[0])
        y_coordinates.append(end_node[1])

    if end_node[0] >= start_node[0] and end_node[1] < start_node[1]:
        lx = 0
        ly = 0
        for i in range(nr_steps):
            x = start_node[0] + lx
            x_coordinates.append(x)
            lx += x_step
            y = start_node[1] - ly
            y_coordinates.append(y)
            ly += y_step
        x_coordinates.append(end_node[0])
        y_coordinates.append(end_node[1])

    if end_node[0] < start_node[0] and end_node[1] < start_node[1]:
        lx = 0
        ly = 0
        for i in range(nr_steps):
            x = start_node[0] - lx
            x_coordinates.append(x)
            lx += x_step
            y = start_node[1] - ly
            y_coordinates.append(y)
            ly += y_step
        x_coordinates.append(end_node[0])
        y_coordinates.append(end_node[1])

    return(x_coordinates, y_coordinates)                

# CREATE RESTURANTS, COSTUMER AND START NODES________________________________________________________

def create_rcs_nodes(n, nr_of_ants):

    
    locations = list(range(n))
    rd.shuffle(locations)

    nr_of_startnodes = nr_of_ants

    nr_resturants = round((n -  nr_of_ants)/2)

    start_nodes, resturants, customers = np.split(locations, [nr_of_startnodes, nr_of_startnodes+nr_resturants])

    #print("Start nodes", start_nodes)
    #print("Possible resturants", resturants)
    #print("Possible customers", customers)


    return (start_nodes, resturants, customers)

# CREATE CUSTOMER DISTRIBUTION_____________________________________________________________________

def create_customer_distribution(n, number_of_cust, number_of_ants):

    # Create initial customer distribution

    start_nodes, resturants_nodes, custumers_nodes = create_rcs_nodes(n, number_of_ants)

    resturants = np.random.choice(resturants_nodes, number_of_cust, replace=False)
    customers = np.random.choice(custumers_nodes, number_of_cust, replace=False)
    
    print("Start nodes", start_nodes)
    print("Resturants orders", resturants)
    print("Customers", customers)


    return (start_nodes, resturants, customers, resturants_nodes, custumers_nodes)

def generate_new_order(n,resturants, customers, number_of_cust, resturant_nodes, custumers_nodes):

    possible_resturants = list(set(resturant_nodes).difference(resturants))

    possible_customers = list(set(custumers_nodes).difference(customers))

    resturants = np.random.choice(possible_resturants, number_of_cust, replace=False)
    customers = np.random.choice(possible_customers, number_of_cust, replace=False)

    return resturants, customers


# MAIN PROGRAM____________________________________________________________________________________

def main(max_iterations, m, number_of_cust, prob):

    n, x, y, D = create_Model()

    order_window = 50
    start_nodes_init, resturants_init, customers_init, resturant_nodes, custumer_nodes = create_customer_distribution(n, number_of_cust,m)

    # Plan tours for cars.
    T = ACO(n, m, D, max_iterations, start_nodes_init, resturants_init, customers_init)

    for each in T:
        print("Planning initial tour:", each.tour)

    # Create copies of the resturant and custuomer list
    resturants = resturants_init.copy()
    customers = customers_init.copy()
    start_nodes = start_nodes_init.copy()

    resturants_init = list(resturants_init)
    customers_init = list(customers_init)

    # For plotting, we need to know when the new orders pop up and when
    new_order_time = []
    new_order_customers = []
    new_order_resturants = []

    # To check if cars have done there deliveries
    delivery_bool = np.full((m), False, dtype=bool)
    print(delivery_bool)

    # Create our cars
    cars = []
    for i in range(m):
        car = Car(start_nodes[i])
        car.planned_tour = T[i].tour[1:]
        car.current_node = start_nodes[i]
        print(car.total_tour)
        cars.append(car)

    timestep = 0
    while timestep < 150:

        # New order might pop up. If it does, replan for the cars. Start nodes are the next drop of place
        r = rd.random()
        if r<prob and timestep<order_window:
            # Generate the new orders
            number_of_cust = 1
            new_resturants, new_customers = generate_new_order(n, resturants, customers, number_of_cust,
                                                               resturant_nodes, custumer_nodes)

            for i in range(len(new_resturants)):
                resturants = list(resturants)
                resturants.append(new_resturants[i])
                customers = list(customers)
                customers.append(new_customers[i])

            # Replan
            print("New order has been made", new_resturants, new_customers)
            print("Replanning...")

            # Save the new order
            new_order_resturants.append(new_resturants[0])
            new_order_customers.append(new_customers[0])
            new_order_time.append(timestep)

            # Start Nodes for the car is the next customer node for the cars
            start_nodes = []
            for car in cars:
                start_nodes.append(car.current_order_customer)

            # Planning the Routes
            T = ACO(n, m, D, max_iterations, start_nodes, np.asarray(resturants), np.asarray(customers))

            # We need to identify with ant that represent each car.
            # This is done by checking where the car is on its way to and matching it with the start node
            # Update the tours for the cars
            counter = 0
            for ant in T:
                for car in cars:
                    if ant.tour[0] == car.on_his_way_to:
                        car.update_tour(ant.tour)


        # Update each car at each timestep
        counter = 0
        for car in cars:

            if car.distance_left_to_customer == 0:
                print("Deliverered from resturant %i to customer %i", (car.current_order_resturant, car.current_order_customer))

                car.order_delivered()

            if car.has_order == False:
                if len(car.planned_tour) > 0:
                    r,c = car.set_new_order(D)

                    # Remove delegeted nodes from order list
                    resturants = list(resturants)
                    customers = list(customers)
                    resturants.remove(r)
                    customers.remove(c)

            car.update_distance()

            # If we are passed the order window and all the orders have been delivered, we save the infomration
            if timestep > order_window and car.done:
                delivery_bool[counter] = True

            counter += 1

        # If all cars are done -> Break simulation
        if np.all(delivery_bool):
            print("All deliveries has been made, we are done. Ending simulation.")
            break

        print("Continuing to drive, time:", int(timestep))

        timestep += 1


    # When the simulation is done, we return the cars driven routes
    sol_tours = []

    for car in cars:
        sol_tours.append(car.total_tour)
        print(tour_length(car.total_tour, D))


    print(sol_tours)


    # Plotting the solution
    plot_solution(x,y,sol_tours, resturants_init, customers_init, start_nodes_init, D, midstep_size, new_order_time, new_order_customers, new_order_resturants)




# CHOOSE PARAMETER VALUS AND RUN THE PROGRAM_______________________________________________________________

max_iterations = 2000
midstep_size = 50
m = 4                     # Number of cars
number_of_ants = m
number_of_cust = 8
prob_of_costumer = 0.1

main(max_iterations, number_of_ants, number_of_cust, prob_of_costumer)
