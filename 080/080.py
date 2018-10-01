# Author: Alice Karnsund
import numpy as np
import math
import itertools
import random


# __________________________________________________________________________________
# PROBLEM 1.1
# __________________________________________________________________________________
def findNE(M):
	Game = np.array([M[0].ravel(), M[1].ravel()])
	values = []
	moves = []

	# P1 started
	for i in range(2):
		if Game[i][1] < Game[i][3]:
			values.append(Game[i,2:4])
			moves.append([i,1])
		elif Game[i][1] == Game[i][3]:
			values.append(Game[i,:2])
			moves.append([i,0])
			values.append(Game[i,2:4])
			moves.append([i,1])
		else:
			values.append(Game[i,:2])
			moves.append([i,0])

	# P2 started
	j = 0
	for i in (0,2):
		if Game[0][i] < Game[1][i]:
			values.append(Game[1,i:i+2])
			moves.append([1,j])
		elif Game[0][i] == Game[1][i]:
			values.append(Game[0,i:i+2])
			moves.append([0,j])
			values.append(Game[1,i:i+2])
			moves.append([1,j])
		else:
			values.append(Game[0,i:i+2])
			moves.append([0,j])
		j+=1

	moves = np.asarray(moves)
	values = np.asarray(values)

	ne = []
	nev = []
	for m1,m2 in itertools.combinations(moves,2):
		if m1[0] == m2[0] and m1[1]==m2[1]:
			ne.append(m1.tolist())
			# List of the nash equilibrium
			index = np.where(np.all(moves==m1, axis=1))[0]
			# Notice ne is a list with ALL moves leading to the nash 
			# equilibrium, if it occurs at more than one place 
			nev.append(values[index,:][:int(0.5*len(index)),:][0].tolist())
	
	if len(ne)==0:
		ne = []
		nev = []

	return(ne, nev)


# __________________________________________________________________________________
# PROBLEM 1.2
# __________________________________________________________________________________
# Suppose that player i assigns probability weight prob[i] to m0 and 
# probability weight (1-prob[i]) to m1
def findmixedNE(M):
	Game = np.array([M[0].ravel(), M[1].ravel()])
	# 2 matrices with each players payoffs (playing by rows)
	M1 = np.array([Game[:,0],Game[:,2]]).T
	M2 = np.array([Game[:,1],Game[:,3]])
	M = np.array([M2,M1])   # M2 needed for to calc prob. for P1 and M1 for P2

	# Caluculate probabilities for P1 and P2 to make move 0 resp.
	probs = []
	for m in M:
		if np.linalg.det(m) != 0:
			numer = (m[1,1]-m[0,1])
			denom = m[0,0]-m[0,1]-m[1,0]+m[1,1]
			if denom == 0:
				break
			p = numer/denom
			probs.append(p)
		else:
			break

	if len(probs) == 2:
		ne = np.array([[probs[0], 1-probs[0]],[probs[1], 1-probs[1]]])
		if probs[0]>1 or probs[1]>1:
			nev = []
			ne = np.array([[-1, -1],[-1, -1]])
		elif probs[0]>0.5 and probs[1]>0.5:
			nev = [Game[0,:2].tolist()]
		elif probs[0]>0.5 and probs[1]<0.5:
			nev = [Game[0,2:].tolist()]
		elif probs[0]<0.5 and probs[1]>0.5:
			nev = [Game[1,:2].tolist()]
		elif probs[0]<0.5 and probs[1]<0.5:
			nev = [Game[1,2:].tolist()]
		else:
			nev = None
			
	else:
		nev = []
		ne = np.array([[-1, -1],[-1, -1]])
	
	if len(ne[ne<0]) != 0:
		ne = np.array([[-1, -1],[-1, -1]])
		nev = []	

	ne = ne.tolist()

	return(ne, nev)







# __________________________________________________________________________________
# PROBLEM 2.1
# __________________________________________________________________________________

def payPromiseE(M):
	nash_ne, nash_nev = findNE(M)
	if len(nash_ne)==0:
		ne = []
		nev = []
		sp = []
	else:
		Game = np.array([M[0].ravel(), M[1].ravel()])
		Sum_util = np.array([[sum(Game[0,:2]), sum(Game[0,2:])],[sum(Game[1,:2]), sum(Game[1,2:])]])
		max_util = np.amax(Sum_util)
		index_max = np.asarray(np.where(Sum_util==max_util)).T
	
		ne = []
		values = []
		for i in range(len(index_max)):
			ne.append(index_max[i].tolist())
			move = index_max[i]
			row = move[0]
			if move[1] == 1:
				values.append(Game[row,2:].tolist())
			else:
				values.append(Game[row,:2].tolist())
	
		sp = []
		nev = []
		for value in values:
			if value[0]<value[1]:
				# Player 2 pays player 1
				pay = (value[1]-value[0])/2
				gain = [value[0]+pay, value[1]-pay]
			elif value[0]>value[1]:
				# Player 1 pays player 2
				pay = (value[0]-value[1])/2
				gain = [value[0]-pay, value[1]+pay]
			else:
				# No payment done, equal payoffs already
				pay = 0
				gain = [value[0], value[1]]
			sp.append(pay)
			nev.append(gain)

	check = []
	for i in range(len(nev)):
		if sum(nev[i])>sum(nash_nev[i]):
			check.append(True)
		else:
			check.append(False)

	if len(np.where(np.asarray(check)==True)[0])!=0:
		return(ne, nev, sp)
	else:
		ne = []
		nev = []
		sp = []
	
	return(ne, nev, sp)
	




# __________________________________________________________________________________
# PROBLEM 2.2
# __________________________________________________________________________________

# Help-function: taking into account the last offers, computes the new offer
def zeuthernstep(ItemsUtility1, ItemsUtility2, lastoffer1, lastoffer2):
	item_utils = [ItemsUtility1, ItemsUtility2]
	last_offers = [lastoffer1, lastoffer2]
	objects = np.array(range(len(ItemsUtility1)))

	utilities = []
	for i in range(len(item_utils)):
		u_ii = sum(item_utils[i][last_offers[i][i]])
		u_ij = sum(item_utils[i][last_offers[i-1][i]])
		utilities.append(u_ii)
		utilities.append(u_ij)

	risks = []
	for i in (0,2):	
		if utilities[i]==0:
			risks.append(1)
		else:
			risk = 1-(utilities[i+1]/utilities[i])
			risks.append(risk)

	# Check if agreement is reached
	if utilities[1] >= utilities[0] or utilities[3] >= utilities[2]:
		return(None)

	# See which player that should concede
	Concede_player = risks.index(min(risks))
	# If the plyers have equal risks, flip a coin
	if risks[0] == risks[1]:
		Concede_player = random.randrange(2)

	# Calc the difference in risks between each offer concede player 
	# can make an the constant offer of the other player		
	diff_risks = []
	tried_offers = []
	if Concede_player == 1:
		other_player = 0
		objects_CPlayer = last_offers[1][1]
		objects_otherPlayer = last_offers[0][0]
		offer_otherPlayer = last_offers[0][1]

		for offer in objects_CPlayer:
			new_risk = np.zeros(2)
			offer_CPlayer = []
			for i in last_offers[1][0]:
				offer_CPlayer.append(i)

			offer_CPlayer.append(offer)
			objs_CPlayer = np.delete(objects, offer_CPlayer).tolist()
			collection = [[objects_otherPlayer, offer_otherPlayer],[offer_CPlayer, objs_CPlayer]]
	
			for i in range(len(item_utils)):
				if sum(item_utils[i][collection[i][i]]) == 0:
					new_risk[i]=1
				else:
					u_ii = sum(item_utils[i][collection[i][i]])
					u_ij = sum(item_utils[i][collection[i-1][i]])
					r = 1 -(u_ij/u_ii)
					new_risk[i]=r

			diff_risks.append(new_risk[1]-new_risk[0])
			tried_offers.append(offer)
	else:       # CPlayer = 0
		other_player = 1
		objects_CPlayer = last_offers[0][0]
		objects_otherPlayer = last_offers[1][1]
		offer_otherPlayer = last_offers[1][0]
		for offer in objects_CPlayer:
			new_risk = np.zeros(2)
			offer_CPlayer = []
			for i in last_offers[0][1]:
				offer_CPlayer.append(i)

			offer_CPlayer.append(offer)
			objs_CPlayer = np.delete(objects, offer_CPlayer).tolist()
			collection = [[objs_CPlayer, offer_CPlayer],[offer_otherPlayer, objects_otherPlayer]]
	
			for i in range(len(item_utils)):
				if sum(item_utils[i][collection[i][i]]) == 0:
					new_risk[i]=1
				else:
					u_ii = sum(item_utils[i][collection[i][i]])
					u_ij = sum(item_utils[i][collection[i-1][i]])
					r = 1 -(u_ij/u_ii)
					new_risk[i]=r

			diff_risks.append(new_risk[0]-new_risk[1])
			tried_offers.append(offer)

	# Find new offer for Concede Player
	diff_risks = np.asarray(diff_risks)
	diff_risks[diff_risks < 0] = None
	new_offer_index = np.nanargmin(diff_risks)
	new_offer = tried_offers[new_offer_index]

	if Concede_player == 0:
		all_offers = []
		for i in last_offers[0][1]:
			all_offers.append(i)
		all_offers.append(new_offer)
		all_objects = np.delete(objects, all_offers).tolist()
		result = ([all_objects, all_offers], 1)

	else:
		all_offers = [] 
		for i in last_offers[1][0]:
			all_offers.append(i)
		all_offers.append(new_offer)
		all_objects = np.delete(objects, all_offers).tolist()
		result = ([all_offers,all_objects], 2)

	return(result)

	

# zeuthern strategy, showning each step of the negotiation
def zeuthern(ItemsUtility1, ItemsUtility2):
	# First proposal = agent's most preffered deal
	Util1 = np.asarray(ItemsUtility1)
	Util2 = np.asarray(ItemsUtility2)

	objects = np.array(range(len(Util1)))
	if len(objects) < 2:
		return('Needs to be at least 2 objects!')
	# diff taking into account both what the agent want looking at
	# where it can get most utility but at the same time making sure
	# that the other agent gives away its least preffered object. If 
	# looking only at the min utility each agent would like to get rid
	# of, agent 2 might as well offer c or d, which is not the best for
	# agent 1, ang thus not Pareto optimal. 
	diff = Util1-Util2
	offer1 = [np.where(-diff==max(-diff))[0][0]]  # P2 wants
	offer2 = [np.where(diff==max(diff))[0][0]]   # P1 wants

	of_1 = [np.where(objects!=offer1)[0].tolist(), offer1]
	of_2 = [offer2, np.where(objects!=offer2)[0].tolist()]

	all_results = []
	
	# Calc the first results [offer, player]
	P1_first_result = (of_1, 1)
	P2_first_result = (of_2, 2)
	all_results.append(P1_first_result)
	all_results.append(P2_first_result)

	# Calc next offer given the last offers
	keep_going = True
	while keep_going:
		latest_offer = zeuthernstep(Util1, Util2, of_1, of_2)
		if latest_offer == None:
			keep_going = False
		elif latest_offer[1]==1:
			of_1 = latest_offer[0]
			all_results.append(latest_offer)
		else:
			of_2 = latest_offer[0]
			all_results.append(latest_offer)

	# Return sequence of all offers and which agent made that offer,
	return(all_results)
	







