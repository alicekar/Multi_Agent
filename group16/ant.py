import numpy as np
class Ant:

	def __init__(self):

		self.tour = []
		self.cost = 3000


	def add_to_tour(self, node):

		self.tour.append(node)

	def add_cost(self, cost):

		self.cost = cost

	def tour_length(self):

		return len(self.tour)
