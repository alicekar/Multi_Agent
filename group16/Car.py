
class Car:

	def __init__(self, start):

		self.first_node = start

		self.start_node = start

		self.planned_tour = []

		self.tour_left = []
		self.total_tour = [start]

		self.current_timestep = 0
		self.has_order = False

		self.current_order_resturant = None
		self.current_order_customer = None

		self.on_his_way_to = None
		self.distance_left_to_customer = None

		self.current_node = None

		self.orders_left = 10000

		self.done = False

	def update_tour(self, new_tour):

		self.planned_tour = new_tour[1:]
		#self.planned_tour.remove(self.current_order_customer)

	def update_distance(self):

		self.distance_left_to_customer -= 1

	def set_new_order(self, D):
		self.done = False

		new_resturant = self.planned_tour[0]
		new_customer = self.planned_tour[1]

		self.planned_tour.remove(new_resturant)
		self.planned_tour.remove(new_customer)

		self.current_order_resturant = new_resturant
		self.current_order_customer = new_customer
		self.has_order = True

		self.on_his_way_to = new_customer

		self.distance_left_to_customer = round((D[self.current_node][new_resturant] + D[new_resturant][new_customer])/50)

		self.current_node = new_customer

		self.orders_left = len(self.planned_tour)/2

		print("New set order", (new_resturant, new_customer))
		print("Remaining tour", self.planned_tour)

		return ( new_resturant, new_customer)
	
	def order_delivered(self):
		self.current_node = self.current_order_customer
		self.total_tour.append(self.current_order_resturant)
		self.total_tour.append(self.current_order_customer)
		self.current_order = None
		self.distance_left_to_customer = 0
		self.has_order = False
		self.on_his_way_to = None

		if len(self.planned_tour) == 0:
			# We are done
			self.done = True 



