class Item:
	def __init__(self,weight,value):
		self.weight = weight
		self.value = value
		self.ratio = value/weight


def fractional_knapsack(items,capacity):
	items.sort(key=lambda x :x.ratio, reverse=True)
	total_value = 0

	for item in items:
		if capacity == 0:
			break

		if item.weight <= capacity:
			total_value += item.value
			capacity -= item.weight

		else:
			fraction = capacity / item.weight
			total_value = item.value * fraction

	return total_value

items = [Item(10,50), Item(20,30), Item(30,80)]
capacity = 60

print("Maximum Value: ", fractional_knapsack(items,capacity))
