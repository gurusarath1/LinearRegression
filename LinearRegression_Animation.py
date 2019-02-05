import numpy as np
import matplotlib.pyplot as plt

def plotStyle(pltX, grid=True, Title='', xlabel='', ylabel=''):
    pltX.grid(grid)
    pltX.title(Title)
    pltX.xlabel(xlabel)
    pltX.ylabel(ylabel)



class LinearRegression:



	def __init__(self, Dataset, alpha = 0.000001):
		self.alpha = alpha
		self.Dataset = np.matrix(np.genfromtxt(Dataset, delimiter=','))
		self.Feature_Vector = self.Dataset[:,0:-1]
		self.Target_Vector = self.Dataset[:,-1]
		self.NumberOfDataPoints = self.Target_Vector.shape[0]
		self.NumberOfFeatures = self.Feature_Vector.shape[1]

		onesArray = np.matrix(np.ones(self.NumberOfDataPoints))

		self.Feature_Vector = np.concatenate((onesArray.T, np.matrix(self.Feature_Vector)), axis = 1)
		self.Target_Vector = np.matrix(self.Target_Vector)

		#self.theta = np.matrix(np.zeros(self.NumberOfFeatures + 1))

		self.theta = np.matrix('5.0,5.0')
		print(self.theta)
		self.theta_Next = np.matrix(np.zeros(self.NumberOfFeatures + 1))


	def updateTheta(self):
		i = 0
		for x in self.theta_Next:
			self.theta[i] = x
			i += 0


	def FindCost(self, index_dataPoint):
		costVector = (self.theta * self.Feature_Vector[index_dataPoint].T) - self.Target_Vector[index_dataPoint]
		return costVector.A1.tolist()[0]

	def update_Theta_0(self):

		sumOfAllCosts = 0

		for i in range(self.NumberOfDataPoints):
			sumOfAllCosts += self.FindCost(i)


		self.theta_Next[0,0] = self.theta_Next[0,0] - self.alpha * (sumOfAllCosts / self.NumberOfDataPoints)


	def update_Theta_x(self, x):

		sumOfAllCosts = 0

		for i in range(self.NumberOfDataPoints):
			sumOfAllCosts += self.FindCost(i) * self.Feature_Vector[i].A1.tolist()[x]


		self.theta_Next[0,x] = self.theta_Next[0,x] - self.alpha * (sumOfAllCosts / self.NumberOfDataPoints)



	def TotalCost(self):

		sumOfAllCosts = 0

		for i in range(self.NumberOfDataPoints):
			sumOfAllCosts += (self.FindCost(i)) ** 2
		
		return  sumOfAllCosts / (2 * self.NumberOfDataPoints)





linR = LinearRegression('ex1data1.txt')

plt.ion()
plt.figure()
plt.show()


previousCost = linR.TotalCost() + 1000
while previousCost - linR.TotalCost() > 0.001:
	previousCost = linR.TotalCost()

	linR.update_Theta_0()
	linR.update_Theta_x(1)
	linR.updateTheta()
	print('Cost', linR.TotalCost())
	T0, T1 = linR.theta.A1

	plt.clf()
	plotStyle(plt, xlabel='Size of House (scaled)--', ylabel = 'Cost of House (scaled)--', Title  = 'DataSet ' + '\nTheta 0 = ' + str(T0) + '\nTheta 1 = ' + str(T1) + '\nLearning rate = ' + str(linR.alpha))
	plt.axis((5,10,0,10))
	plt.plot(linR.Dataset[:,0:-1], linR.Dataset[:,-1], 'xr', label = 'DataSet')
	plt.plot([x for x in range(50)], [T0 + T1 * x for x in range(50)])
	plt.pause(0.005)



x = input('Press enter')