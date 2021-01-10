import math
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl

# step of the grid
x = y = 11
a = 2*pi
b = 1
a0 = 0
b0 = 0.5
hx = (a-a0) / (x - 1)
hy = (b - b0) / (x - 1)


def toFixed(numObj, digits=0):
		return f"{numObj:.{digits}f}"


def print_matrix(M, isFloat=False):
		if isFloat:
				for row in M:
						for element in row:
								print(toFixed(element, 5), end=' ')
						print()
		else:
				for row in M:
						for element in row:
								if (element == 0):
										print(0, end=' ')
								else:
										print(element, end=' ')
						print()


def print_to_file(M, fileName='result.csv'):
		myFile = open(fileName, 'w+')
		line = ""
		for row in M:
				for element in row:
						line = line + toFixed(element, 5) + ';'
				myFile.write(line + '\n')
				line = ""
		myFile.close()


def create_grid_U_numeration(x, y):
		return [[(i * x + j) for j in range(x)] for i in range(y)]


def create_equation_system_matrix(hx, hy, x, y, U):
  Result_matrix = []
  left_equation_part = [[0.] for k in range(x * y)]
  for j in range(y):
    for i in range(x):
      equation = [0. for i in range(x * y)]
      r = j * hy + b0
      if j == 0 :
        equation[U[j][i]] = 1.
        left_equation_part[U[j][i]][0] = 0
      if j == y - 1 :
        equation[U[j][i]] = 1.
        left_equation_part[U[j][i]][0] =  1
          
      else:
        equation[U[j][i]] = (-2 / (r * hy**2)) + (1 / (r * hy) + (-2 / (r**2 * hy ** 2)))
        
        if (i - 1) >= 0:
            equation[U[j][i - 1]] = 1 / (r**2 * hx ** 2)
        if (i + 1) < x:
            equation[U[j][i + 1]] = 1 / (r**2 * hx ** 2)
        if (j - 1) >= 0:
            equation[U[j - 1][i]] = (1 / (hy ** 2)) - (1 / (r * hy))
        if (j + 1) < y:
            equation[U[j + 1][i]] = 1 / (hy ** 2)

        left_equation_part[U[j][i]][0] = -r

      Result_matrix.append(equation)
  return np.array(Result_matrix), np.array(left_equation_part)


def result_to_grid(R, U):
		Grid = []
		for j in range(len(U)):
				Grid.append([])
				for i in range(len(U[j])):
						Grid[j].append(R[U[j][i]][0])
		return Grid


# numeration for every U
U = create_grid_U_numeration(x, y)
A, B = create_equation_system_matrix(hx, hy, x, y, U)

R = np.linalg.solve(A, B)
R = np.array(R)

Result = result_to_grid(R, U)
print_matrix(U)
print_matrix(A)
print_matrix(B)
print_matrix(Result, True)
print_to_file(Result)
print_to_file(A, 'A.csv')

sin = math.sin
cos = math.cos
pi = math.pi


# def surface_plot(matrix):
# 		# acquire the cartesian coordinate matrices from the matrix
# 		# x is cols, y is rows
# 		matrix = np.matrix(matrix)
# 		(x,y) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))


# 		fig = plt.figure()
# 		ax = fig.add_subplot(111, projection='3d')
# 		surf = ax.plot_surface(x, y, matrix)
# 		return (fig, ax, surf)


# (fig, ax, surf) = surface_plot(Result, )

# fig.colorbar(surf)

# ax.set_xlabel('fi')
# ax.set_ylabel('r')
# ax.set_zlabel('U')

# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

matrix = np.matrix(Result)
(P, R) = np.meshgrid(np.arange(matrix.shape[0]), np.arange(matrix.shape[1]))
R = [j * hy + b0 for j in R]
P = [j * hx + a0 for j in P]
# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)

# Plot the surface.
ax.plot_surface(X, Y, np.matrix(Result), cmap=plt.cm.coolwarm)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('U')

plt.show()