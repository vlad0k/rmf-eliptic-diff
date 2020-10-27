import math

import numpy as np

# step of the grid
x = y = 7
a = b = 1
h = hx = hy = a / (x - 1)


def toFixed(numObj, digits=0):
		return f"{numObj:.{digits}f}"


def print_matrix(M, isFloat=False):
		if isFloat:
				for row in M:
						for element in row:
								print(toFixed(element, 4), end=' ')
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
						line = line + toFixed(element, 4) + ';'
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
						if i == 0 or j == 0 or i == x - 1 or j == y - 1:
								equation[U[j][i]] = 1.
						else:
								equation[U[j][i]] = -2 * ((1 / (hx ** 2)) + (1 / (hy ** 2)))
								if (i - 1) >= 0:
										equation[U[j][i - 1]] = 1 / (hx ** 2)
								if (i + 1) < x:
										equation[U[j][i + 1]] = 1 / (hx ** 2)
								if (j - 1) >= 0:
										equation[U[j - 1][i]] = 1 / (hy ** 2)
								if (j + 1) < y:
										equation[U[j + 1][i]] = 1 / (hy ** 2)

								left_equation_part[U[j][i]][0] = -2.

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

# accurate equetion salvation
koef = 32 / (pi ** 4)
sum = 0

x = y = 0.5

for m in range(0, 7):
		for n in range(0, 7):
				m1 = (2 * m + 1)
				n1 = (2 * n + 1)
				sum = sum + math.sin(m1 * pi * x) * math.sin(n1 * pi * y) / (m1 * n1 * (m1 ** 2 + n1 ** 2))

res = sum * koef
print(res)
