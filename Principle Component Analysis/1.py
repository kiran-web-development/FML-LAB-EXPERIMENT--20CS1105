#Aim :- Write a program to implement Principle Component Analysis for Dimensionality Reduction.


#Program :-


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a small 3×2 matrix
matrix = array([[5, 6], [8, 10], [12, 18]])
print("original Matrix: ")
print(matrix)

# calculate the mean of each column
Mean_col = mean(matrix.T, axis=1)
print("Mean of each column: ")
print(Mean_col)

# center columns by subtracting column means
Centre_col = matrix - Mean_col
print("Covariance Matrix: ")
print(Centre_col)

# calculate covariance matrix of centered matrix
cov_matrix = cov(Centre_col.T)
print(cov_matrix)

# eigendecomposition of covariance matrix
values, vectors = eig(cov_matrix)
print("Eigen vectors: ",vectors)
print("Eigen values: ",values)

# project data on the new axes
projected_data = vectors.T.dot(Centre_col.T)
print(projected_data.T)
	


# Output:-

"""
original Matrix: 
[[ 5  6]
 [ 8 10]
 [12 18]]
Mean of each column:
[ 8.33333333 11.33333333]
Covariance Matrix:
[[-3.33333333 -5.33333333]
 [-0.33333333 -1.33333333]
 [ 3.66666667  6.66666667]]
[[12.33333333 21.33333333]
 [21.33333333 37.33333333]]
Eigen vectors:  [[-0.86762506 -0.49721902]
 [ 0.49721902 -0.86762506]]
Eigen values:  [ 0.10761573 49.55905094]
[[ 0.24024879  6.28473039]
 [-0.37375033  1.32257309]
 [ 0.13350154 -7.60730348]]       """


#Result :- The above program is executed successfully.