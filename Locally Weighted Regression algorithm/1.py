#Aim :- Implement the non-parametric Locally Weighted Regression algorithm in order to fit data points. Select appropriate data set for your experiment and draw graphs.


# Program:-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr # Changed import

def kernel(point, xmat, k):
    m, n = xmat.shape
    weights = np.eye(m)
    for j in range(m):
        diff = point - X[j]
        weights[j,j] = np.exp(diff.dot(diff.T)/(-2.0*k**2))
    return weights
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    # Add a small value to the diagonal to prevent singularity
    weighted_x = xmat.T.dot(wei).dot(xmat)
    # Check for singularity before inverting
    if np.linalg.det(weighted_x) == 0:
        print("Warning: Singular matrix encountered. Consider increasing k.")
        # You might want to handle this case differently, e.g., return a default value or skip the point
        # For now, we'll add a small value to the diagonal to attempt inversion
        weighted_x += np.eye(weighted_x.shape[0]) * 1e-9 # Add small value to diagonal

    W = np.linalg.inv(weighted_x).dot(xmat.T).dot(wei).dot(ymat)
    return W

def localWeightRegression(xmat, ymat, k):
    m, n = xmat.shape
    ypred = np.zeros(m)
    for i in range(m):
        # Ensure xmat[i] is a 2D array for the dot product with localWeight result
        ypred[i] = xmat[i].reshape(1, -1).dot(localWeight(xmat[i], xmat, ymat, k))
    return ypred
# load data points
data = pd.read_csv("Locally Weighted Regression algorithm\tips.csv")
bill = np.array(data.total_bill)
tip = np.array(data.tip)

#preparing and add 1 in bill
mbill = bill.reshape(-1, 1)
mtip = tip.reshape(-1, 1)
m = mbill.shape[0]
one = np.ones(m).reshape(-1, 1)
X = np.hstack((one, mbill))
#set k here
ypred = localWeightRegression(X,mtip,0.8) # Increased k
SortIndex = X[:,1].argsort(0)
xsort = X[SortIndex][:,1] # Corrected indexing
# Create figure and plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1,1,1)
ax.scatter(bill,tip, color='green', label='Data Points')
ax.plot(xsort,ypred[SortIndex], color='red', linewidth=3, label='Locally Weighted Regression') # Corrected xsort
plt.xlabel('Total Bill ($)')
plt.ylabel('Tip ($)')
plt.title('Locally Weighted Regression: Tips vs Total Bill')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Save the plot
plt.savefig('locally_weighted_regression.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot has been saved as 'locally_weighted_regression.png'")




#Data Set:-
# The dataset was saved in the locally weighted regression algorithm folder as tips.csv.



#Output:-
# The output will be a plot saved as 'locally_weighted_regression.png' showing the relationship between total bill and tip with the locally weighted regression line fitted to the data points.





#Result:-The above program is executed successfully.