import pandas as pd
import numpy as np
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from math import fabs

#---------------Read csv file---------------#
data= pd.read_csv('Your file!!!.csv',sep=";", header=0)

print('Lets take a look at the data')
print(data)

print('Lets take a look at the types of the data')
print(data.dtypes)

# Check for NaN values in the DataFrame
nan_columns = data.columns[data.isna().any()].tolist()

if nan_columns:
    print("NaN values found in the following columns:")
    for column in nan_columns:
        print(column)
        # Drop rows containing NaN values
        data = data.dropna()
else:
    print("No NaN values found in the DataFrame.")


num_columns = data.shape[1]
datapca = data


#---------------Normalaziation of Data---------------#
print('Normalaziation of Data', end="")
scaler = StandardScaler()
X = scaler.fit_transform(datapca)
print("...done")


#---------------Performing PCA---------------#
print('Performing PCA', end="")
pca = decomposition.PCA(n_components= num_columns)
tdata=  pca.fit_transform(X)
print("...done")
cov_mat = np.cov(tdata)
print(tdata)
#---------------Print the variance ratio---------------#

for i in range(len(pca.explained_variance_ratio_)) :
    print("---------------+++++++++++++++---------------")
    print(f"The variance ratio for λ{i+1} is {pca.explained_variance_ratio_[i]}")


#---------------Print Eigenvalues---------------#

print("Eigenvalues:")
for i in range(len(pca.explained_variance_)):
    print("-----+++++-----")
    print(f"λ{i+1}: {pca.explained_variance_[i]}")


#---------------Print Eigenvectors---------------#
print("Eigenvectors:")
eigen = []
z = 0
for eigenvector in pca.components_:
    print("---------------++++++++++---------------")
    z += 1
    print(f"Eigenvector {z} : {eigenvector}")
    eigen.append(eigenvector)

#---------------Print the names of the variables---------------#
variables = []
for col in datapca.columns:
    variables.append(col)
print("The variables of the data are", variables)


#---------------Choose the number of the new variables that we want to use---------------#
print("--------------------++++++++++--------------------")
print("                                                  ")
new_variables = int(input("How many of the new variables do you want to use? "))
print("                                                  ")
print("--------------------++++++++++--------------------")



for i in range(new_variables):
    print(f'New variable {i + 1}= ')
    for j in range(len(variables)):
         if eigen[i][j] >= 0:
            print(f'+({eigen[i][j]} * {variables[j]})',)
         else:
             print(f'-({fabs(eigen[i][j])} * {variables[j]})',)
    print('\n')

total = 0
new_data = pd.DataFrame()
for i in range(new_variables):
    total += pca.explained_variance_ratio_[i]
    component_name = f"New variable_{i + 1}"  # You can name the new variables as PC_1, PC_2, ...
    new_data[component_name] = tdata[:, i]  # Add the PCA component to the new DataFrame

print(f"The {new_variables} new variables can explain {total*100}% of the total variance.")

print("New DataFrame with selected new variables:")
print(new_data)
