# Numpy array iteration and array join and split
import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()

arr = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
# iteration of each element in 3d array
# for a in arr:
#     for x in a:
#         for y in x:
#            print(y)

#using nditer(): removes the redundancy of for loops for arrays with high dimensions
for nd_x in np.nditer(arr):
    print(nd_x)

# ndenumerate method
for idx, x in np.ndenumerate(arr):
    print(idx,x) # gives dimensions and enumerations

#iterating with different step size
arr1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
for step in np.nditer(arr1[:,::2]): #::2 represents to skip 1 step ::3 is skip 2 steps
    print(step)

# iteration with different datatypes --- ????Numpy array iteration
arr2 = np.array([1,2,3])
for x_n in np.nditer(arr2, flags=['buffered'], op_dtypes=['S']):
    print(x_n)

# ------concatenation------
arr_a = np.array([1,2,3])
arr_b = np.array([4,5,6])
arr = np.concatenate((arr_a,arr_b))
print(arr)

# ------stack fxn. is same as concatenation but it is done along an axis
arr1 = np.stack((arr_a,arr_b), axis=1)
print(arr1)

# ??? hstack() to stack along rows --- need 2 brackets ----also vstack(()) and dstack(())
arr2 = np.hstack((arr_a,arr_b))
print(arr2)


# -----------array split// in notes-------------
print('array splitting')
arr_c=np.array([[[4,5,6],[7,8,9]],[[10,11,12],[1,2,3],[4,5,6]]])
arr3= np.array_split(arr_c,2)
print(arr3)
x = np.arange(-2,3)

# ----------stacking-----------------
print("array stacking")
array_d = np.arange(6).reshape(2,3)
print(array_d)
print(array_d.shape)

