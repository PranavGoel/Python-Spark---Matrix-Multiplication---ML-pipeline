Spark MLlib and Spark ML

matrix_multiply.py

Check data input in data_matrix.txt. Note: The original input data is huge and could not be uploadded to GitHub. The code is also good for n rows data input when n is huge.
The code in matrix_multiply.py is using Spark python to calculate transpose(A)*A, A is a matrix (input).
Instead of using the traditional way to calculate matric multiply, here I am using the theory of outer product, https://en.wikipedia.org/wiki/Matrix_multiplication#Outer_product
It means, no matter how large n (row) will be, if the number of columns of a matrix can be put on a single machine, we can just parallely use an individual column of tranpose(A) to multiply the relative row of A, and finally add all the matrics up.
matrix_multiply_sparse.py

The code here is using the same idea for matrix multiply, but is desiged for handling sparse matrix, by taking advantage of python csr_matrix.
spark_ml_pipline.py

The code in this python file is to get to know Spark machine learning pipeline.
Using cross validation to tune parameters, getting the best prediction model
