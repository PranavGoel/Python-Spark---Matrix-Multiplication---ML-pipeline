---------------------------------------------------------------------------------------------------------------------------------------------------------------
File Name: matrix_multiply.py

Description: Executes Matrix multiplication using a single matrix A, by multiplying the Transpose of Matrix A with the actual Matrix A.

Notes: Instead of using the traditional way to calculate matrix multiply, here I am using the theory of outer product, https://en.wikipedia.org/wiki/Matrix_multiplication#Outer_product
It means, no matter how large n (row) will be, if the number of columns of a matrix can be put on a single machine, we can just parallely use an individual column of tranpose(A) to multiply the relative row of A, and finally add all the matrics up.

---------------------------------------------------------------------------------------------------------------------------------------------------------------

Input File Name: matrix_data.txt 

Description: Here, a simplified 5x5 matrix is used.

Note: The original input data is huge (File Size is 15MB) and could not be uploadded to GitHub. The code is also good for n rows data input when n is huge.


---------------------------------------------------------------------------------------------------------------------------------------------------------------
