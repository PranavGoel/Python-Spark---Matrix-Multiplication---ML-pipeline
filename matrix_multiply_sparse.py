__author__ = 'pranavgoel'

from pyspark import SparkConf,SparkContext
from scipy import *
from scipy.sparse import csr_matrix
import sys, operator

def csr_matrix_create(values):

    row = []
    col = []
    data = []
    for i in values:
        a = i.split(':')
        col.append(int(a[0]))
        data.append(float(a[1]))
        row.append(0)

    return csr_matrix((data,(row,col)), shape=(1,100))

def calculate(mat):

    mat_transponse = mat.transpose(copy=True)

    return (mat_transponse*mat)

def main():
    conf = SparkConf().setAppName('Matrix')
    sc = SparkContext(conf=conf)
    assert sc.version >= '1.5.1'

    raw_matrix_file = sc.textFile(sys.argv[1])
 # Read matrix from file and split the lines based on space and use float for items
    matrix = raw_matrix_file.map(lambda line: line.split()).map(csr_matrix_create).map(calculate).reduce(operator.add)

    flocation = sys.argv[2]

    #filelocation = '/Users/pranavgoel/PycharmProjects/CMPT733/sparse2.txt'

    t_file = open(flocation,'w')

    for i in range(len(matrix.indptr)-1):
        col = matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]
        dat = matrix.data[matrix.indptr[i]:matrix.indptr[i+1]]
        total = zip(col,dat)
        val = map(lambda l: str(l[0]) + ':' + str(l[1]),total)
        val = ' '.join(val)
        t_file.write(val + '\n')


if __name__ == "__main__":
    main()