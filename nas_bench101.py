import sys
import pickle

sys.path.append('../nasbench')
from nasbench import api

CONV1X1 = "conv1x1-bn-relu"
CONV3X3 = "conv3x3-bn-relu"
MAXPOOL3X3 = "maxpool3x3"
INPUT = "input"
OUTPUT = "output"


# Load the data from file (this will take some time)
#nasbench = api.NASBench('../nasbench/nasbench_only108.tfrecord')


#with open('nacbench.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(nasbench, f)
with open('nasbench.pkl', 'rb') as f:
    nasbench = pickle.load(f)
# Create an Inception-like module (5x5 convolution replaced with two 3x3
# convolutions).
model_spec = api.ModelSpec(
    # Adjacency matrix of the module
    matrix=[[0, 1, 0, 0, 0, 1, 1],    # input layer
            [0, 0, 0, 1, 0, 1, 1],    # 1x1 conv
            [0, 0, 0, 1, 1, 0, 1],    # 3x3 conv
            [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
            [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
            [0, 0, 0, 0, 0, 0, 0]],   # output layer
    # Operations at the vertices of the module, matches order of matrix
    ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Query this model from dataset, returns a dictionary containing the metrics
# associated with this model.
data = nasbench.query(model_spec)
print(data)
