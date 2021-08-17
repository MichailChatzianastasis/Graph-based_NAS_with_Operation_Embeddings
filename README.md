Graph-based Neural Architecture Search with Operation Embeddings
===============================================================================

Abstract:
-----
Neural Architecture Search (NAS) has recently gained increased attention, as a class of approaches that automatically searches in an input space of network architectures.
A crucial part of the NAS pipeline is the encoding of the architecture that consists of the applied computational blocks,
namely the operations and the links between them. Most of the existing approaches either fail to capture the structural properties of the architectures or use hand-engineered
vector to encode the operator information. In this paper, we propose the replacement of fixed operator encoding with
learnable representations in the optimization process. This approach, which effectively captures the relations of different operations, leads to smoother and more accurate representations of the architectures and consequently to improved performance of the end task. Our extensive evaluation in ENAS benchmark demonstrates the effectiveness
of the proposed operation embeddings to the generation of highly accurate models, achieving state-of-the-art performance. Finally, our method produces top-performing
architectures that share similar operation and graph patterns, highlighting a strong correlation between the structural properties of the architecture and its performance.

Installation:
------------

Tested with Python 3.6, PyTorch 0.4.1.

Install [PyTorch](https://pytorch.org/) >= 0.4.1

Install python-igraph by:

    pip install python-igraph

Install pygraphviz by:

    conda install graphviz
    conda install pygraphviz

Other required python libraries: tqdm, six, scipy, numpy, matplotlib

Training:
--------
We provide the code to incorporate operation embeddings into [DVAE] models (https://github.com/muhanzhang/D-VAE).
To train the model with operation embeddings run:

    python train.py --data-name final_structures6 --save-interval 100 --save-appendix _DVAE-EMB --epochs 300 --lr 1e-4 --model DVAE_EMB --bidirectional --nz 56 --batch-size 32

Bayesian Optimization:
---------------------

To perform Bayesian optimization experiments after training the graph autoencoder models, the following additional steps are needed.

Install sparse Gaussian Process (SGP) based on Theano:

    cd bayesian_optimization/Theano-master/
    python setup.py install
    cd ../..

Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) by: 

    cd software/enas
    mkdir data
    cd data
    wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    mv cifar-10-batches-py/ cifar10/
    cd ../..

Download the [pretrained ENAS model](https://www.dropbox.com/sh/h5q9g784uf41xhi/AADZaGvYqHucoQ373U17J_pPa?dl=0) to "software/enas/" (for evaluating a neural architecture's weight-sharing accuracy). There should be a folder named "software/enas/outputs_6/", which contains four model files.

Install [TensorFlow](https://www.tensorflow.org/install/gpu) >= 1.12.0

Install R package _bnlearn_:

    R
    install.packages('bnlearn', lib='/R/library', repos='http://cran.us.r-project.org')

Then, in "bayesian_optimization/", type:

    ./run_bo_ENAS.sh


to run Bayesian optimization for neural architecturs and Bayesian networks, respectively.


Latent Space Visualazation:
---------------------
You can reduce the dimensions of the latent space using t-sne, in order to visualize it in 2d space.
To use t-sne in the learned latent space run:

`python latent_space_tsne.py --data-name final_structures6 --model DVAE_EMB --save-appendix DVAE_EMB --epochs 300`

It will construct the appropriate tsne.pkl files in the results directory of the model. Then you can read the tsne.pkl files and plot the 2 dimensions of the latent space.  

Architecture Performance and Graph Properties:
-------------------------------------------------

To extract the igraphs from the dataset run:

`python igraph_extraction.py --data-name final_structrures6 --model DVAE_EMB --save-appendix DVAE_EMB`

It will create a pickle file igraphs_train.pickle. Then in file igraphs_analysis.py you can read the pickle file and perform the analysis of the graphs.
