# Devil-Detector: Evidential Deep Learning to Prevent and Detect Backdoor Attack in Image Classification

The code is written by Python 3.6 and pytorch 1.7 in GPU version. You need to install the dependent python library to run our code.

## Quick Start

1. Create folders 'datasets' and 'results' to save downloaded datasets and output results.

2. Run our Devil-Detector to detect the backdoor attack on the poisoned dataset via following command:

``python3 test_demo.py``

The default dataset is MNIST.

3. You may change the parameter 'use_softmax' and 'poison_rate' in the ``test_demo.py`` to run different experimental settings. The default poison_rate of poisoned training dataset is set to 0.2.
