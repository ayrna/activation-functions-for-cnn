# Activation functions for CNN
## Activation functions for convolutional neural networks: proposals and experimental study


## Algorithms included

This repo contains the code to run experiments with different activation functions that have been used recently for convolutional network models. These are:

* ReLU
* LReLU
* RTReLU
* RTPReLU
* PairedReLU
* EReLU
* EPReLU
* SQRT
* RReLU
* ELU
* SlopedReLU
* PELU
* PTELU
* MPELU
* s<sub>+</sub>
* s<sub>++</sub>
* s<sub>+2</sub>
* s<sub>+2</sub>L
* ELUs<sub>+2</sub>
* ELUs<sub>+2</sub>L


## Installation

### Dependencies

This repo basically requires:

 * Python         (>= 3.6.8)
 * click          (>=6.7)
 * h5py           (>=2.9.0)
 * Keras          (==2.2.4)
 * matplotlib     (>=3.1.1)
 * numpy          (>=1.17.2)
 * opencv-python  (>=4.1.2)
 * pandas         (>=0.23.4)
 * Pillow         (>=5.2.0)
 * prettytable    (>=0.7.2)
 * scikit-image   (>=0.15.0)
 * scikit-learn   (>=0.21.3)
 * tensorflow     (==1.13.1)


### Compilation

To install the requirements, use:

**Install for CPU**
  `pip install -r requirements.txt`

**Install for GPU**
  `pip install -r requirements_gpu.txt`


## Development

Contributions are welcome. Pull requests are encouraged to be formatted according to [PEP8](https://www.python.org/dev/peps/pep-0008/), e.g., using [yapf](https://github.com/google/yapf).

## Usage

You can run all the experiments with CIFAR-10, CIFAR-100, CINIC-10, MNIST and Fashion-MNIST by running the following lines:

  ```sh
  python main_experiment.py experiment -f exp/activations_cifar10.json
  python main_experiment.py experiment -f exp/activations_cifar100.json
  python main_experiment.py experiment -f exp/activations_cinic10.json
  python main_experiment.py experiment -f exp/activations_mnist.json
  python main_experiment.py experiment -f exp/activations_fashion.json
  ```

Note that the CINIC dataset must be stored under `../datasets/CINIC`.

The `.json` files contain all the details about the experiments settings.

After running the experiments, you can use `tools.py` to watch the results:

```sh
python tools.py
```

## Citation

The paper titled "Activation functions for convolutional neural networks: proposals and experimental study" has been submitted to IEEE Transactions on Neural Networks and Learning Systems (IEEE TNNLS).

## Contributors

#### Activation functions for convolutional neural networks: proposals and experimental study

* Víctor Manuel Vargas ([@victormvy](https://github.com/victormvy))
* Pedro Antonio Gutiérrez ([@pagutierrez](https://github.com/pagutierrez))
* César Hervás-Martínez (chervas@uco.es)
