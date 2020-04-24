# Optimizing Deep Neural Network

This project aims to train CNN models for image classification tasks and reduces the computational overhead of large models by compressing the weights of deep neural networks using Keras.

- Basic CIFAR10 CNN classifier for Benchmarking
- **Parameter pruning** 
   - *Paper* - [Learning both Weights and Connections for Efficient
Neural Networks](https://arxiv.org/pdf/1506.02626)
   - Compresses the weights of CNN by removing weights with small absolute value 
- **Tensor decomposition**
   - *Paper* - [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530)
   - Applies Tucker decomposition to the weights of a pre-trained CNN  to reduce computations
        - Decomposes a single Convolution layer into a number of convolutional layers where total number of computations are lesser. 



## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install keras and related dependencies.

- [tensorly](http://tensorly.org/stable/index.html) - used for Tensor Decompositions

```bash
pip install keras
pip install tensorly
```


## References
- [Learning both Weights and Connections for Efficient
Neural Networks](https://arxiv.org/pdf/1506.02626) - Song Han, Jeff Pool, John Tran, William J. Dally
- [Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications](https://arxiv.org/abs/1511.06530) - Yong-Deok Kim, Eunhyeok Park, Sungjoo Yoo, Taelim Choi, Lu Yang, Dongjun Shin
- [Review on tensor decompositions](http://www.sandia.gov/~tgkolda/pubs/pubfiles/TensorReview.pdf)
## License
[MIT](https://choosealicense.com/licenses/mit/)
