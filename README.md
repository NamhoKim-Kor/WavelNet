# WavelNet

_WavelNet_ is a modified convolutional neural network (CNN) for processing raw physiological signals, like electrocardiogram (ECG). It performs wavelet transfrom-based spectral analysis.

The architecture of _WavelNet_ is almost identical to that of a vanilla CNN, except for the first convolutional layer. _WavelNet_'s first convolutional layer, **WaveletConv layer**, possesses kernels that are differently scaled wavelet functions. In the case of a conventional convolutional layer, whole kernel values are updated over a training process. In the case of WaveletConv layer, however, there is only one learnable parameter **a**, a scale factor, for each kernel. In this way, the WaveletConv layer gives optimized wavelet transform of the input physiological signal, and further layers conduct additional analysis upon highlighted spectral components.

By analyzing the optimized scale factors and corresponding pseudo-freqeuncies, it is possible to figure out the spectral bands of interest that _WavelNet_ considers in achieving a target task. In addition, there are different mother wavelets, and different mother wavelet can emphasize different information. By varying a mother wavelet, _WavelNet_ can handle the various aspects of the input physiological signal.

This project releases a source code of a WaveletConv layer. It is based on PyTorch, and can be used as 'torch.nn.Conv1d'.

Please note that this project is based on the environment of _Google Colab Pro+_. I recommend to check wehther the versions required packages of your environment are compatible with those of _Google Colab Pro+_.

# Reference
[1] Namho Kim et al., "_WavelNet_: A novel convolutional neural network architecture for arrhythmia classification from electrocardiogram," _Computer Methods and Programs in Biomedicine_, vol. 231, 107375, Apr. 2023. [DOI: https://doi.org/10.1016/j.cmpb.2023.107375]
