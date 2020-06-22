# FSRCNN-TensorFlow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## Prerequisites
 * Python 3
 * TensorFlow-gpu >= 1.8
 * CUDA & cuDNN >= 6.0
 * Pillow
 * ImageMagick (optional)
 * Wand (optional)

## Usage
For training: `python main.py`
<br>
For testing: `python main.py --train False`

To use FSCRNN-s instead of FSCRNN: `python main.py --fast True`

Can specify epochs, learning rate, data directory, etc:
<br>
`python main.py --epoch 100 --learning_rate 0.0002 --data_dir Train`
<br>
Check `main.py` for all the possible flags

## Result

Original butterfly image:

![orig](https://github.com/igv/FSRCNN-Tensorflow/blob/master/Test/Set5/butterfly_GT.bmp?raw=true)


Ewa_lanczos interpolated image:

![ewa_lanczos](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/ewa_lanczos.png?raw=true)


Super-resolved image:

![fsrcnn](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/fsrcnn.png?raw=true)

## Additional datasets

* [General-100](https://drive.google.com/open?id=0B7tU5Pj1dfCMVVdJelZqV0prWnM)

## TODO

* Add RGB support (Increase each layer depth to 3)

## References

* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)

* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
