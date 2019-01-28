An implementation of a DCGAN in Keras. Not too many fancy tricks, uses Spectral Normalization and Leaky Relu. 

### Results

Here's a gif of the network overfit to a single image of a cloud.
![Here's a gif](https://github.com/maxisawesome/keras_GAN/blob/master/imgs/gan.gif?raw=true)

Some other images of clouds: 
![https://github.com/maxisawesome/keras_GAN/blob/master/imgs/epoch_521.png?raw=true]
![https://github.com/maxisawesome/keras_GAN/blob/master/imgs/epoch_514.png?raw=true]

I also ran this GAN on a dataset of tattoos from (this)[http://tattoodles.com/] website. CW: lots of nudity
This dataset was much more varied, and produced worse results. See below. Perhaps with some better hyperparameter tuning I could get more than color blobs.
![https://github.com/maxisawesome/keras_GAN/blob/master/imgs/epoch_997.png?raw=true]
![https://github.com/maxisawesome/keras_GAN/blob/master/imgs/epoch_999.png?raw=true]
