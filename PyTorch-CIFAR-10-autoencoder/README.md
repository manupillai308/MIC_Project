# building-autoencoders-in-Pytorch
This is a reimplementation of the blog post "Building Autoencoders in Keras". Instead of using MNIST, this project uses CIFAR10.

## Current Results (Trained on Tesla K80 using Google Colab)
First attempt: (BCEloss=~0.57)  
![decode](/weights/colab_predictions.png)

Best Predictions so far: (BCEloss=~0.555)  
![decode](/weights/colab_predictions2.png)
![decode](/weights/colab_predictions22.png)

Targets:  
![target](/weights/colab_tar.png)
![target](/weights/target2.png)

## Previous Results (Trained on GTX1070)
First attempt: (Too much MaxPooling and UpSampling)  
![decode](/weights/decoded_img.png)

Second attempt: (Model architecture is not efficient)  
![decode](/weights/decoded_img2.png)

Targets:  
![decode](/weights/target.png)

## License
[MIT](LICENSE)
