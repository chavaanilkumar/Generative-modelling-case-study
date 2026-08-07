# Generative-modelling-case-study

## Introduction

This project focuses on understanding and implementing Generative Adversarial Networks (GANs) using PyTorch. The project starts with simple synthetic datasets to understand how GANs work and then applies the same concepts to real-world datasets.

The project is divided into two main parts. The first part focuses on building GANs from scratch and understanding their behaviour. The second part applies GANs to practical problems involving medical images, cybersecurity network traffic, and creative sketch generation.

## Objectives

The main objectives of this project are:

* To understand the basic working principle of GANs.
* To implement a GAN using PyTorch.
* To generate and model synthetic data distributions.
* To experiment with different GAN architectures.
* To generate synthetic retinal OCT images.
* To generate synthetic cybersecurity network traffic.
* To generate birthday cake sketches using the Google QuickDraw dataset.
* To evaluate the quality of generated data using different methods.

## Part 1: GANs from Scratch

### Sine-Wave GAN

The first experiment involved creating a simple sine-wave dataset and training a fully connected GAN on it. Random values were used to create the input data, and the generator was trained to produce samples that followed the sine-wave distribution.

The Generator produces fake samples from random noise, while the Discriminator tries to distinguish between real and generated samples.

Binary Cross Entropy (BCE) loss and the Adam optimizer were used during training. As training progressed, the generated samples became closer to the original sine-wave pattern.

### Mixture of Gaussians

The second experiment used a Mixture of Gaussians dataset containing multiple clusters. This was more challenging than the sine-wave example because the generator had to learn several different regions of the data distribution.

The same fully connected GAN architecture was used. The generated samples gradually became closer to the original Gaussian clusters during training.

### Modified GAN Architecture

The final experiment in Part 1 focused on improving the basic GAN architecture.

Two main changes were made:

* Additional hidden layers were added to increase the capacity of the network.
* LeakyReLU activation functions were used instead of standard ReLU functions.

These changes resulted in more stable training and generated samples that were closer to the original Gaussian distribution.

## Part 2: Applications of GANs

### 1. OCT Retinal Image Generation

The first real-world application used the OCTMNIST dataset from MedMNIST. The dataset contains retinal Optical Coherence Tomography (OCT) images.

A Deep Convolutional GAN (DCGAN) was used because convolutional layers are better suited for image data and can learn spatial features such as edges, textures, and anatomical structures.

The images were normalised before training. The Generator was trained to create synthetic retinal images, while the Discriminator was trained to identify whether an image was real or generated.

The generated images were compared with the original OCT images. The model was able to learn the overall structure and intensity patterns of the retinal images.

The reported FID score was 0.03, indicating a high level of similarity between the feature distributions of the real and generated images. However, visual inspection is also important when evaluating generated medical images.

### 2. Cybersecurity Synthetic Traffic Generation

The second application used the CICIDS2017 dataset to generate synthetic network traffic.

Since the dataset consists mainly of numerical network traffic features, a fully connected GAN was used instead of a convolutional GAN.

The data was preprocessed and standardised before training. The Generator produced synthetic network traffic records from random noise, while the Discriminator attempted to distinguish them from real network traffic.

The model was trained for 30 epochs using the Adam optimizer.

Principal Component Analysis (PCA) was used to compare the real and generated data in a two-dimensional space. The PCA results showed that the generated samples followed many of the same patterns as the original network traffic.

There was still some separation between the real and generated samples, which suggests that the GAN learned the general distribution but did not reproduce every feature perfectly.

### 3. Creative AI – Birthday Cake Generation

The final application used the Google QuickDraw Birthday Cake dataset. The dataset contains hand-drawn birthday cake sketches.

Since the data consists of images, a DCGAN was used for this task.

The Generator created new birthday cake sketches from random noise, while the Discriminator tried to distinguish between real and generated sketches.

The generated sketches were generally able to capture the overall shape and composition of birthday cakes. Some generated images contained incomplete outlines or small deformations, which is expected when training GANs for a limited number of epochs.

The reported FID score was 0.03, showing good similarity between the generated and original sketch distributions.

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* MedMNIST
* TorchMetrics
* Google QuickDraw Dataset

## Models Used

The project uses different GAN architectures depending on the type of data:

* Fully Connected GAN – used for synthetic and tabular data.
* DCGAN – used for image-based datasets.

## Evaluation

Different methods were used to evaluate the generated data:

* Visual comparison of real and generated samples
* Generator and Discriminator loss curves
* Principal Component Analysis (PCA) for cybersecurity traffic
* Fréchet Inception Distance (FID) for image generation

## Results

Overall, the experiments showed that GANs can learn different types of data distributions and generate new synthetic samples.

The basic GAN was able to learn the sine-wave and Gaussian distributions. Changes to the network architecture improved the quality of the generated samples.

The DCGAN was able to generate retinal OCT images and birthday cake sketches with similar overall characteristics to the original datasets. The GAN trained on CICIDS2017 was also able to capture many of the statistical patterns present in the original network traffic.

However, GAN training can be sensitive to hyperparameters and training stability. The quality of the generated samples can be improved further with longer training and more advanced GAN architectures.

## Future Work

Some possible improvements for this project include:

* Training the models for more epochs.
* Using larger datasets.
* Performing more extensive hyperparameter tuning.
* Experimenting with Conditional GANs (cGANs).
* Experimenting with Wasserstein GANs (WGANs).
* Using more advanced GAN architectures for better image quality.

## Repository Contents

The main notebook containing the implementation and experiments is:

`Anil1.ipynb`

The project report is also included in the repository.

## Author

**Anil Kumar Chava**

## Conclusion

This project provided practical experience with Generative Adversarial Networks and showed how the same basic idea can be applied to different types of data.

Starting with simple synthetic datasets made it easier to understand the interaction between the Generator and Discriminator. The project then demonstrated how GANs can be used for practical applications such as medical image generation, cybersecurity traffic generation, and creative sketch generation.

The experiments also showed that GAN performance depends strongly on the network architecture, training process, and hyperparameters. Overall, the project demonstrates the flexibility of GANs for generating synthetic data across different domains.
