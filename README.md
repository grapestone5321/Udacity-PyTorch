# Udacity-PyTorch
Udacity-PyTorch

--------


# Intro to Deep Learning with PyTorch

In this course, you’ll learn the basics of deep learning, and build your own deep neural networks using PyTorch. 

You’ll get practical experience with PyTorch through coding exercises and projects implementing state-of-the-art AI applications such as style transfer and text generation.


### LESSON 1: Welcome to the course!

Welcome to this course on deep learning with PyTorch!


### LESSON 2: Introduction to Neural Networks

Learn the concepts behind how neural networks operate and how we train them using data.

- Discover the basic concepts of deep learning such as neural networks and gradient descent

- Implement a neural network in NumPy and train it using gradient descent with in-class programming exercises

- Build a neural network to predict student admissions

### LESSON 3: Talking PyTorch with Soumith Chintala

Hear from Soumith Chintala, the creator of PyTorch, about the past, present, and future of the PyTorch framework.

- Hear from Soumith Chintala, the creator of PyTorch, how the framework came to be, where it’s being used now, and how it’s changing the future of deep learning

### LESSON 4: Introduction to PyTorch

Learn how to use PyTorch to build and train deep neural networks. By the end of this lesson, you will build a network that can classify images of dogs and cats with state-of-the-art performance.

- Build your first neural network with PyTorch to classify images of clothing

- Work through a set of Jupyter Notebooks to learn the major components of PyTorch

- Load a pre-trained neural network to build a state-of-the-art image classifier

### LESSON 5: Convolutional Neural Networks

Learn how to use convolutional neural networks to build state-of-the-art computer vision models.

- Use PyTorch to build Convolutional Neural Networks for state-of-the-art computer vision applications

- Train a convolutional network to classify dog breeds from images of dogs

### LESSON 6: Style Transfer

Use a deep neural network to transfer the artistic style of one image onto another image.

- Use a pre-trained convolutional network to create new art by merging the style of one image with the content of another image

- Implement the paper "A Neural Algorithm of Artistic Style” by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge"

### LESSON 7: Recurrent Neural Networks

Learn how to use recurrent neural networks to learn from sequential data such as text. Build a network that can generate realistic text one letter at a time.

- Build recurrent neural networks with PyTorch that can learn from sequential data such as natural language

- Implement a network that learns from Tolstoy’s Anna Karenina to generate new text based on the novel

### LESSON 8: Sentiment Prediction RNNs

Here you'll build a recurrent neural network that can accurately predict the sentiment of movie reviews.

- Use PyTorch to implement a recurrent neural network that can classify text

- Use your network to predict the sentiment of movie reviews

### LESSON 9: Deploying PyTorch Models

In this lesson, we'll walk through a tutorial showing how to deploy PyTorch models with Torch Script.

- Soumith Chintala teaches you how to deploy deep learning models with PyTorch

- Build a chatbot and compile the network for deployment in a production environment

-------

# udacity/deep-learning-v2-pytorch
https://github.com/udacity/deep-learning-v2-pytorch


## Deep Learning (PyTorch)
This repository contains material related to Udacity's Deep Learning Nanodegree program. It consists of a bunch of tutorial notebooks for various deep learning topics. In most cases, the notebooks lead you through implementing models such as convolutional networks, recurrent networks, and GANs. There are other topics covered such as weight initialization and batch normalization.

There are also notebooks used as projects for the Nanodegree program. In the program itself, the projects are reviewed by real people (Udacity reviewers), but the starting code is available here, as well.


### attention
Attention basics

### autoencoder
data handling

### batch-norm
fix typo

### convolutional-neural-networks
copy edit

### cycle-gan
copy edits and matrix shape for char rnn

### dcgan-svhn
DCGAN for SVHN images

### gan-mnist
mnist GAN notebooks

### intro-neural-networks
alt solution to one hot encoding

### intro-to-pytorch
Update Part 3 - Training Neural Networks (Solution).ipynb

### keras
keras exercises and examples

### project-bikesharing
Fixes Issue #136

### project-dog-classification
Docs: note dataset location in workspace

### project-face-generation
face generation project

### project-tv-script-generation
copy edits and matrix shape for char rnn

### recurrent-neural-networks
copy edits and matrix shape for char rnn

### sentiment-analysis-network
add alternate weight initialization strategy

### sentiment-rnn
Merge pull request #96 from Fernandohf/bugFix-RuntimeError

### style-transfer
fix typo in comment

### tensorflow/intro-to-tensorflow
Update reqs

### transfer-learning
minor copy edit

### weight-initialization
fix link

### word2vec-embeddings
SkipGram Word2Vec implementation




# Table Of Contents
## Tutorials
### Introduction to Neural Networks

- Introduction to Neural Networks: Learn how to implement gradient descent and apply it to predicting patterns in student admissions data.

- Sentiment Analysis with NumPy: Andrew Trask leads you through building a sentiment analysis model, predicting if some text is positive or negative.

- Introduction to PyTorch: Learn how to build neural networks in PyTorch and use pre-trained networks for state-of-the-art image classifiers.

### Convolutional Neural Networks

- Convolutional Neural Networks: Visualize the output of layers that make up a CNN. Learn how to define and train a CNN for classifying MNIST data, a handwritten digit database that is notorious in the fields of machine and deep learning. Also, define and train a CNN for classifying images in the CIFAR10 dataset.

- Transfer Learning. In practice, most people don't train their own networks on huge datasets; they use pre-trained networks such as VGGnet. Here you'll use VGGnet to help classify images of flowers without training an end-to-end network from scratch.

- Weight Initialization: Explore how initializing network weights affects performance.

- Autoencoders: Build models for image compression and de-noising, using feedforward and convolutional networks in PyTorch.

- Style Transfer: Extract style and content features from images, using a pre-trained network. Implement style transfer according to the paper, Image Style Transfer Using Convolutional Neural Networks by Gatys et. al. Define appropriate losses for iteratively creating a target, style-transferred image of your own design!

### Recurrent Neural Networks
- Intro to Recurrent Networks (Time series & Character-level RNN): Recurrent neural networks are able to use information about the sequence of data, such as the sequence of characters in text; learn how to implement these in PyTorch for a variety of tasks.

- Embeddings (Word2Vec): Implement the Word2Vec model to find semantic representations of words for use in natural language processing.

- Sentiment Analysis RNN: Implement a recurrent neural network that can predict if the text of a moview review is positive or negative.

- Attention: Implement attention and apply it to annotation vectors.

### Generative Adversarial Networks
- Generative Adversarial Network on MNIST: Train a simple generative adversarial network on the MNIST dataset.

- Batch Normalization: Learn how to improve training rates and network stability with batch normalizations.

- Deep Convolutional GAN (DCGAN): Implement a DCGAN to generate new images based on the Street View House Numbers (SVHN) dataset.

- CycleGAN: Implement a CycleGAN that is designed to learn from unpaired and unlabeled data; use trained generators to transform images from summer to winter and vice versa.

- Deploying a Model (with AWS SageMaker)

- All exercise and project notebooks for the lessons on model deployment can be found in the linked, Github repo. Learn to deploy pre-trained models using AWS SageMaker.

### Projects
- Predicting Bike-Sharing Patterns: Implement a neural network in NumPy to predict bike rentals.

- Dog Breed Classifier: Build a convolutional neural network with PyTorch to classify any image (even an image of a face) as a specific dog breed.

- TV Script Generation: Train a recurrent neural network to generate scripts in the style of dialogue from Seinfeld.

- Face Generation: Use a DCGAN on the CelebA dataset to generate images of new and realistic human faces.

### Elective Material
- Intro to TensorFlow: Starting building neural networks with TensorFlow.

- Keras: Learn to build neural networks and convolutional neural networks with Keras.






-------

## Udacity-deep-learning-v2-pytorch/intro-to-pytorch/

: https://github.com/udacity/deep-learning-v2-pytorch/tree/master/intro-to-pytorch

### Deep Learning with PyTorch
This repo contains notebooks and related code for Udacity's Deep Learning with PyTorch lesson. This lesson appears in our AI Programming with Python Nanodegree program.

- Part 1: Introduction to PyTorch and using tensors
- Part 2: Building fully-connected neural networks with PyTorch
- Part 3: How to train a fully-connected network with backpropagation on MNIST
- Part 4: Exercise - train a neural network on Fashion-MNIST
- Part 5: Using a trained network for making predictions and validating networks
- Part 6: How to save and load trained models
- Part 7: Load image data with torchvision, also data augmentation
- Part 8: Use transfer learning to train a state-of-the-art image classifier for dogs and cats



-------

## pytorch.org
https://pytorch.org/

## GET STARTED
https://pytorch.org/get-started/locally/

Select preferences and run the command to install PyTorch locally, or get started quickly with one of the supported cloud platforms.

## WELCOME TO PYTORCH TUTORIALS
https://pytorch.org/tutorials/





-------

