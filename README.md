# Emotion-Detection-using-Deep-Learning
This project demonstrates the use of Deep Learning to detect emotion (sad, angry, happy etc) from the images of  faces. 

## Dataset used: 
[Challenges in Representation Learning: Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/)

![](https://i.ibb.co/gMPnd28/Screen-Shot-2020-02-01-at-2-36-22-PM.png)


## Network architectures/techniques tried:
- Shallow fully connected network
- Mini VGG16
- Mini GoogLeNet
- Shallow CNN with progressively increasing channels
- Mini VGG16 with LSUV

The combination of VGG16, SGD, a bit of data augmentation, and Label Smoothing yielded the best generalization. 

![](https://i.ibb.co/92nQjFC/W-B-Chart-2-14-2020-7-22-52-PM.png)

Experimentation report available here: https://app.wandb.ai/sayakpaul/emotion-detection

I happily welcome any feedback :)

