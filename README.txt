
This project is Scene Recognition.
It is to build a set of visual recognition which is classifying scenes in different categories.
There are 15 different categories and 100 imanges in each.
All tested(used) images are 200 x 300 pixels.
Since a lot of pictures are used and total size is big, I do not post the pictures in github.
If want to run the code with the images, you can download the file with the below link.
 https://drive.google.com/file/d/15T2xONreheaI_M8L8WGzp6gnNtjYMxw9/view?usp=sharing

When loading the pictures, there are four different kinds of data.
train_image, test_image, train_labels, and test_label.

First, with "train_image", I built the two different dictionaries of size 20 and 50 
with three different features: "surf", "sift", and "orb".  
There are two different algorithm is used "kmeans" and "hierarchical".
The number of descriptors are a lot so that I randomly picked 3000 samples and 
126 features are extracted each. 
In other words, there are total 12 files in each of the different settings.

With this extracted feature, I created "Bag of Words" which is a kind of a bag to contain
features. This bag of words is a different format of image representation. This collection is 
histogramized in each of pictures' features.

Based on the histogramized features collection, I tested the other pictures with the following different ways:
rbf SVM, linear SVM and KNN.

 (openCV should be this version --> "opencv-contrib-python==3.4.2.16")
When feature=128   randompick = 3000
rbf_accuracies.npy
Max : 50.65326633165829
Min : 13.266331658291458

lin_accuracies.npy
Max : 51.69179229480737
Min : 14.706867671691793

knn_accuracies.npy
Max : 39.89949748743719  
Min : 11.624790619765495

