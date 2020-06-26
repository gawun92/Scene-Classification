import os
import cv2
import numpy as np
import timeit
from time import time
import sklearn
from sklearn import neighbors, svm, cluster, preprocessing
from collections import Counter


def load_data():
    # modified path
    test_path = 'data/test/'
    train_path = 'data/train/'

    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename,  cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename,  cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)

    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    neighbor = sklearn.neighbors.KNeighborsClassifier(n_neighbors=num_neighbors)
    neighbor.fit(train_features, train_labels)
    predicted_categories = neighbor.predict(test_features)

    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):

    kernel_type = 'rbf'
    regulation_value = svm_lambda
    if is_linear is True:
        kernel_type = 'linear'
    clf = svm.SVC(kernel=kernel_type, C=regulation_value, gamma='scale')
    clf.fit(train_features, train_labels)
                            # let train_features(N x d) : X   train_labels(N x 1) : Y
    predicted_categories = clf.predict(test_features)
                            # puting test_features(M x d) as X and predicting the value of Y which is test_label(M x 1)
    return predicted_categories # (M x 1)


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size].
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.resize(input_image, (target_size, target_size))
    cv2.normalize(input_image, input_image, -1, 1, cv2.NORM_MINMAX)
    return output_image


def reportAccuracy(true_labels, predicted_labels):


    num_of_correct_predictions = 0
    Num_Predict = len(predicted_labels)  # the value is N
    for index in range(0, Num_Predict):
        if true_labels[index] == predicted_labels[index]:
            num_of_correct_predictions += 1
    accuracy = 100 * (num_of_correct_predictions / Num_Predict) # percentage

    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):

    if (feature_type == 'sift'):
        features = cv2.xfeatures2d.SIFT_create(nfeatures=128)
    elif (feature_type == 'surf'):
        features = cv2.xfeatures2d.SURF_create(extended=True)
    elif (feature_type == 'orb'):
        features = cv2.ORB_create(nfeatures=128)

    des_coll = []
    for an_img in train_images:
        kp, descriptors = features.detectAndCompute(an_img, None)
        if descriptors is None:
            continue
        for index in descriptors:
            des_coll.append(index)
    if des_coll is None:
        return None
    des_coll = np.array(des_coll)
    idx = np.random.randint(len(des_coll), size=3000)
    des_coll = des_coll[idx,:]

    if (clustering_type == 'kmeans'):
        Clustring = sklearn.cluster.KMeans(n_clusters=dict_size).fit(des_coll)
        vocabulary = Clustring.cluster_centers_
    elif (clustering_type == 'hierarchical'):
        label = sklearn.cluster.AgglomerativeClustering(n_clusters=dict_size).fit(des_coll).labels_
        arr = []
        for key in Counter(label).keys():  #
            collection = []
            for num in range(len(label)):
                if key == label[num]:
                    collection.append(des_coll[num])
            new_ROW = []
            for num_col in range(len(collection[0])):
                val = 0
                for num_row in range(len(collection)):
                    val += collection[num_row][num_col]
                val = val / (len(collection))
                new_ROW.append(val)
            arr.append(new_ROW)
        vocabulary = arr

    return vocabulary    #( the format : dict_size x d )


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    if (feature_type == 'sift'):
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=128)
        kp, descriptors = sift.detectAndCompute(image, None)
    elif (feature_type == 'surf'):
        surf = cv2.xfeatures2d.SURF_create(extended=True)
        kp, descriptors = surf.detectAndCompute(image, None)
    elif (feature_type == 'orb'):
        orb = cv2.ORB_create(nfeatures=128)
        kp, descriptors = orb.detectAndCompute(image, None)

    size_vocab = len(vocabulary)
    Bow = size_vocab * [0]
    vocabulary = np.array(vocabulary)
    if descriptors is None:
        return Bow
    for a_des in descriptors:
        min_val = 999
        save_index = 0

        for index_voc in range(size_vocab):
            temp = np.linalg.norm(a_des - vocabulary[index_voc])

            if (min_val > temp):
                min_val = temp
                save_index = index_voc
        if(save_index != size_vocab):
            Bow[save_index] += 1

    sumOfAllBins = np.sum(Bow)
    for index in range(len(Bow)):
        Bow[index] = Bow[index] / sumOfAllBins

    return Bow


def tinyImages(train_features, test_features, train_labels, test_labels):

    classResult = []
    size_8_train_features = []
    size_8_test_features = []
    size_16_train_features = []
    size_16_test_features = []
    size_32_train_features = []
    size_32_test_features = []
    Number_Train_Img = len(train_features)
    Number_Test_Img  = len(test_features)

    for index in range(0,Number_Train_Img):
        temp = imresize(train_features[index], 8).flatten()
        size_8_train_features.append(temp)
        temp = imresize(train_features[index], 16).flatten()
        size_16_train_features.append(temp)
        temp = imresize(train_features[index], 32).flatten()
        size_32_train_features.append(temp)
    for index in range(0,Number_Test_Img):
        temp = imresize(test_features[index], 8).flatten()
        size_8_test_features.append(temp)
        temp = imresize(test_features[index], 16).flatten()
        size_16_test_features.append(temp)
        temp = imresize(test_features[index], 32).flatten()
        size_32_test_features.append(temp)
    ''' The time When   Cluster : 1 3 6   and  8x8 '''
    init_time = time()
    predicted1_8 = KNN_classifier(size_8_train_features, train_labels, size_8_test_features, 1)
    classResult.append(reportAccuracy(test_labels, predicted1_8))
    classResult.append(time()-init_time)

    init_time = time()
    predicted3_8 = KNN_classifier(size_8_train_features, train_labels, size_8_test_features, 3)
    classResult.append(reportAccuracy(test_labels, predicted3_8))
    classResult.append(time() - init_time)

    init_time = time()
    predicted6_8 = KNN_classifier(size_8_train_features, train_labels, size_8_test_features, 6)
    classResult.append(reportAccuracy(test_labels, predicted6_8))
    classResult.append(time() - init_time)
    ''' The time When   Cluster : 1 3 6   and  16x16 '''
    init_time = time()
    predicted1_16 = KNN_classifier(size_16_train_features, train_labels, size_16_test_features, 1)
    classResult.append(reportAccuracy(test_labels, predicted1_16))
    classResult.append(time() - init_time)

    init_time = time()
    predicted3_16 = KNN_classifier(size_16_train_features, train_labels, size_16_test_features, 3)
    classResult.append(reportAccuracy(test_labels, predicted3_16))
    classResult.append(time() - init_time)

    init_time = time()
    predicted6_16 = KNN_classifier(size_16_train_features, train_labels, size_16_test_features, 6)
    classResult.append(reportAccuracy(test_labels, predicted6_16))
    classResult.append(time() - init_time)

    ''' The time When   Cluster : 1 3 6   and  32x32 '''
    init_time = time()
    predicted1_32 = KNN_classifier(size_32_train_features, train_labels, size_32_test_features, 1)
    classResult.append(reportAccuracy(test_labels, predicted1_32))
    classResult.append(time() - init_time)

    init_time = time()
    predicted3_32 = KNN_classifier(size_32_train_features, train_labels, size_32_test_features, 3)
    classResult.append(reportAccuracy(test_labels, predicted3_32))
    classResult.append(time() - init_time)

    init_time = time()
    predicted6_32 = KNN_classifier(size_32_train_features, train_labels, size_32_test_features, 6)
    classResult.append(reportAccuracy(test_labels, predicted6_32))
    classResult.append(time() - init_time)


    return classResult

