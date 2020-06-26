from utils import *
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Scene Classification')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()


if __name__ == "__main__":

    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results')
        SAVEPATH = 'Results/'

    train_images, test_images, train_labels, test_labels = load_data()

    if args.tiny:
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
        # Split accuracies and runtimes for saving
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            assert (7 < element and element < 20)
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])
        # Save results
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)

    vocabularies = []
    vocab_idx = []
    #
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                vocabularies.append(vocabulary)  # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0])  # Save the map from index to vocabulary

    #if there are already voc_ files, you dont need to make them again. Just load them with this code
    # for feature in ['sift', 'surf', 'orb']:
    #     for algo in ['kmeans', 'hierarchical']:
    #         for dict_size in [20, 50]:
    #             file = np.load('Results/voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy')
    #             vocabularies.append(file)


    test_rep = []  # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = []  # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4  # Order in which features were used

    for i in range(len(vocabularies)):
        for image in train_images:  # Compute the BOW representation of the training set
            rep = computeBow(image, vocabularies[i], features[i])  # Rep is a list of descriptors for a given image
            train_rep.append(rep)
        np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy',
                np.asarray(train_rep))  # Save the representations for vocabulary i
        train_rep = []  # reset the list to save the following vocabulary
        for image in test_images:  # Compute the BOW representation of the testing set
            rep = computeBow(image, vocabularies[i], features[i])
            test_rep.append(rep)
        np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep))  # Save the representations for vocabulary i
        test_rep = []  # reset the list to save the following vocabulary





    knn_accuracies = []
    knn_runtimes = []

    # Your code below
    for i in range(0, len(vocabularies)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
                bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
                init_time = time()
                knn_predicted_categories = KNN_classifier(bow_train, train_labels, bow_test, 9)
                knn_time = time() - init_time
                knn_accuracy = reportAccuracy(test_labels, knn_predicted_categories)  # true labels from?
            knn_runtimes.append(knn_time)
            knn_accuracies.append(knn_accuracy)

    np.save(SAVEPATH + 'knn_accuracies.npy', np.asarray(knn_accuracies))  # Save the accuracies in the Results/ directory
    np.save(SAVEPATH + 'knn_runtimes.npy', np.asarray(knn_runtimes))  # Save the runtimes in the Results/ directory



    lin_accuracies = []
    lin_runtimes = []

    # Your code below
    for i in range(0, len(vocabularies)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
                bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
                init_time = time()
                svm_predicted_categories = SVM_classifier(bow_train, train_labels, bow_test, True, 100)  # svm_lambda i'm not sure what to put
                svm_linear_time = time() - init_time
                svm_accuracy = reportAccuracy(test_labels, svm_predicted_categories)
                lin_runtimes.append(svm_linear_time)
                lin_accuracies.append(svm_accuracy)

    np.save(SAVEPATH + 'lin_accuracies.npy', np.asarray(lin_accuracies))  # Save the accuracies in the Results/ directory
    np.save(SAVEPATH + 'lin_runtimes.npy', np.asarray(lin_runtimes))  # Save the runtimes in the Results/ directory

    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []

    # Your code below
    for i in range(0, len(vocabularies)):
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                bow_train = np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy')
                bow_test = np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy')
                init_time = time()
                svm_predicted_categories = SVM_classifier(bow_train, train_labels, bow_test, False,100)  # svm_lambda i'm not sure what to put
                svm_rbf_time = time() - init_time
                svm_rbf_accuracy = reportAccuracy(test_labels, svm_predicted_categories)
                rbf_runtimes.append(svm_rbf_time)
                rbf_accuracies.append(svm_rbf_accuracy)

    np.save(SAVEPATH + 'rbf_accuracies.npy', np.asarray(rbf_accuracies))  # Save the accuracies in the Results/ directory
    np.save(SAVEPATH + 'rbf_runtimes.npy', np.asarray(rbf_runtimes))  # Save the runtimes in the Results/ directory