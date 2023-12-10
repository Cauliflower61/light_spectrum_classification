from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import skimage.feature, skimage.filters
import numpy as np
from tqdm import tqdm
from option import parser

mat2gray = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))


def GLCM(image, window_size, step, angles, gray_dim):
    C, H, W = image.shape
    feature_image = np.zeros(shape=(C, 4*5, H, W))
    for c in range(C):
        image_channel = image[c, :, :]
        image_gray = np.uint8(mat2gray(image_channel) * (2**gray_dim - 1))
        image_gray = np.pad(image_gray, window_size // 2, mode='reflect')
        for h in tqdm(range(H)):
            for w in range(W):
                image_windows = image_gray[h:h+window_size, w:w+window_size]
                glcm = skimage.feature.greycomatrix(image_windows, step, angles, levels=(2**gray_dim), symmetric=True, normed=True)
                energy = skimage.feature.greycoprops(glcm, prop='energy')
                dissimilarity = skimage.feature.greycoprops(glcm, prop='dissimilarity')
                homogeneity = skimage.feature.greycoprops(glcm, prop='homogeneity')
                correlation = skimage.feature.greycoprops(glcm, prop='correlation')
                entropy = np.zeros_like(energy)
                for angle in range(4):
                    temp = glcm[:, :, 0, angle].reshape(-1)
                    entropy_ = 0
                    for p in temp:
                        entropy_ -= p * np.log2(p) if p != 0 else 0
                    entropy[0, angle] = entropy_

                space_feature = np.concatenate([energy, dissimilarity, homogeneity, correlation, entropy], axis=1)
                feature_image[c, :, h, w] = space_feature
    feature_image = feature_image.reshape(-1, H, W)
    return feature_image


def LBP(image, window_size, radius, n_point, gray_dim):
    C, H, W = image.shape
    feature_image = np.zeros(shape=(C, 15, H, W))
    for c in range(C):
        image_channel = image[c, :, :]
        image_gray = np.uint8(mat2gray(image_channel) * (2**gray_dim - 1))
        image_gray = np.pad(image_gray, window_size // 2, mode='reflect')
        for h in tqdm(range(H)):
            for w in range(W):
                image_windows = image_gray[h:h+window_size, w:w+window_size]
                LBP = skimage.feature.local_binary_pattern(image_windows, n_point, radius, method='nri_uniform')
                hist = np.histogram(LBP.ravel(), bins=15, range=(0, 15), density=True)
                feature_image[c, :, h, w] = hist[0]
    feature_image = feature_image.reshape(-1, H, W)
    return feature_image


def Gabor(image, frequency, scale, angle, gray_dim):
    C, H, W = image.shape
    feature_image = np.zeros(shape=(C, scale*angle, H, W))
    for c in range(C):
        image_channel = image[c, :, :]
        image_gray = np.uint8(mat2gray(image_channel) * (2**gray_dim - 1))
        theta = [T/angle * np.pi for T in range(angle)]
        kernel_size = [2*i+1 for i in range(scale)]
        for ii in theta:
            for jj in kernel_size:
                real, imag = skimage.filters.gabor(image_gray, frequency=frequency, theta=ii, n_stds=jj)
                space_feature = np.sqrt(real**2 + imag**2)
                feature_image[c, :, :, :] = space_feature / np.max(space_feature)

    feature_image = feature_image.reshape(-1, H, W)
    return feature_image


def KNN(train_data_total, train_label_total, test_data_total):
    classifier = KNeighborsClassifier(n_neighbors=parser.neighbor_num)
    classifier.fit(train_data_total, train_label_total)
    predict_train = classifier.predict(train_data_total)
    predict_test = classifier.predict(test_data_total)
    return predict_train, predict_test


def SVM(train_data_total, train_label_total, test_data_total):
    classifier = svm.SVC(kernel='rbf', C=1)
    classifier.fit(train_data_total, train_label_total)
    predict_train = classifier.predict(train_data_total)
    predict_test = classifier.predict(test_data_total)
    return predict_train, predict_test
