import scipy.io as scio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import utils
import seaborn as sns
from option import parser


def data_loading(image_path, GT_path):
    image = scio.loadmat(image_path)['paviaU'].transpose(2, 0, 1)       # the spectrum image
    GT = scio.loadmat(GT_path)['paviaU_gt']                             # the ground truth
    return image / np.max(image), GT


def PCA(image, dim):
    C, H, W = image.shape                               # the shape of image
    image = image.reshape(C, -1)                        # flatten the image to [spectrum_dim, space_dim]
    mean = np.mean(image, axis=1, keepdims=True)        # calculate mean
    image = image - mean                                # decentralization
    Covar = np.matmul(image, image.T)                   # calculate covariance matrix
    eig, eigvec = np.linalg.eig(Covar)                  # eigenvalue decomposition
    PCA_eig = eig[:dim]                                 # select larger eigenvalue
    PCA_eigvalues = eigvec[:, :dim]                     # select responded eigenvector
    error_ratio = 1 - np.sum(PCA_eig) / np.sum(eig)     # calculate the compressed error ratio
    PCA_image = np.matmul(PCA_eigvalues.T, image)       # project image to feature space
    return PCA_image.reshape(dim, H, W), error_ratio


def calculate_class_num(GT):
    # calculate numbers of class
    class_num = np.max(GT)                              # total class numbers
    label2index = {}                                    # label -> index dictionary
    for i in range(class_num + 1):
        index = np.where(GT == i)
        label2index[str(i)] = np.stack([index[0], index[1]], axis=1)
    per_class_num = [len(label2index[i])
                     for i in label2index.keys()]       # each class number
    return label2index, class_num, per_class_num


def feature_extraction(image, method):
    if method is None:
        feature_image = image
    elif method == 'GLCM':
        feature_image = utils.GLCM(image, parser.GLCM_window_size,
                           parser.GLCM_step, parser.GLCM_angles, parser.GLCM_gray_dim)
        feature_image = np.concatenate([image, feature_image], axis=0)
    elif method == 'LBP':
        feature_image = utils.LBP(image, parser.LBP_window_size, parser.LBP_R, parser.LBP_N, parser.LBP_gray_dim)
        feature_image = np.concatenate([image, feature_image], axis=0)
    elif method == 'Gabor':
        feature_image = utils.Gabor(image, parser.Gabor_frequency, parser.Gabor_scale,
                                    parser.Gabor_angle, parser.Gabor_gray_dim)
        feature_image = np.concatenate([image, feature_image], axis=0)
    elif method == 'mixed':
        LBP_feature = utils.LBP(image, parser.LBP_window_size, parser.LBP_R, parser.LBP_N, parser.LBP_gray_dim)
        GLCM_feature = utils.GLCM(image, parser.GLCM_window_size, parser.GLCM_step, parser.GLCM_angles, parser.GLCM_gray_dim)
        feature_image = np.concatenate([image, LBP_feature, GLCM_feature], axis=0)
        # for i in range(feature_image.shape[0]):
        #     feature_image[i, :, :] = utils.mat2gray(feature_image[i, :, :])
    return feature_image


def datasets_dividing(feature_image, label2index, class_num, per_class_num, train_data_ratio):
    # dataset  dividing
    train_data = {}
    train_index = {}
    test_data = {}
    test_index = {}
    np.random.seed(0)
    for i in range(1, class_num + 1):
        index = label2index[str(i)]
        np.random.shuffle(index)
        train_index[str(i)] = index[:int(train_data_ratio * per_class_num[i])]
        test_index[str(i)] = index[:]
        train_data[str(i)] = feature_image[:, train_index[str(i)][:, 0], train_index[str(i)][:, 1]]
        test_data[str(i)] = feature_image[:, test_index[str(i)][:, 0], test_index[str(i)][:, 1]]
    return train_data, train_index, test_data, test_index


def calculate_indices(train_predict, train_label_total, test_predict, test_label_total):

    confusion_matrix_train = confusion_matrix(train_predict, train_label_total)
    confusion_matrix_test = confusion_matrix(test_predict, test_label_total, labels=None)

    acc_train = np.sum(np.diag(confusion_matrix_train)) / np.sum(confusion_matrix_train)
    acc_test = np.sum(np.diag(confusion_matrix_test)) / np.sum(confusion_matrix_test)

    return acc_train, acc_test, confusion_matrix_train, confusion_matrix_test


def classifier(train_data, test_data, test_index, method):

    for iter1, key in enumerate(train_data.keys()):
        data = train_data[key]
        train_data_total = data if iter1 == 0 else np.concatenate([train_data_total, data], axis=1)
        label = np.ones(shape=(int(data.shape[1]), 1)) * int(key)
        train_label_total = label if iter1 == 0 else np.concatenate([train_label_total, label], axis=0)

    for iter1, key in enumerate(test_data.keys()):
        data = test_data[key]
        test_data_total = data if iter1 == 0 else np.concatenate([test_data_total, data], axis=1)
        label = np.ones(shape=(int(data.shape[1]), 1)) * int(key)
        test_label_total = label if iter1 == 0 else np.concatenate([test_label_total, label], axis=0)
        index = test_index[key]
        test_index_total = index if iter1 == 0 else np.concatenate([test_index_total, index], axis=0)

    train_data_total = train_data_total.transpose((1, 0))
    test_data_total = test_data_total.transpose((1, 0))

    if method == 'KNN':
        train_predict, test_predict = utils.KNN(train_data_total, train_label_total, test_data_total)
    elif method == 'SVM':
        train_predict, test_predict = utils.SVM(train_data_total, train_label_total, test_data_total)

    acc_train, acc_test, confusion_matrix_train, confusion_matrix_test = \
        calculate_indices(train_predict, train_label_total, test_predict, test_label_total)
    predict_GT = np.zeros(shape=(610, 340))
    for iter1, index in enumerate(test_index_total):
        predict_GT[index[0], index[1]] = test_predict[iter1]

    return acc_train, acc_test, confusion_matrix_train, confusion_matrix_test, predict_GT


if __name__ == '__main__':
    image, GT = data_loading(parser.image_path, parser.GT_path)
    print('-----------------------------data loading finished-----------------------------')
    PCA_image, PCA_err = PCA(image, parser.PCA_dim)

    print('-----------------------------dimension reducing finished-----------------------------')

    feature_image = feature_extraction(PCA_image, parser.feature_extraction_method)
    print('-----------------------------feature extracting finished-----------------------------')

    label2index, class_num, per_class_num = calculate_class_num(GT)
    train_data, train_index, test_data, test_index = datasets_dividing(feature_image, label2index,
                                                                           class_num,per_class_num, parser.train_data_ratio)
    print('-----------------------------datasets dividing finished-----------------------------')

    acc_train, acc_test, confusion_matrix_train, confusion_matrix_test, predict_GT = classifier(train_data, test_data, test_index, parser.classify_method)
    print('-----------------------------classify finished-----------------------------')
    print(acc_train, acc_test)
    # plot #
    plt.figure()
    sns.heatmap(confusion_matrix_train / np.sum(confusion_matrix_train), annot=True, cmap='Blues', fmt='.1%')
    plt.title('train data confusion matrix')
    plt.figure()
    sns.heatmap(confusion_matrix_test / np.sum(confusion_matrix_test), annot=True, cmap='Blues', fmt='.1%')
    plt.title('test data confusion matrix')
    plt.figure(), plt.subplot(121), plt.imshow(predict_GT, cmap='viridis'), plt.colorbar()
    plt.subplot(122), plt.imshow(GT, cmap='viridis'), plt.colorbar()
    plt.show()