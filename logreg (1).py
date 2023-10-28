import pandas as pd
from Logistic_Regression import LogReg
from Bag_Of_Words import BagOfWords
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('spam_ham_dataset.csv')

    # Bag of words model
    '''
    This code of task 3 use logistic regression and a bag-of-words. It reads a dataset of emails labeled as either 
    spam or ham, then preprocesses the text data by creating a bag-of-words model, 
    which represents the frequency of each word in the text as a feature vector.
    '''
    print('Building bag of words model')
    bw = BagOfWords(df['text'].values)
    X = bw.encode_documents(df['text'].values)
    y = df['label_num'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    print('train shape', X_train.shape)
    print('test shape', X_test.shape)

    logreg = LogReg()
    logreg.fit(X_train, y_train, learning_rate=0.01, epochs=1500, verbose=True, print_every_n=500)

    # Calculate accuracy on train and test set
    y_train_pred = logreg.predict(X_train, threshold=0.5)
    train_acc = logreg.accuracy(y_train_pred, y_train)

    y_test_pred = logreg.predict(X_test, threshold=0.5)
    test_acc = logreg.accuracy(y_test_pred, y_test)

    print("Qustion 3")
    print(f'Train accuracy: {train_acc:.3f}')
    print(f'Test accuracy: {test_acc:.3f}')
    print(f'Weights vector: {logreg.weights()}')
    print("=======================================================================\n")

    # ROC curve
    '''
    The code generates a Receiver Operating Characteristic (ROC) curve for the logistic regression model, 
    which is a graphical representation of the trade-off between the true positive rate (TPR)
    and false positive rate (FPR) for different threshold values.
    '''
    fpr = []
    tpr = []
    tt = []
    for t in range(1, 10):
        threshold = (5 * t) / 100.0
        m = logreg.confusion_matrix(X_test, y_test, threshold=threshold)
        true_positive_rate = m[0][0] / (m[0][0] + m[0][1])
        false_positive_rate = m[1][0] / (m[1][0] + m[1][1])
        fpr.append(false_positive_rate)
        tpr.append(true_positive_rate)
        tt.append(threshold)

    plt.title('Receiver Operating Characteristic')
    plt.scatter(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    for i in range(len(tt)):
            plt.annotate(tt[i], (fpr[i], tpr[i]))
    plt.legend()
    plt.show()

    print("Qustion 4")
    print("Prediction with optimal threshold")
    thr = 0.45
    acc = logreg.accuracy(logreg.predict(X_test, thr), y_test)
    print(f'thr = {thr}, acc = {acc}')
    print("the chosen thr is 0.45 becouse it is the closet point to (0,1) which is where the TP is the hiest and the FP is lowest")









