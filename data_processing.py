import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torchmetrics


def read_file():
    src_file = pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')
    print("Initial data:\n", src_file.info())
    return src_file


# remove unwanted columns
def remove_columns(input_file):
    input_file.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    print("After removing columns: \n", input_file.info())
    return input_file


# function to check null values in df
def remove_null_val(input_data):
    # check any null values in the columns
    null_status = input_data.isnull().values.any()
    print("Null count: \n", input_data.isna().sum())
    if null_status:
        return True
    else:
        return False


def check_outliers(input_data):
    # we need to check any outlier values which may affect model accuracy. In our case we need to check for columns
    # 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary'
    column_name = list(input_data.columns)
    column_name.remove('Geography')
    column_name.remove('Gender')
    column_name.remove('HasCrCard')
    column_name.remove('IsActiveMember')
    column_name.remove('Exited')

    for name in range(len(column_name)):
        plt.subplot(3, 3, name + 1)
        plt.boxplot(input_data[column_name[name]].values)
        plt.xlabel(column_name[name])
        plt.plot()

    plt.tight_layout()
    plt.show()

    # get count of rows for outliers for the columns and drop those rows
    credit_score_outlier = list(input_data[input_data['CreditScore'] <= 395].index)
    print("Total Credit score outlier count: ", len(credit_score_outlier))
    for row in range(len(credit_score_outlier)):
        input_data.drop(index=credit_score_outlier[row], axis=0, inplace=True)

    age_outlier = list(input_data[input_data['Age'] >= 57].index)
    print("Total age outlier count: ", len(age_outlier))
    for row in range(len(age_outlier)):
        input_data.drop(index=age_outlier[row], axis=0, inplace=True)

    product_outlier = list(input_data[input_data['NumOfProducts'] >= 4].index)
    print("Total num of product outlier count: ", len(product_outlier))
    for row in range(len(product_outlier)):
        input_data.drop(index=product_outlier[row], axis=0, inplace=True)

    print("After dropping outliers:\n", input_data.info())

    # after dropping outliers
    for name in range(len(column_name)):
        plt.subplot(3, 3, name + 1)
        plt.boxplot(input_data[column_name[name]].values)
        plt.xlabel(column_name[name])
        plt.plot()

    plt.tight_layout()
    plt.show()

    return input_data


def normalize_data(input_data):
    # convert columns Geography and Gender to numeric values using pandas.Dummies
    # first create dummies
    geo_dummy = pd.get_dummies(input_data['Geography'], dtype=int, prefix='geography_')
    gender_dummy = pd.get_dummies(input_data['Gender'], dtype=int, prefix='gender_')
    # concat with main df
    input_data = pd.concat([input_data, geo_dummy, gender_dummy], axis=1)
    input_data.drop(['Geography', 'Gender'], axis=1, inplace=True)
    print("After dummies normalization: \n", input_data.info())

    return input_data


def standardize_data(x_tr, x_te, y_tr, y_te):
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_tr)
    x_test_scaled = scalar.transform(x_te)

    # print("Values x train after standardization:\n", x_train_scaled)
    # print("Values x test after standardization:\n", x_test_scaled)

    return x_train_scaled, x_test_scaled


def knn_model(x_tr, x_te, y_tr, y_te):
    # select a range for n neighbors for KNN
    score_list = {}
    average_knn_score = 0.0
    total_knn_score = 0.0
    for n in np.arange(1, 25):
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(x_tr, y_tr)
        knn.predict(x_te)
        # predict method will determine the class/label of the output predicted
        # print("Predictions: \n", len(predicted_val))
        accuracy_score = knn.score(x_te, y_te)
        score_list.update({n: accuracy_score})
        total_knn_score += accuracy_score

    print('KNN: Average Accuracy Score: ', total_knn_score/25)

    # print(sorted_score)
    for key in score_list:
        plt.bar(key, score_list[key])
        plt.xlabel('N-neighbors')
        plt.ylabel('Accuracy Score')
        plt.title('KNN Accuracy Graph')

    plt.show()


def logistic_model(x_tr, x_te, y_tr, y_te):
    logreg = LogisticRegression()
    logreg.fit(x_tr, y_tr)
    logreg.predict(x_te)
    prob = logreg.predict_proba(x_te)[:, 1]
    #print("Probabilities: ", prob)
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_te, prob)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    plt.title('ROC curve')
    plt.show()

    # get area under curve
    print("Logistic Regression: Area under curve: ", roc_auc_score(y_te, prob))


def deepl_learning_model(x_tr, x_te, y_tr, y_te):
    # reshaping the target values for processing, 1D to 2D numpy array
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)
    # create dataset
    dataset = TensorDataset(torch.tensor(x_tr).to(torch.float32), torch.tensor(y_tr).to(torch.float32))
    validation_dataset = TensorDataset(torch.tensor(x_te).to(torch.float32), torch.tensor(y_te).to(torch.float32))
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=50, shuffle=True)
    # create linear layer model, using Sigmoid() activation function for non linearity and Dropout() to reduce over
    # fitting
    linear_layer = nn.Sequential(nn.Linear(13, 9),
                                 nn.Linear(9, 6),
                                 nn.Linear(6, 1),
                                 nn.Sigmoid(),
                                 nn.Dropout(p=0.5)
                                 )
    # loss function initialization
    criterion = nn.MSELoss()
    # optimizer to tune hyperparameters like learning rate and weigh decay for reducing over fitting
    optimizer = optim.SGD(linear_layer.parameters(), lr=0.001, weight_decay=0.0001)
    # metric to check accuracy
    metric = torchmetrics.classification.BinaryAccuracy()

    training_loss = 0.0
    validation_loss = 0.0
    loss_list = []
    test_loss_list = []
    training_epoch_list = []
    test_epoch_list = []
    # running 3 epochs for better learning
    for epoch in range(3):
        for data in dataloader:
            # zeroing gradients before prediction
            optimizer.zero_grad()
            # load data
            feature, target = data
            # predict using linear layer model
            pred = linear_layer(feature)
            # calculate loss between prediction and real values
            loss = criterion(pred, target)
            # compute gradients
            loss.backward()
            # update gradients according to the loss
            optimizer.step()
            # again setting gradients to 0 for efficient loss calculation
            optimizer.zero_grad()
            # calculate total training loss
            training_loss = training_loss + loss.item()
            metric.update(pred, target)
        # calculate mean training loss after each epoch
        mean_training_loss = training_loss / len(dataloader)
        loss_list.append(mean_training_loss)
        # compute accuracy after each epoch
        accuracy = metric.compute()
        training_epoch_list.append(accuracy)

        # after training, we need to calculate test loop
        # put model into evaluation mode
        linear_layer.eval()
        # making gradients 0
        with torch.no_grad():
            # same steps as training loop
            for feature, target in validation_dataloader:
                validation_pred = linear_layer(feature)
                loss = criterion(validation_pred, target)
                validation_loss = validation_loss + loss.item()

        mean_validation_loss = validation_loss / len(validation_dataloader)
        test_loss_list.append(mean_validation_loss)
        test_epoch_list.append(mean_validation_loss)

        linear_layer.train()
    print("Deep Learning model:")
    print("Training loss for each epoch: ", loss_list)
    print("Training accuracy for each epoch: ", training_epoch_list)

    print("Test loss for each epoch: ", test_loss_list)
    print("Test accuracy for each epoch: ", test_epoch_list)

    plt.plot([1, 2, 3], training_epoch_list, label='TRaining Accuracy', marker='o')
    plt.plot([1, 2, 3], test_epoch_list, label='TRaining Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def process_data():
    # read data from the file
    input_data = read_file()

    # based on the data, we do not need columns 'RowNumber','CustomerId' and 'Surname'
    input_data_removed_col = remove_columns(input_data)

    # check null values
    null_status_return = remove_null_val(input_data_removed_col)

    # remove null values if present, in this case there is none so no need to do anything

    # check for outliers
    input_data_after_outlier = check_outliers(input_data_removed_col)

    # normalize data
    normalized_data = normalize_data(input_data_after_outlier)

    features = normalized_data.drop(['Exited'], axis=1).values
    target = normalized_data['Exited'].values

    # split the data
    X_train, x_test, Y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=21)

    # standardize data
    x_train_val, x_test_val = standardize_data(X_train, x_test, Y_train, y_test)

    # apply K-Means
    knn_model(x_train_val, x_test_val, Y_train, y_test)

    # apply logistic regression
    logistic_model(x_train_val, x_test_val, Y_train, y_test)

    # apply deep learning with Sigmoid function
    deepl_learning_model(x_train_val, x_test_val, Y_train, y_test)


process_data()
