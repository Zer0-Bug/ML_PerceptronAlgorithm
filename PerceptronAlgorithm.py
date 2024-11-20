import numpy as np
import pandas as pd


# This function trains the Perceptron model on the given training data
def Perceptron(trainData, alfa, epochs):

    X = trainData.iloc[:, 1:-1].values  # Extracting features from columns (ignoring subject id and class)
    y = trainData.iloc[:, -1].values    # Extracting class labels (the last column)

    #print("X HEAD", X)
    #print("Y HEAD", y)

    weights = np.zeros(X.shape[1])  # Initialize weights to zero for each feature
    bias = 0

    # Iterate over the training data for a fixed number of epochs (1000)
    for _ in range(epochs):

        # For each sample in the training data
        for i in range(len(X)):

            # Calculate the prediction by multiplying the features with weights and adding
            prediction = np.dot(X[i], weights) + bias
            
            if prediction < 0:
                predictionClass = 2
            else:
                predictionClass = 4


            # w = w + alfa.(gercekDeger - tahmin).x
            # bias = bias + alfa.(gercekDeger - tahmin)

            # If the predicted class is different from the actual class
            if predictionClass != y[i]:
                error = y[i] - predictionClass
                weights += alfa * error * X[i]
                bias += alfa * error

    return weights, bias



# This function makes a prediction for each sample based on learned weights and bias
def Predict(X, weights, bias):
    
    predict = np.dot(X, weights) + bias

    if predict < 0:
        return 2
    else:
        return 4



# This function tests the model on the test dataset and returns the predictions
def Test(testData, weights, bias):

    # Extract features from the test data (excluding subject id and class label)
    testX = testData.iloc[:, 1:-1].values

    # List to store predictions for each test sample
    testPredictions = []

    # For each sample in the test set, use the Predict function to make predictions
    for i in range(len(testX)):
        testPrediction = Predict(testX[i], weights, bias)
        testPredictions.append(testPrediction)
    
    return testPredictions



# Fill the empty Class column in TESTData with the Predictions I found
def Save(testData, predictions):

    updatedTESTData = testData.copy()

    updatedTESTData['Class'] = predictions

    save = input("\nDo you want to save the updated data to 'Updated_TESTData.xlsx'? (y/n): ").strip().lower()

    if save == 'y':
        updatedTESTData.to_excel("Updated_TESTData.xlsx", index=False)
        print("\nFile saved successfully as 'Updated_TESTData.xlsx'.")
    else:
        print("\nFile was not saved.\n")


def main():

    file = "DataForPerceptron.xlsx"

    trainData = pd.read_excel(file, sheet_name='TRAINData')
    testData = pd.read_excel(file, sheet_name='TESTData')

    """
    print("Train Data:")
    print(trainData.head())

    print("\nTest Data:")
    print(testData.head())
    """

    weights, bias = Perceptron(trainData, alfa = 0.1, epochs=1000)
    #print(f"Weights {weights} ---- Bias {bias}")

    predictions = Test(testData, weights, bias)
    print("\nPredictions:", predictions)

    Save(testData, predictions)


main()



#######  Written by  #######
######    Zer0-Bug    ######