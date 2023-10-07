% Load your feature vectors and labels here
% Replace 'X' with your feature vectors and 'y' with your labels
load('your_data.mat'); 

% Split the data into training and testing sets (e.g., 70% train, 30% test)
trainRatio = 0.7;
testRatio = 1 - trainRatio;
[Train, Test] = crossvalind('HoldOut', size(X, 1), trainRatio);

% Split the data
XTrain = X(Train, :);
yTrain = y(Train);
XTest = X(Test, :);
yTest = y(Test);

% Train the SVM classifier
svm = fitcsvm(XTrain, yTrain, 'KernelFunction', 'linear', 'Standardize', true);

% Predict on the test set
yPred = predict(svm, XTest);

% Evaluate the classifier's performance
accuracy = sum(yPred == yTest) / numel(yTest);
disp(['Accuracy: ' num2str(accuracy)]);

% You can also use other evaluation metrics like confusion matrix, precision, recall, etc.
% Example of confusion matrix:
C = confusionmat(yTest, yPred);
disp('Confusion Matrix:');
disp(C);
