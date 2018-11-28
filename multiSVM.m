%% SVM

kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,1);
data_rand_features = data_rand(:,2:7);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);

noClass = 3;

% model = svm.train(xtrain, ytrain);
% predict = svm.predict(model, xtest);
% % disp([noClass, predict]);
% 
% Accuracy=mean(noClass==predict)*100;
% fprintf('\nAccuracy =%d\n',Accuracy)

results = multiClassSVM(xtrain,ytrain, xtest);
% disp(results);

resultsMapped = zeros(length(ytest),1);
for i=1:length(results)
   if results(i) == 1
       resultsMapped(i) = -1;
   elseif results(i) == 2
       resultsMapped(i) = 0;
   else
       resultsMapped(i) = 1;
   end
end

[confusionMatrix, order] = confusionmat(ytest', resultsMapped');
SVM_accuracy1 = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));

% m = ceil(n/10);
% k = 1:m:n-m;
% testingData = data_rand(k:k+m-1,:);
% trainingData = [data_rand(1:k-1,:); data_rand(k+m:end,:)];
% 
% Xtrain = trainingData(:, 1:3);
% Xtrain = [Xtrain trainingData(:, 5:7)];
% ytrain = trainingData(:,4);
% 
% Xtest = testingData(:, 1:3);
% Xtest = [Xtest testingData(:, 5:7)];
% ytest = testingData(:,4);
% options = statset('UseParallel',true);

% Mdl1 = fitcecoc(xtrain,ytrain,'Coding','onevsone');
% TestPredictionSVM1 = predict(Mdl1,xtest);
% [confusionMatrixSVM1, order] = confusionmat(ytest,TestPredictionSVM1);
% SVM_accuracy1 = sum(diag(confusionMatrixSVM1))/sum(sum(confusionMatrixSVM1));

% CVMdl = crossval(Mdl1);

% figure
% plotconfusion(ytest, TestPredictionSVM1);

% m = length(ytrain);  %Number of training examples
% d = size(xtrain,2); % Number of features.
% theta = zeros(d+1,1); % Initialize thetas to zero.
% % Choose some alpha value
% alpha = 0.01; 
% num_iters = 400;

%% SVM

kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,1);
data_rand_features = data_rand(:,2);
data_rand_features = [data_rand_features data_rand(:,4) data_rand(:,5)];
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);

noClass = 3;

% model = svm.train(xtrain, ytrain);
% predict = svm.predict(model, xtest);
% % disp([noClass, predict]);
% 
% Accuracy=mean(noClass==predict)*100;
% fprintf('\nAccuracy =%d\n',Accuracy)

results = multiClassSVM(xtrain,ytrain, xtest);
% disp(results);

resultsMapped = zeros(length(ytest),1);
for i=1:length(results)
   if results(i) == 1
       resultsMapped(i) = -1;
   elseif results(i) == 2
       resultsMapped(i) = 0;
   else
       resultsMapped(i) = 1;
   end
end

[confusionMatrix, order] = confusionmat(ytest', resultsMapped');
SVM_accuracy1 = sum(diag(confusionMatrix))/sum(sum(confusionMatrix));

% m = ceil(n/10);
% k = 1:m:n-m;
% testingData = data_rand(k:k+m-1,:);
% trainingData = [data_rand(1:k-1,:); data_rand(k+m:end,:)];
% 
% Xtrain = trainingData(:, 1:3);
% Xtrain = [Xtrain trainingData(:, 5:7)];
% ytrain = trainingData(:,4);
% 
% Xtest = testingData(:, 1:3);
% Xtest = [Xtest testingData(:, 5:7)];
% ytest = testingData(:,4);
% options = statset('UseParallel',true);

% Mdl1 = fitcecoc(xtrain,ytrain,'Coding','onevsone');
% TestPredictionSVM1 = predict(Mdl1,xtest);
% [confusionMatrixSVM1, order] = confusionmat(ytest,TestPredictionSVM1);
% SVM_accuracy1 = sum(diag(confusionMatrixSVM1))/sum(sum(confusionMatrixSVM1));

% CVMdl = crossval(Mdl1);

% figure
plotconfusion(ytest', resultsMapped');

% m = length(ytrain);  %Number of training examples
% d = size(xtrain,2); % Number of features.
% theta = zeros(d+1,1); % Initialize thetas to zero.
% % Choose some alpha value
% alpha = 0.01; 
% num_iters = 400;