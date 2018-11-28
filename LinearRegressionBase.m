%% Linear Regression
kill
clear
clc
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

% WakeUpClass	Start	End	DurationInHourMin	DurationInMins	HeartRate	Activity(Steps)	SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,7);
data_rand_features = data_rand(:,1:6);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);
    
m = length(ytrain);  %Number of training examples
d = size(xtrain,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

% Add intercept term to X


%%%%% All features
M1= xtrain;  

% reminder: number of columns of M <=30

w = ((M1'*M1)\M1')*ytrain;
avgSqErr = sum((ytest-xtest*w).^2)./length(xtest);
avgDevErr = sum(abs(ytest-xtest*w))./length(xtest);

% x1 = 0:.01:1;
% y1 = w*x1;
figure;
plot(ytest,xtest*w,'+');
hold on
% plot(x1, y1, '--g', 'LineWidth' ,3);
title(sprintf('Linear Regression Base w/ All Features\n AvgSqErr=%2.2f%%, AvgDevErr=%2.2f%%',avgSqErr*100,avgDevErr*100));
print -dpng LinearRegressionBaseErr_AllFeatures.png
eachSqErr = (ytest-xtest*w).^2;
h = hist(eachSqErr,0:.01:max(eachSqErr));


figure;
plot(0:.01:max(eachSqErr),h,'linewidth',3); grid on;

title(sprintf('Linear Regression Base w/ All Features Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng LinearRegressionBaseErr_AllFeaturesHistogram.png

%%
%%%%% Top 3 Features from Forward Feature Selection

clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

% WakeUpClass	Start	End	DurationInHourMin	DurationInMins	HeartRate	Activity(Steps)	SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,7);
data_rand_features = data_rand(:,1:6);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);

m = length(ytrain);  %Number of training examples
d = size(xtrain,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

% xtest = testingData(:, 1:6);
% ytest = testingData(:, 7);

% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
% X = featureNormalize(xtrain);
% xtest = featureNormalize(xtest);

% Add intercept term to X
xtrain = [ones(m,1) xtrain(:,2) xtrain(:,3) xtrain(:,4)];
% xtest = [ones(length(xtest),1) xtest];

%%%%% Top 3 Features

M2= xtrain; 

% xtest = X(1:n-ceil(n*.9),:);
xtest = [ones(length(xtest),1) xtest(:, 2) xtest(:, 3) xtest(:, 4)];

w = ((M2'*M2)\M2')*ytrain;
avgSqErr = sum((ytest-xtest*w).^2)./length(xtest);
avgDevErr = sum(abs(ytest-xtest*w))./length(xtest);

% x1 = 0:.01:1;
% y1 = w*x1;
figure;
plot(ytest,xtest*w,'+');
hold on
% plot(x1, x1*w, '--g', 'LineWidth' ,2);
title(sprintf('Linear Regression Base w/ Top 3 Features\n AvgSqErr=%2.2f%%, AvgDevErr=%2.2f%%',avgSqErr*100,avgDevErr*100));
print -dpng LinearRegressionBaseErr_Top3Features.png
eachSqErr = (ytest-xtest*w).^2;
h = hist(eachSqErr,0:.01:max(eachSqErr));


figure;
plot(0:.01:max(eachSqErr),h,'linewidth',3); grid on;
title(sprintf('Error w/ Top 3 Features Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng LinearRegressionBaseErr_Top3FeaturesHistogram.png
