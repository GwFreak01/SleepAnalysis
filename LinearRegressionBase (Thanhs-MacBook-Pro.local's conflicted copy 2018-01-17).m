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
testingData = data_rand(1:n-ceil(n*.85),:);
trainingData = data_rand(ceil(n*.85)+1:n,:);


xtrain = trainingData(:, 1:6);
ytrain = trainingData(:, 7);
m = length(ytrain);  %Number of training examples
d = size(xtrain,2); % Number of features.
theta = zeros(d+1,1); % Initialize thetas to zero.
% Choose some alpha value
alpha = 0.01; 
num_iters = 400;

xtest = testingData(:, 1:6);
ytest = testingData(:, 7);

% Scale features and set them to zero mean with std=1
% Write a function featureNormalize.m which computes
% the mean and std of X, then returns a normalized version % of X, where we substract the mean form each feature,
% then scale so that std dev = 1
X = featureNormalize(xtrain);
xtest = featureNormalize(xtest);

% Add intercept term to X
xtrain = [ones(m,1) xtrain];
xtest = [ones(length(xtest),1) xtest];

%Two sample solutions below, neither is good enough for points:
M1= xtrain;  %avgSqErr=0.0081
M2=[ones(length(xtrain),1) xtrain(:,3) xtrain(:,5) xtrain(:,6)];  %avgSqErr=0..0071
% reminder: number of columns of M <=30

w = ((M1'*M1)\M1')*ytrain;
avgSqErr=sum((ytest-M1*w).^2)./length(xtest)

avgDevErr=sum(abs(ytest-M1*w))./length(xtest)
figure;
plot(ytest,M1*w,'+');
title(sprintf('avgSqErr=%6.4f avDevErr=%6.4f',avgSqErr,avgDevErr));
print -dpng sleepDataErr.png
eachSqErr = (ytest-M1*w).^2;
h = hist(eachSqErr,0:.01:3);

figure;
plot(0:.01:3,h,'linewidth',3); grid on;
xlabel('Avg Square Error'); ylabel('# of occurances');
print -dpng sleepDataErrHist.png
