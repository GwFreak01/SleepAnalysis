%% Final Run Code
kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

% WakeUpClass	Start	End	DurationInHourMin	DurationInMins	HeartRate	Activity(Steps)	SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);

data_rand = data(randperm(n),:);
labels_SleepQuality = data_rand(:,7);
labels_Mood = data_rand(:,1);
data_rand_features = data_rand(:,1:6);
data_rand_featuresMood = data_rand(:,2:7);
X = featureNormalize(data_rand_features);
XMood = featureNormalize(data_rand_featuresMood);

ytest = labels_SleepQuality(1:n-ceil(n*.9), 1);
ytrain = labels_SleepQuality(n-ceil(n*.9)+1:n,1);

ytestMood = labels_Mood(1:n-ceil(n*.9), 1);
ytrainMood = labels_Mood(n-ceil(n*.9)+1:n,1);

xtest1Mood = XMood(1:n-ceil(n*.9),:);
xtest2Mood = [xtest1Mood(:, 2) xtest1Mood(:, 3) xtest1Mood(:, 4)];

xtrain1Mood = XMood(n-ceil(n*.9)+1:n,:);
xtrain2Mood = [xtrain1Mood(:,2) xtrain1Mood(:,3) xtrain1Mood(:,4)];




m = length(ytrain);  %Number of GT training examples

xtest1 = X(1:n-ceil(n*.9),:);
xtest2 = [ones(length(xtest1),1) xtest1(:, 2) xtest1(:, 3) xtest1(:, 4)];

xtrain1 = X(n-ceil(n*.9)+1:n,:);
xtrain2 = [ones(m,1) xtrain1(:,2) xtrain1(:,3) xtrain1(:,4)];



% xtest1Mood = X(1:n-ceil(n*.9),:);
% xtest2 = [ones(length(xtest1Mood),1) xtest1Mood(:, 2) xtest1Mood(:, 3) xtest1Mood(:, 4)];
% 
% xtrain1 = X(n-ceil(n*.9)+1:n,:);
% xtrain2 = [ones(m,1) xtrain1(:,2) xtrain1(:,3) xtrain1(:,4)];


M1 = xtrain1;

M2 = xtrain2;


w1 = ((M1'*M1)\M1')*ytrain;
w2 = ((M2'*M2)\M2')*ytrain;

avgSqErr1 = sum((ytest-xtest1*w1).^2)./length(xtest1);
avgDevErr1 = sum(abs(ytest-xtest1*w1))./length(xtest1);
avgSqErr2 = sum((ytest-xtest2*w2).^2)./length(xtest2);
avgDevErr2 = sum(abs(ytest-xtest2*w2))./length(xtest2);


x1 = 0:.01:1;
% y1 = x1*w;
figure;
plot(ytest,xtest1*w1,'+');
hold on
plot(ytest,xtest2*w2,'o');
% plot(x1, y1, '--g', 'LineWidth' ,3);
legend('All Features','Top 3 Features');
title(sprintf('Least-Squares Fit \n All: AvgSqErr=%2.2f%%, AvgDevErr=%2.2f%%\n Top 3: AvgSqErr=%2.2f%%, AvgDevErr=%2.2f%%',avgSqErr1*100,avgDevErr1*100,avgSqErr2*100,avgDevErr2*100));
print -dpng LSFErr.png
eachSqErr1 = (ytest-xtest1*w1).^2;
h1 = hist(eachSqErr1,0:.01:max(eachSqErr1));

eachSqErr2 = (ytest-xtest2*w2).^2;
h2 = hist(eachSqErr2,0:.01:max(eachSqErr2));

figure;
plot(0:.01:max(eachSqErr1),h1,'linewidth',3, 'Color', 'red'); 
hold on
plot(0:.01:max(eachSqErr2),h2,'linewidth',3, 'Color', 'blue'); 
legend('All Features','Top 3 Features');
grid on;

title(sprintf('Least-Squares Fit Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng LSFErr_Histogram.png

%%

d1 = size(xtrain1,2); % Number of features.
d2 = size(xtrain2,2); % Number of features.
theta1 = zeros(d1,1); % Initialize thetas to zero.
theta2 = zeros(d2,1); % Initialize thetas to zero.

% Choose some alpha value
alpha = 0.01; 
num_iters = 400;


% xtrain1 = [ones(m,1) xtrain1];


[theta1, J_history1] = gradientDescentMulti(xtrain1, ytrain,theta1, alpha, num_iters);
[theta2, J_history2] = gradientDescentMulti(xtrain2, ytrain,theta2, alpha, num_iters);


avgSqErr1 = sum((ytest-xtest1*theta1).^2)./length(xtest1)
avgDevErr1 = sum(abs(ytest-xtest1*theta1))./length(xtest1)
avgSqErr2 = sum((ytest-xtest2*theta2).^2)./length(xtest2);
avgDevErr2 = sum(abs(ytest-xtest2*theta2))./length(xtest2);





figure;
plot(ytest,xtest1*theta1,'+');
hold on
plot(ytest,xtest2*theta2,'o');
legend('All Features','Top 3 Features');
title(sprintf('Gradient Descent \n All: AvgSqErr=%2.2f%% AvgDevErr=%2.2f%%\n Top 3: AvgSqErr=%2.2f%% AvgDevErr=%2.2f%%',avgSqErr1*100,avgDevErr1*100,avgSqErr2*100,avgDevErr2*100));
print -dpng GradientDescentErr.png

eachSqErr1 = (ytest-xtest1*theta1).^2;
h1 = hist(eachSqErr1,0:.01:max(eachSqErr1));
eachSqErr2 = (ytest-xtest2*theta2).^2;
h2 = hist(eachSqErr2,0:.01:max(eachSqErr2));

figure;
plot(0:.01:max(eachSqErr1),h1,'linewidth',3, 'Color', 'red');
hold on
grid on;
plot(0:.01:max(eachSqErr2),h2,'linewidth',3, 'Color', 'blue');
legend('All Features','Top 3 Features');
title(sprintf('Gradient Descent Error Histogram'));
xlabel('Average Square Error'); ylabel('# of Occurances');
print -dpng GradientDescentErr_Histogram.png

%% 


[B1, FitInfo1] = lasso(xtrain1, ytrain, 'cv', 5);
[B2, FitInfo2] = lasso(xtrain2, ytrain, 'cv', 5);
% ax1 = lassoPlot(B1, FitInfo1, 'PlotType', 'CV');
% hold on
ax2 = lassoPlot(B2, FitInfo2, 'PlotType', 'CV');
print -dpng LassoFitCV.png


lambda = .001;

% Init Theta and Run Gradient Descent
[theta1, J_history1] = gradientDescentMultiReg(xtrain1, ytrain, theta1, alpha, num_iters, lambda);
[theta2, J_history2] = gradientDescentMultiReg(xtrain2, ytrain, theta2, alpha, num_iters, lambda);

% 
% fprintf('Regularized Gradient Descent: [%f,%f]\n',theta, J_history);
% 
avgSqErr1 = sum((ytest-xtest1*theta1).^2)./length(xtrain1)
avgDevErr1 = sum(abs(ytest-xtest1*theta1))./length(xtrain1)
avgSqErr2 = sum((ytest-xtest2*theta2).^2)./length(xtrain2)
avgDevErr2 = sum(abs(ytest-xtest2*theta2))./length(xtrain2)



figure;
plot(ytest,xtest1*theta1,'+');
hold on
plot(ytest,xtest2*theta2,'+');
legend('All Features','Top 3 Features');
title(sprintf('Gradient Descent w/ Regularization\nAll: AvgSqErr=%2.2f%% AvgDevErr=%2.2f%%\n Top 3: AvgSqErr=%2.2f%% AvgDevErr=%2.2f%%',avgSqErr1*100,avgDevErr1*100,avgSqErr2*100,avgDevErr2*100));
print -dpng GradientDescentRegErr.png


figure;
plot(1:num_iters,J_history1,'LineWidth',3, 'Color', 'red');
title('All Features: Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;

% hold on;

figure;
plot(1:num_iters,J_history2,'LineWidth',3, 'Color', 'blue');

% legend('All Features','Top 3 Features');
title('Top 3 Features: Cost over 400 iterations')
xlabel('Number of Iterations'), ylabel('Cost'); grid on;
print -dpng GradientDescentCostTop.png

%% multiSVM
kill;

Model1=svm.train(xtrain1Mood,ytrainMood);
predict1=svm.predict(Model1,xtest1Mood);
[confusionMatrixSVM1, order1] = confusionmat(ytestMood,predict1);

Model2=svm.train(xtrain2Mood,ytrainMood);
predict2=svm.predict(Model2,xtest2Mood);
[confusionMatrixSVM2, order2] = confusionmat(ytestMood,predict2);


figure;
subplot(1,2,1);
plotConfMat(confusionMatrixSVM1, order1);
% title('All Features');
% print -dpng svmConfuseAll.png
subplot(1,2,2);
% figure;
plotConfMat(confusionMatrixSVM2, order2);
% print -dpng svmConfuseTop3.png


% 
% for i=1:length(results2)
%    if results2(i) == 1
%        resultsMapped2(i) = -1;
%    elseif results2(i) == 2
%        resultsMapped2(i) = 0;
%    else
%        resultsMapped2(i) = 1;
%    end
% end
% 
% [confusionMatrix1, order1] = confusionmat(ytest', resultsMapped1');
% SVM_accuracy1 = sum(diag(confusionMatrix1))/sum(sum(confusionMatrix1));
% [confusionMatrix2, order2] = confusionmat(ytest', resultsMapped2');
% SVM_accuracy2 = sum(diag(confusionMatrix2))/sum(sum(confusionMatrix2));
% 
% z1 = plotconfusion(ytest', resultsMapped1');
% z2 = plotconfusion(ytest', resultsMapped2');

%%



x = data_rand_featuresMood';
t = labels_Mood';
size(x)
size(t)
net = patternnet(10);
% view(net)
[net,tr] = train(net,x,t);
nntraintool


