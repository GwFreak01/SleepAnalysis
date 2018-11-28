%% Forward Feature Selection using Sleep Quality
kill
clc
clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');
feature_rank1 = zeros(size(data,2)-1,1);
% Create randomized 90/10 training/testing set
n = size(data, 1);
iters = 1000;
for i=1:iters
    data_rand = data(randperm(n),:);
    labels=data_rand(:,7);
    data_rand_features= data_rand(:,1:6);
    X = featureNormalize(data_rand_features);
    xtest = X(1:n-ceil(n*.9),:);
    xtrain = X(ceil(n*.9)+1:n,:);

    ytest = labels(1:n-ceil(n*.9), 1);
    ytrain = labels(ceil(n*.9)+1:n,1);

    ypred = classify(xtest, xtrain, ytrain);

    c = cvpartition(ytest,'k',10);
    opts = statset();
    f = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));
    %   
    [inmodel, history] = sequentialfs(f,xtest,ytest,'cv',c,'options',opts);
    feature_rank1 = feature_rank1 + inmodel';
    % Selects 2 & 4 most commonly, 5 and then 6
    % End of Sleep, HeartRate, Steps, Start, Duration

end

%% Forward Feature Selection using Mood
% kill
% clc
% clear
% Data organized as Start, End, DurationInMins, WakeupClass,
% HeartRate, Activity Steps, SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');
feature_rank2 = zeros(size(data,2)-1,1);
% Create randomized 90/10 training/testing set
n = size(data, 1);
iters = 10;
for i=1:iters
    data_rand = data(randperm(n),:);
    labels=data_rand(:,1);
    data_rand_features= data_rand(:,2:7);
    X = featureNormalize(data_rand_features);
    xtest = X(1:n-ceil(n*.9),:);
    xtrain = X(ceil(n*.9)+1:n,:);

    ytest = labels(1:n-ceil(n*.9), 1);
    ytrain = labels(ceil(n*.9)+1:n,1);

    c = cvpartition(ytest,'k',10);
    opts = statset();
    f = @(xtrain, ytrain, xtest, ytest) sum(ytest ~= classify(xtest, xtrain, ytrain));
    %   
    [inmodel, history] = sequentialfs(f,xtest,ytest,'cv',c,'options',opts);
    feature_rank2 = feature_rank2 + inmodel';
    % Selects 2 & 4 most commonly, 5 and then 6
    % End of Sleep, HeartRate, Steps, Start, Duration

end
