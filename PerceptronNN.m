%% Perceptron Neural Network

% WakeUpClass	Start	End	DurationInHourMin	DurationInMins	HeartRate	Activity(Steps)	SleepQuality

data = xlsread('sleepdata_original_synthesized_sanitized.csv');

% Create randomized 90/10 training/testing set
n = size(data, 1);
data_rand = data(randperm(n),:);
labels = data_rand(:,1);

resultsMapped = zeros(length(labels),1);
for i=1:length(results)
   if labels(i) == 1
       resultsMapped(i) = 1;
   elseif results(i) == 0
       resultsMapped(i) = 0;
   else
       resultsMapped(i) = 0;
   end
end

% labels = resultsMapped;

data_rand_features = data_rand(:,2:7);
X = featureNormalize(data_rand_features);

xtest = X(1:n-ceil(n*.9),:);
xtrain = X(n-ceil(n*.9)+1:n,:);

ytest = labels(1:n-ceil(n*.9), 1);
ytrain = labels(n-ceil(n*.9)+1:n,1);

x = data_rand_features';
t = labels';
size(x)
size(t)
net = patternnet(10);
% view(net)
[net,tr] = train(net,x,t);
nntraintool

% plotperform(tr)
% 
% testX = x(:,tr.testInd);
% testT = t(:,tr.testInd);
% 
% testY = net(testX);
% testIndices = vec2ind(testY)
% 
% plotconfusion(testT,testY)
