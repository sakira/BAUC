% check the number of training and testing samples

% load data 
load ovariancancer


X = double(obs);
Y = +strcmp(grp, 'Cancer') + 1; % Cancer = class 2 (# = 121) & Normal = class 1 (# = 95)
posclass = max(unique(Y)); % positive class



iter = 1;
seed = iter;
% rng(seed, 'twister'); % twister -- kind of generator


[N,nFeatures] = size (X);
% Randomly choose 90% of the training data: 
perc = 0.1:0.05:0.95;
nTrain = round (N * perc); % number of training data
nTest = N - nTrain; % number of test data
Kfold = 5; % for CV 
Th = 0.5; % threshold

nTr = [];
nTe = [];

for i = perc
    disp(['For ' num2str(i) '% of samples, at iteration: ' num2str(iter) '\n']);

    % Define train data and test data
    % Divide the dataset into p% of training set.
    rng(seed, 'twister'); % twister -- kind of generator
    %holdoutCVP = cvpartition(Y,'holdout',nTest(i));
    holdoutCVP = cvpartition (Y, 'holdout', 1 - i);
    X_train = X(holdoutCVP.training,:);
    y_train = Y(holdoutCVP.training);

    X_test = X(holdoutCVP.test,:);
    y_test = Y(holdoutCVP.test);
    
    nTr = [nTr size(X_train,1)];
    nTe = [nTe size(X_test,1 )];
    disp(['-----'])

end

[nTrain' nTest' nTr' nTe']


%%


nTrain = ceil (N * perc); % number of training data
nTest = N - nTrain; 
[nTrain' nTest' nTr' nTe']
