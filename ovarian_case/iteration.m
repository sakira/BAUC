function iteration(iter)
% This file generates auc results of different sample size at each iteration
% The necessary steps are:
%     -	load the data ( in this case ovariancancer)
%     -	Convert the numerical values from single to double
%     -	Set the group values: Normal = 1 and Cancer = 2
%     -	set the positive class label ‘posclass’ to 2 for computing perfcurve()
%     -	set the necessary parameters. 
%         o	lambdas = 10.^(0:-0.1:-7)
%         o	Kfold = 5 (5-fold cv)
%         o	alpha = 1
%         o	perc = 0.1:0.05:0.95 ( sample size 10%, 15%, … 95% )
%         o	Th = 0.5 (Threshold level)
%     -	for each sample size p%
%         o	Divide the dataset into p% of training set using cvpartition() holdout method.
%         o	Separate the test dataset
%         o	Randomize/shuffle the training and test dataset
%         o	Normalize the training and test dataset with the given trained dataset parameters( mu and sigma)
%         o	Separate the trained and test dataset to ‘Normal’ and ‘Cancer’ dataset. We may need this separation for later experiment.
%         o	Train the classifier with the training dataset
%         o	Compute the AUC of 3 methods for given lambdas on training dataset
%         o	store the AUC values under folder ‘iteration/AUC/’
%         o	Find the lambda with maximum AUC, ignore the Inf value
%         o	Compute the maximum AUC value on the test dataset with the selected lambda for each method
%         o	Store the results under folder ‘iteration/’
%     -	end of forloop

% load data 
load ovariancancer


X = double(obs);
Y = +strcmp(grp, 'Cancer') + 1; % Cancer = class 2 (# = 121) & Normal = class 1 (# = 95)
posclass = max(unique(Y)); % positive class




seed = iter;
% rng(seed, 'twister'); % twister -- kind of generator


[N,nFeatures] = size (X);
% Randomly choose 90% of the training data: 
perc = 0.1:0.05:0.95;
nTrain = round (N * perc); % number of training data
nTest = N - nTrain; % number of test data
Kfold = 5; % for CV 
Th = 0.5; % threshold



for i = 1:length(perc) % start from 10% data as training
    
    disp(['For ' num2str(perc(i)) '% of samples, at iteration: ' num2str(iter) '\n']);

    % Define train data and test data
    % Divide the dataset into p% of training set.
    rng(seed, 'twister'); % twister -- kind of generator
    %holdoutCVP = cvpartition(Y,'holdout',nTest(i));
    holdoutCVP = cvpartition (Y, 'holdout', 1 - perc(i));
    X_train = X(holdoutCVP.training,:);
    y_train = Y(holdoutCVP.training);

    X_test = X(holdoutCVP.test,:);
    y_test = Y(holdoutCVP.test);

    % randomization
    [X_train, y_train] = shuffling(X_train, y_train);
    [X_test, y_test] = shuffling(X_test, y_test);

    % Normalize
    mu = mean(X_train);
    sigma = std(X_train);
    X_train = normalized(X_train,mu,sigma);
    X_test = normalized(X_test,mu,sigma);


    % define options
    liblinear_options = '-s 0 -c 1'; % L2-regularized

    % Learn classifier
    % X_train = sparse (X_train);
    % best = train(y_train, X_train, '-C -s 0');
    % liblinear_options = sprintf('-c %f -s 0', best(1));
    % model = train(y_train, X_train, liblinear_options); % use the same solver: -s 0

    model = trainModel (y_train, X_train, liblinear_options);


    % Project training data
    % separate the training data into two parts
    indices = (y_train == 1); % indices in the class c
    other_indices = ~indices;
    X1 = X_train(indices,:);
    X2 = X_train(other_indices,:);
    p1 = X1 * model.w';
    p2 = X2 * model.w';


    % Make sure p1 gets lower values than p2

    if mean(p1) > mean(p2)
        model.w = -1 * model.w; % update the model parameter too.
        w = model.w';
        p1 = X1 * w;
        p2 = X2 * w;
    end

    % Get all reasonable thresholds

    t = sort ([p1; p2]);

    % CALCULATE AUC FOR DIFFERENT METHODS 
    % CALCULATE CV AUC
    % CALCULATE BEE AUC
    % CALCULATE Closed form BEE AUC
    % CALCULATE TEST (ORACLE/TARGET) AUC


    % 1. CV_AUC
    [CV_AUC] = getCV_AUC (X_train, y_train, Kfold, model, liblinear_options, t);


    % 2. EBAUC
    %beeOpts = struct('covMode', 'general', 'prior', 'proper', 'std', 1);
    [EBAUC] = getEmpericalBEEauc (X_train, y_train, model, t);


    % 3. CBAUC
    beeOpts = struct('covMode', 'general', 'prior', 'proper', 'std', 1);
    [CBAUC] = getBinaryBeeClosedFormAUC (X_train, y_train, model.w', beeOpts);



    % Calculate testAUC for C AUC_TEST
    TeAUC = getTestAUC (X_test, y_test, model, posclass);


    % save AUC for each method
    AUC = [CV_AUC EBAUC CBAUC TeAUC];
    % store
    save(sprintf('iteration/AUC_iter%d_perc%d',iter, i), 'AUC');


end




end 



