function iterationUnbalanced(iter)
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
%         o	unbalancedPartition = .1:.1:.95 ( sample size 10%, 20%, … 90% )
%         o	In order to have unbalanced data distribution, for example, 
%             100% non-cancerous data and 10% cancer data from the selected dataset (#190), we have 95 non-cancerous data and 11 cancer data. 
%             Thus the total samples size is 106. Based on this total samples, we compute the non-cancerous sample sizes to be 
%             95,    85,   74,    64,    53,    42,    32,    21,    11 ( May need more explanation??)
%         o	Th = 0.5 (Threshold level)
%     -	for each partition size p
%         o	Randomly choose p% data from cancerous data and (106-p%) from noncancerous data
%         o	From newly generated dataset, split to train and test data ( starified splitting of 90% training set and 10% test set )
%         o	Randomize/shuffle the training and test dataset
%         o	Normalize the training and test dataset with the given trained dataset parameters( mu and sigma)
%         o	Train the classifier with the training dataset
%         o	Compute the AUC of 3 methods for given lambdas on training dataset
%         o	store the AUC values under folder ‘iterationUnbalanced/AUC/’
%         o	Find the lambda with maximum AUC, ignore the Inf value
%         o	Compute the maximum AUC value on the test dataset with the selected lambda for each method
%         o	Store the results under folder ‘iterationUnbalanced/’
%     -	end of forloop


% load data 
load ovariancancer
%load unbalancedDataset/ovariancancer9

X = double(obs);
Y = +strcmp(grp, 'Cancer') + 1; % Cancer = class 2 (# = 121) & Normal = class 1 (# = 95)
posclass = max(unique(Y)); % positive class

% X = newX;
% Y = newY; % Cancer = class 2 (# = 121) & Normal = class 1 (# = 95)
% posclass = max(unique(Y)); % positive class
% 


seed = iter;
% rng(seed, 'twister'); % twister -- kind of generator


[N,nFeatures] = size(X);
% Randomly choose 90% of the training data: 
unbalancedPartition = .1:.1:.95;

% Separate the classes
indices = (Y == 1); % indices in the class 1 (Normal)
other_indices = ~indices; % indices in the class 2 (Cancer)
X1 = X(indices,:); Y1 = Y(indices); % class 1
X2 = X(other_indices,:); Y2 = Y(other_indices); % class 2



% For example: 10% from class 1 and 90% from class 2
N1 = size(Y1,1);
N2 = size(Y2,1);
N = N1 + N2;

% throw N2 - N1 sample from class 2
rng(seed, 'twister');
idx = randperm(N2,N1);
newX2 = X2(idx,:); newY2 = Y2(idx);
X2 = newX2; Y2 = newY2;

N2 = size(Y2,1);
N = N1 + N2;


N = 106;
nTrain = N - round(N * unbalancedPartition); % number of training data
nTest = N - nTrain; % number of test data
Kfold = 5; % for CV 
Th = 0.5; % threshold

% add library paths
% addpath('~/glmnet2/');
%addpath('P:\Users personal data\phd materials\libraries\glmnet_matlab\glmnet_matlab');


% set parameters

%options.lambda = 10.^(7:-0.1:-7);
%lambdas = 10.^(0:-0.1:-7);
%options = glmnetSet;
%options.alpha = 1;
%options.lambda = lambdas; % set the given lambda values;


for i = 1:length(unbalancedPartition) % start from 10% data as training
        
        
        partition = unbalancedPartition(i);

        
        disp(['For ' num2str(partition) '% of samples, at iteration: ' num2str(iter) '\n']);
        
        % Dissimate the  data from the original data
        %n1 = round(N1 * partition);
        %n2 = round(N2 * (1 - partition));
        %n1 = round((N1 - N1 * (1 - partition))/partition); % Normal
        %n2 = round((N2 - N2 * (1 - partition))/(1-partition)); % cancer
        n1 = nTrain(i);
        n2 = N - n1;

        rng(seed, 'twister'); % twister -- kind of generator
        indices1 = randperm(N1, n1); 
        indices2 = randperm(N2, n2); 
        newX1 = X1(indices1,:); newY1 = Y1(indices1);
        newX2 = X2(indices2,:); newY2 = Y2(indices2);
        
        newX = [newX1;newX2]; newY = [newY1;newY2];
        
        % Split to train and test data ( starified splitting of 90%
        % training set and 10% test set
        ind = cvpartition(newY, 'HoldOut', .1);
        X_train = newX(ind.training,:); y_train = newY(ind.training);
        X_test = newX(ind.test,:); y_test = newY(ind.test);
        
        %X_train = [newX1;newX2]; y_train = [newY1;newY2];
        
        % Define test data
        %testX1 = X1; testY1 = Y1;
        %testX1(indices1,:) = []; testY1(indices1) = [];
        %testX2 = X2; testY2 = Y2;
        %testX2(indices2,:) = []; testY2(indices2) = [];
        
        %X_test = [testX1; testX2]; y_test = [testY1;testY2];
        
        % randomization
        [X_train, y_train] = shuffling(X_train, y_train);
        [X_test, y_test] = shuffling(X_test, y_test);

        
        % Normalize
        mu = mean(X_train);
        sigma = std(X_train);
        X_train = normalized(X_train,mu,sigma);
        X_test = normalized(X_test,mu,sigma);
                
        % For testing comment it later
        trainClass1 = (sum(y_train==1))/length(y_train);
        trainClass2 = (sum(y_train==2))/length(y_train);
%         size(X_train);
%         size(X_test);
%         unique(y_test);
%         sum(y_train==1)
%         sum(y_train==2)
%         
%         sum(y_test==1)
%         sum(y_test==2)
        
%         lambdas = 10.^(0:-0.1:-7);
%         options = glmnetSet;
%         options.alpha = 1;
%         options.lambda = lambdas; % set the given lambda values;
% 
% 
%         % Learn classifier
%         % full model: train the full model with P% samples
%         tic;
%             glmnetModel = glmnet(X_train, y_train, 'binomial',options);
%         % save time for full model
%         full_model_time = toc;
%         
        
        
        
        % CALCULATE AUC FOR DIFFERENT METHODS 
        % FOR EACH LAMBDA:
            % CALCULATE BEE AUC
            % CALCULATE PARAM AUC
            % CALCULATE PARAM(JUSSI) AUC
            % CALCULATE CV AUC
            % CALCULATE TEST (ORACLE/TARGET) AUC
        
            
            
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
    Xtr1 = X_train(indices,:);
    Xtr2 = X_train(other_indices,:);
    p1 = Xtr1 * model.w';
    p2 = Xtr2 * model.w';
    
        
            
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
    save(sprintf('iterationIm/AUC_iter%d_perc%d',iter, i), 'AUC');

        
        
 
        
       

end

    

end 



