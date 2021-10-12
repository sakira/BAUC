function model = trainModel (Y, X, options)
    %addpath('P:\Users personal data\libraries\liblinear-2.11\windows\');
    %addpath('~/libraries/liblinear-2.11/matlab');
    %addpath('../libraries/liblinear-2.11/windows/');
    addpath('C:/Local/hassans/Programs/libraries/liblinear-2.11/windows/');
    X = sparse (X);
    model = train(Y, X, options); % use the same solver: -s 0

    
end