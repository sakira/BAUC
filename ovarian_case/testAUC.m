function max_AUC = testAUC(X_test, y_test, glmnetModel, maxLambda, posclass)

    
    
    yHat = glmnetPredict(glmnetModel, X_test, maxLambda, 'class');
    err = mean(yHat ~= repmat(y_test, [1, size(yHat, 2)]));
    
    yHat = glmnetPredict(glmnetModel, X_test, maxLambda, 'response');
        
    [~, ~, ~, max_AUC] = perfcurve(y_test, yHat, posclass);
    



end