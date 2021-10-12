function [CV_AUC] = getCV_AUC(X, y, Kfold, model, options, th)

    
    K = 5; % Number of CV folds
    cvIdx = crossvalind('Kfold', y, Kfold);
    
    

    CV_AUC = [];
    for fold = 1:Kfold
        fprintf('Training fold %d/%d...\n', fold, Kfold);
        
        % Divide the train data into K folds
        trIdx = (cvIdx ~= fold);
        teIdx = (cvIdx == fold);
        
        % separate the training and test data
        trX = X(trIdx, :); trY = y(trIdx);
        teX = X(teIdx, :); teY = y(teIdx);
        
        % Train the model with K-1 fold training data
        % trX = sparse (trX);
        foldModel = trainModel (trY, trX, options);
        % K-folds model weight
        w = foldModel.w';
%         yHat = teX * w; % response
        
        %teX = sparse(teX);
        %[predict_label, accuracy, prob_estimates] = predict(teY, teX, model, '-b 1');
%         [~, ~, ~, foldAUC] = perfcurve(teY, yHat, 2);
%         CV_AUC2 = [CV_AUC2; foldAUC];
%         
        
        c1_idx = (teY == 1); % class 1 index in test data
        c2_idx = (teY == 2); % class 2 index in test data
        teX1 = teX(c1_idx, :);
        teX2 = teX(c2_idx, :);
        
        p1 = teX1 * w; 
        p2 = teX2 * w;
                
        if mean(p1) > mean(p2)
            w = -1*w; % update the model parameter too.
            p1 = teX1 * w;
            p2 = teX2 * w;
        
        end
        
        ROC_CV = zeros(length(th), 2);    

        for k = 1:length(th)
            
            FPR = nnz(p1 >= th(k)) / length(p1); % number of false positive samples (FP)/P;
            TPR = nnz(p2 >= th(k)) / length(p2);
            ROC_CV(k,:) = [FPR, TPR];
        end

        ROC_CV = flipud(ROC_CV);
        CV_AUC = [CV_AUC; trapz(ROC_CV(:,1), ROC_CV(:,2))];

        
        
    end

    
    CV_AUC = nanmean(CV_AUC);


end
