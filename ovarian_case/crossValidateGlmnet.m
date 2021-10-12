[CV_AUC] = getCVauc(X, y, Kfold, model, th)

    % 1. CV-ROC
    % Simulate what CV-5 would do --- except the actual training

    K = 5; % Number of CV folds
    CV_part = cvpartition(y,'KFold',K);

    CV_AUC = [];

    for i = 1:CV_part.NumTestSets

        trIdx = CV_part.training(i);
        teIdx = CV_part.test(i);
        c1_idx = teIdx(1 : N);
        c2_idx = teIdx(N+1 : end);

        ROC_CV = zeros(length(th), 2);    

        for k = 1:length(th)
            p1_fold = p1(c1_idx);
            p2_fold = p2(c2_idx);

            FPR = nnz(p1_fold >= t(k)) / length(p1_fold);
            TPR = nnz(p2_fold >= t(k)) / length(p2_fold);
            ROC_CV(k,:) = [FPR, TPR];
        end

        ROC_CV = flipud(ROC_CV);
        CV_AUC = [CV_AUC; trapz(ROC_CV(:,1), ROC_CV(:,2))];

    end

    CV_AUC_5 = nanmean(CV_AUC);


end
