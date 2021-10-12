function AUC = getTestAUC(Xt, yt, model, posclass)

    
    % using matlab perfcurve
    yHat = Xt * model.w';
    [~, ~, ~, AUC] = perfcurve(yt, yHat, posclass);
    



end