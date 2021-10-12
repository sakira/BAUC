function [NBAUC] = getClosedBEEauc(X_train, y_train, model, lambdas)
% FOR EACH LAMBDA CALCULATE Non-emperical (closed form) BEE AUC

beeOpts = struct('covMode', 'general', 'prior', 'proper', 'std', 1);

NBAUC = zeros(length(lambdas),1);

for l = 1:length(lambdas)
    lambda = lambdas(l);
    
    w = model.beta(:,l);
    b = model.a0(l); % needed??
    yHat = X_train*w;
    trainingTh = unique(yHat);
    
    % compute BAUC (closed form BEE AUC)
    NBAUC(l) = getBinaryBeeClosedFormAUC (X_train, y_train, w, b, beeOpts);

end


end