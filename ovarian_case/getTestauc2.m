function [TAUC] = getTestauc(X, Y, model, lambdas)
% FOR EACH LAMBDA CALCULATE test AUC


TAUC = zeros(length(lambdas),1);
posclass = max(unique(Y)); % positive class

for l = 1:length(lambdas)
    lambda = lambdas(l);
    TAUC(l) = testAUC(X, Y, model, lambda, posclass);
    
end


end