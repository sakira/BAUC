function [EBAUC] = getEmpericalBEEauc(X, y, model, th)
% FOR EACH LAMBDA CALCULATE Emperical BEE AUC

beeOpts = struct('covMode', 'general', 'prior', 'proper', 'std', 1);

w = model.w';


% t_min = min(th);
% t_max = max(th);
% t_min_new = t_min - (t_max - t_min);
% t_max_new = t_max + (t_max - t_min);
% t_new = linspace(t_min_new, t_max_new, 100);
% th = t_new;

[~, E1, E2] = getBinaryBeeError (X, y, w, -th, beeOpts);
FPR = E1;
TPR = 1-E2;
ROC_EBEE(:,1) = FPR;
ROC_EBEE(:,2) = TPR;

% for k = 1:length(t_new)
%     [~, E1, E2] = getBinaryBeeError (X, y, w, -t_new(k), beeOpts);
% 
%     FPR = E1;
%     TPR = 1-E2;
%     ROC_EBEE(k,:) = [FPR, TPR];
% end
% 
% 
% AUC
ROC_EBEE = flipud(ROC_EBEE);
EBAUC   = trapz(ROC_EBEE(:,1), ROC_EBEE(:,2));



            
            
end