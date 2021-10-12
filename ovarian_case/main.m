%% This is the main script for ovarian cancer dataset that is used in MLSP paper
% ChangeLog: 27.04.2018
%%

iters = 3;
for iter = 1:iters
    iteration(iter);
end

%%

clear 
clc
iters = 1000;
samples = 10:5:95; % the percentage of samples.

%iters = 5;

AUCs = cell(length(samples),1);

for k = 1:length(samples)
    AUC_iter = zeros(iters,4);
    
    for iter = 1:iters
        
        % open file of each iteration
        try
            fileName  = sprintf('iteration/AUC_iter%d_perc%d',iter,k);
            load(fileName);
            AUC_iter(iter, :) = AUC;
        catch 
            disp(['Missing iteration ' num2str(iter) ' ...']); 
            
        end
        
        
        disp(['Data loaded successfully at Iteration: ' num2str(iter) ' for ' num2str(samples(k)) '% samples ...']); 
        
    end
    
    AUCs{k} = AUC_iter;
    
end

disp(['Finish loading data ...']);
save('AUCs.mat', 'AUCs', 'samples');

%% Plot
% close all
% clear all
clc
load AUCs


N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.



for k = 1:length(samples)
    AUC = AUCs{k};
    CV_AUC(k) = mean(AUC(:,1)); 
    EBAUC(k) = mean(AUC(:,2)); % emperical BEE AUC
    CBAUC(k) = mean(AUC(:,3)); % closed BEE AUC
    TAUC(k) = mean(AUC(:,4)); % Test AUC
    
 
end


% cross-validation: 'ro:'  
% emperical BEE: 'b+-.'
% closed form BEE: 'gx--'
% test: 'k*-'

% cross-validation: 'bo:'  
% closed form BEE: 'r+-'

figure;
hold on
plot(K, CV_AUC, 'ro:', 'LineWidth', 2);
plot(K, EBAUC,  'b+-.', 'LineWidth', 2);
plot(K, CBAUC,  'gx-', 'LineWidth', 2);
plot(K, TAUC,   'k*-', 'LineWidth', 2);



leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
leg2 = sprintf('EBAUROC (AUC = %.4f)', mean(EBAUC));
leg3 = sprintf('CBAUROC (AUC = %.4f)', mean(CBAUC));
leg4 = sprintf('TAUROC (AUC = %.4f)', mean(TAUC));

xlabel('Number of training samples');
ylabel('Average AUC');
legend(leg1, leg2, leg3, leg4);
grid on

%% Accuracy: Difference between Estimated AUC and Test AUC
% Updated: 27.04.2018 for MLSP'2018
%clear all
clc

load AUCs

iters = 1000;

CV_AUC_5_diff = zeros ( iters, length (samples) );
EBAUC_diff    = zeros ( iters, length (samples) );
CBAUC_diff    = zeros ( iters, length (samples) );

for k = 1:length(samples)
    AUC_iters = AUCs{k};
    CV_AUC_5_diff(:, k)  = abs (AUC_iters(:,1) - AUC_iters(:,4)); 
    EBAUC_diff(:, k)     = abs (AUC_iters(:,2) - AUC_iters(:,4)); 
    CBAUC_diff(:, k)     = abs (AUC_iters(:,3) - AUC_iters(:,4)); 
end

N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.

figure;

set(gca,'FontSize', 16); % added on 12.09.2018 for PRL manuscript
hold on
plot(K, mean(CV_AUC_5_diff), 'ro:', 'LineWidth', 2);
plot(K, mean(EBAUC_diff), 'b+-.', 'LineWidth', 2);
plot(K, mean(CBAUC_diff),  'gx--', 'LineWidth', 2);

leg1 = sprintf('CV-AUROC') ;
leg2 = sprintf('EBAUROC');
leg3 = sprintf('CBAUROC');

xlabel('Number of training samples');
%ylabel('MAE of estimated AUROC to test AUROC');
ylabel('Mean absolute error');
legend(leg1, leg2, leg3);
grid on



%% 10% samples
k = 1;

AUC = AUCs{k};
CV_AUC_10 = mean(AUC(:,1)); 
EBAUC_10 = mean(AUC(:,2)); % emperical BEE AUC
CBAUC_10 = mean(AUC(:,3)); % closed BEE AUC
TAUC_10 = mean(AUC(:,4)); % Test AUC
    
%% 90% samples
k = 17;

AUC = AUCs{k};
CV_AUC_90 = mean(AUC(:,1)); 
EBAUC_90 = mean(AUC(:,2)); % emperical BEE AUC
CBAUC_90 = mean(AUC(:,3)); % closed BEE AUC
TAUC_90 = mean(AUC(:,4)); % Test AUC
 
%% 95% samples
k = 18;

AUC = AUCs{k};
CV_AUC_95 = mean(AUC(:,1)); 
EBAUC_95 = mean(AUC(:,2)); % emperical BEE AUC
CBAUC_95 = mean(AUC(:,3)); % closed BEE AUC
TAUC_95 = mean(AUC(:,4)); % Test AUC

[CV_AUC_10, EBAUC_10, CBAUC_10, TAUC_10]
[CV_AUC_90, EBAUC_90, CBAUC_90, TAUC_90]
[CV_AUC_95, EBAUC_95, CBAUC_95, TAUC_95]


%%
% [samples' mean(CV_AUC_5_diff)' std(CV_AUC_5_diff)']
% 
% [mean(EBAUC_diff)' std(EBAUC_diff)']
% 
% [mean(CBAUC_diff)' std(CBAUC_diff)']

%% Difference and standard deviation in 10%, 90% and avg
clc
mean_CV = mean(CV_AUC_5_diff);
std_CV = std(CV_AUC_5_diff);
mean_EBAUC = mean(EBAUC_diff);
std_EBAUC = std(EBAUC_diff);
mean_CBAUC = mean(CBAUC_diff);
std_CBAUC = std(CBAUC_diff);
% avg
[mean(mean_CV)  std(std_CV)  mean(mean_EBAUC)  std(std_EBAUC)  mean(mean_CBAUC)  std(std_CBAUC)] 
% 10%, 90% ...
[samples' mean_CV' std_CV' mean_EBAUC' std_EBAUC' mean_CBAUC' std_CBAUC']
