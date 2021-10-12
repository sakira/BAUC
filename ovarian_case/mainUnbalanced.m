%% Main script for unbalanced test on ovarian cancer dataset
%% First part:
% This part produces the AUC results of 3 different methods:
%    - CVAUC ( cross validation AUC) we choose 5-fold CV
%    - EBAUC (emperical BEE AUC)
%    - BAUC ( nonemperical/ closed form BEE AUC)   
% 
% The iterations are executed 500 times. Please change the glmnet library
% path and iteration numbers as necessary.
% Output: All results of the first part are stored in 'iterationUnbalanced' folder with the following format: 
%          iter_<iteration_number>_samples_<sample_size>.txt
clc
close all
clear all

% second experiment with unbalanced dataset
%addpath('F:\heikki\HeikkiHuttunen\glmnet_matlab\glmnet_matlab');
iters = 10;
for iter = 1:iters
    disp(['At Iteration ' num2str(iter)]); 
    iterationUnbalanced(iter);
end

%% 2nd part: combine the files of AUC results for three(?) different methods
% This part combines the different sample size results of AUC methods at each iteration
% The AUC is a cell variable of dimension 1xsamplesize. Here samplesize = 9.
% Each cell respresents a sample size (for example, 10%, 20%, ..., 90%) .Each cell contains 500 values of auc of 3 different methods. 
clear all
clc
samples = 10:10:95;
iters = 1000;
clear all
clc
samples = 10:10:95; % the percentage of samples.

iters = 10;

AUCs = cell(length(samples),1);

for k = 1:length(samples)
    AUC_iter = zeros(iters,4);
    
    for iter = 1:iters
        
        % open file of each iteration
        try
            fileName  = sprintf('iterationIm/AUC_iter%d_perc%d',iter,k);
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

save('AUCunbalanced.mat', 'AUCs');



%% 3rd  part: Plotting
%close all

close all;
clear all;
clc;

load AUCunbalanced.mat
unbalancedPartition = .1:.1:.95;
N = 106;
K = round(N * unbalancedPartition);

for k = 1:length(K)
    sample = AUCs{k};
    CV_AUC(k) = mean(sample(:,1)); 
    EBAUC(k) = mean(sample(:,2)); % EBAUROC
    CBAUC(k) = mean(sample(:,3)); % CBAUROC
    
    
 
end


% cross-validation: 'ro:'  
% emperical BEE: 'b+-.'
% closed form BEE: 'gx--'
% 

% cross-validation: 'bo:'  
% closed form BEE: 'r+-'


figure;
hold on
% plot(K, CV_AUC, 'r-o', 'LineWidth', 2);
% plot(K, NBAUC, 'g-o', 'LineWidth', 2);
% plot(K, EBAUC, 'b-o', 'LineWidth', 2);
plot(K, CV_AUC, 'bo:', 'LineWidth', 1);
plot(K, EBAUC, 'g+-.', 'LineWidth', 1);
plot(K, CBAUC,  'r+-', 'LineWidth', 1);



% leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
% leg2 = sprintf('BAUC (AUC = %.4f)', mean(NBAUC));
% leg3 = sprintf('EBAUC (AUC = %.4f)', mean(EBAUC));
leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
leg2 = sprintf('EBAUC (AUC = %.4f)', mean(EBAUC));
leg3 = sprintf('CBAUC (AUC = %.4f)', mean(CBAUC));

%gcapercent('X')
xlabel('Number of cancer patients');
ylabel('Average AUC');
legend(leg1, leg2, leg3);
%legend(leg1, leg3);

grid on

% Convert y-axis values to percentage values by multiplication

a=[cellstr(num2str(get(gca,'xtick')'))];

% Create a vector of '%' signs

pct = char(ones(size(a,1),1)*'%');

% Append the '%' signs after the percentage values

new_xticks = [char(a),pct];

% 'Reflect the changes on the plot

set(gca,'xticklabel',new_xticks)

%%
% plot(K, CV_AUC, 'ro:', 'LineWidth', 2);
% plot(K, EBAUC, 'b+-.', 'LineWidth', 2);
% plot(K, NBAUC,  'gx--', 'LineWidth', 2);
% 
% 
% 
% leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
% leg2 = sprintf('EBAUC (AUC = %.4f)', mean(EBAUC));
% leg3 = sprintf('CBAUC (AUC = %.4f)', mean(NBAUC));
% 
% %title(['AUC']);
% xlabel('Number of training samples');
% ylabel('Average AUC');
% legend(leg1, leg2, leg3);
% grid on



%% Proof of accuracy


clear all
clc
samples = 10:10:95; % the percentage of samples.

iters = 1000;

samplesAUC = cell(length(samples),1);

for k = 1:length(samples)
    accuracy = [];
%k = 15;

%for k = 1:length(samples)
    
    %AUC{end+1} = zeros(iters, 3);
    
    for iter = 1:iters
        %indata = cell(1,3);
        
        % open file of each iteration
        try
            fileName  = sprintf('iterationUnbalanced/AUC/AUC_iter%d_perc%d',iter,k);
            accuracy = [accuracy load(fileName)];
        catch 
            disp(['Missing iteration ' num2str(iter) ' ...']); 
            %indata{1} = iter;
        end
        
        
        disp(['Data loaded successfully at Iteration: ' num2str(iter) ' for ' num2str(samples(k)) '% samples ...']); 
        
    end
    
    samplesAUC{k} = accuracy;
    
end

disp(['Finnish loading data ...']);
save('samplesUnbalancedAUC.mat', 'samplesAUC');
%% Plot histograms
close all;
clear all;
clc
load samplesUnbalancedAUC.mat

samples = 10:5:95; % the percentage of samples.
N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.

%figure;

samples_mean_differences = zeros(length(samplesAUC),3); 
samples_std_differences = zeros(length(samplesAUC),3); 

for k = 1: length(samplesAUC)
    accuracy = samplesAUC{k};
    accuracies = [];

    for ind = 1:1000
        AUC_CV = accuracy(ind).AUC(:,1);
        AUC_EBAUC = accuracy(ind).AUC(:,3);
        AUC_CBAUC = accuracy(ind).AUC(:,2);
        AUC_TEST = accuracy(ind).AUC(:,5);
        acc = [AUC_TEST, AUC_CV, AUC_EBAUC, AUC_CBAUC];
        accuracies = [accuracies; acc];
    end

    accuracies = accuracies (all(~isinf(accuracies ), 2), :);

    differences = [accuracies(:, 1) - accuracies(:, 2), accuracies(:, 1) - accuracies(:, 3), accuracies(:, 1) - accuracies(:, 4)];
    %mean(abs(differences))
    corr(accuracies)
    
    samples_mean_differences(k,:) = mean(differences);
    samples_std_differences(k,:) = std(differences);


end
figure;
plot(samples, samples_mean_differences, 'LineWidth', 2);
legend('CV', 'EBAUC', 'CBAUC')
% save('sample_differences.mat', 'samples', 'samples_mean_differences',
% 'samples_std_differences');
%%
close all
clear all
clc

load sample_differences.mat
N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.



figure;
f = [samples_mean_differences+2*sqrt(samples_std_differences); flipdim(samples_mean_differences-2*sqrt(samples_std_differences),1)];
fill([K'; flipdim(K',1)], f(:,1), 'r', 'LineStyle', ':')
hold on;
fill([K'; flipdim(K',1)], f(:,2), 'g', 'LineStyle', '--')
fill([K'; flipdim(K',1)], f(:,3), 'b', 'LineStyle', '-.')

p1 = plot(K, samples_mean_differences(:,1), 'ro:', 'LineWidth', 1); 
p2 = plot(K, samples_mean_differences(:,2), 'gx--', 'LineWidth', 1);
p3 = plot(K, samples_mean_differences(:,3), 'b+-.', 'LineWidth', 1);
alpha(0.75)
xlabel('Number of training samples');
ylabel('Average AUC deviation from true AUC');

legend([p1 p2 p3], 'CV', 'EBAUC', 'CBAUC')
grid on
