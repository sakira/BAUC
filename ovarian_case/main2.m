%% This is the main file for computation of ovarian cancer data which has balanced dataset.
%% First part:
% This part produces the AUC results of 3 different methods:
%    - CVAUC ( cross validation AUC) we choose 5-fold CV
%    - EBAUC (emperical BEE AUC)
%    - BAUC ( nonemperical/ closed form BEE AUC)   
% 
% The iterations are executed 500 times. Please change the glmnet library path and iteration numbers in order to get the exact results as uploaded in the link:
% https://www.overleaf.com/2786676mzrgpv#/7471689/
% Output: All results of the first part are stored in 'iteration' folder with the following format: 
%          iter_<iteration_number>_samples_<sample_size>.txt



%% Change Logs: 01.02.2017: iterations are executed for 1000 times.
clc
close all
clear all

% First experiment with balanced dataset
addpath('P:\Users personal data\phd materials\libraries\glmnet_matlab\glmnet_matlab');
iters = 5;
for iter = 1:iters
    disp(['At Iteration ' num2str(iter)]); 
    iteration(iter);
end


% class1 = (sum(Y==1))/length(Y)
% class2 = (sum(Y==2))/length(Y)
% class1 + class2
% 
% trainClass1 = (sum(y_train==1))/length(y_train)
% trainClass2 = (sum(y_train==2))/length(y_train)
% trainClass1 + trainClass2
% 
% testClass1 = (sum(y_test==1))/length(y_test)
% testClass2 = (sum(y_test==2))/length(y_test)
% testClass1 + testClass2


%% 2nd part: combine the files of AUC results for three(?) different methods
% This part combines the different sample size results of AUC methods at each iteration
% The AUC is a cell variable of dimension 1xsamplesize. Here samplesize = 18.
% Each cell respresents a sample size (for example, 10%, 15%, ..., 95%) .Each cell contains 1000 values of auc of 3 different methods. 
clear all
clc
samples = 10:5:95; % the percentage of samples.
iters = 1000;
%iters = 2;

AUC = {};


for k = 1:length(samples)
    
    AUC{end+1} = zeros(iters, 3);
    
    for iter = 1:iters
        indata = cell(1,3);
        
        % open file of each iteration
        try
            inFile = fopen(sprintf('iteration/iter_%d_samples_%d.txt',iter,samples(k)), 'r');
            indata = textscan(inFile, '%f %f %f', 'HeaderLines',1);
            fclose(inFile);
        catch 
            disp(['Missing iteration ' num2str(iter) ' ...']); 
            indata{1} = iter;
        end
        
        % indata{1} = CV_AUC
        % indata{2} = NBAUC
        % indata{3} = EBAUC
        % indata{4} = PARAM_AUC
        AUC{end}(iter,:) = [indata{1}, indata{2}, indata{3}];
        
        disp(['Data loaded successfully at Iteration: ' num2str(iter) ' for ' num2str(samples(k)) '% samples ...']); 
        
    end
    
end

disp(['Finish loading data ...']);

save('AUC.mat', 'AUC');



%% 3rd part: Plotting
% This part average the auc results for each sample size.
%close all
close all;
clear all;
clc;
load AUC.mat
samples = 10:5:95;
N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.



for k = 1:length(samples)
    sample = AUC{k};
    CV_AUC(k) = mean(sample(:,1)); 
    NBAUC(k) = mean(sample(:,2)); % closed BEE AUC
    EBAUC(k) = mean(sample(:,3)); % emperical BEE AUC
    
    
 
end


% cross-validation: 'ro:'  
% emperical BEE: 'b+-.'
% closed form BEE: 'gx--'
% 

% cross-validation: 'bo:'  
% closed form BEE: 'r+-'

figure;
hold on
plot(K, CV_AUC, 'bo:', 'LineWidth', 1);
plot(K, EBAUC, 'g+-.', 'LineWidth', 2);
plot(K, NBAUC,  'r+-', 'LineWidth', 1);



leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
leg2 = sprintf('EBAUC (AUC = %.4f)', mean(EBAUC));
leg3 = sprintf('CBAUC (AUC = %.4f)', mean(NBAUC));

%title(['AUC']);
xlabel('Number of training samples');
ylabel('Average AUC');
legend(leg1, leg2, leg3);
%legend(leg1, leg3);
grid on
%% Display the result

[K' CV_AUC' EBAUC' NBAUC']




%%




% %% combine the files of AUC results for three(?) different methods
% clear all
% clc
% samples = 10:10:95;
% iters = 200;
% iters = 100;
% %iters = 1
% AUC = {};
% 
% 
% for k = 1:length(samples)
%     %outFile = fopen(sprintf('resultsOvarianAlpha1/auc_samples_%d.txt',samples(k)), 'wt');
%     %fprintf(outFile,'%10s %10s %10s %10s\n','EBAUC','AUC_ORACLE', 'AUC_CV', 'BAUC');
%     %formatSpec = '%f %f %f %f\n';
%     
%     %AUC{end+1} = zeros(iters, 4);
%     AUC{end+1} = zeros(iters, 3);
%     
%     for iter = 1:iters
%         indata = cell(1,3);
%         %indata = cell(1,4);
%         % open file of each iteration
%         try
%             inFile = fopen(sprintf('iteration/iter_%d_samples_%d.txt',iter,samples(k)), 'r');
%             indata = textscan(inFile, '%f %f %f %f', 'HeaderLines',1);
%             fclose(inFile);
%         catch 
%             disp(['Missing iteration ' num2str(iter) ' ...']); 
%             indata{1} = iter;
%         end
%         
%         % indata{1} = CV_AUC
%         % indata{2} = NBAUC
%         % indata{3} = EBAUC
%         % indata{4} = PARAM_AUC
%         AUC{end}(iter,:) = [indata{1}, indata{2}, indata{3}, indata{4}];
%         
%         % write to file
%         %fprintf(outFile,formatSpec, indata{1,:});
%         
%         
%     end
%     %fclose(outFile);
% end
% 
% 
% 
% 
% %% Plotting
% close all
% 
% for k = 1:length(samples)
%     sample = AUC{k};
%     CV_AUC(k) = mean(sample(:,1)); 
%     NBAUC(k) = mean(sample(:,2)); % closed BEE AUC
%     EBAUC(k) = mean(sample(:,3)); % emperical BEE AUC
%     PARAM_AUC(k) = mean(sample(:,4)); 
%     
%  
% end
% 
% 
% figure(1);
% hold on
% plot(samples, CV_AUC, 'r-o');
% plot(samples, NBAUC, 'g-o');
% plot(samples, EBAUC, 'b-o');
% plot(samples, PARAM_AUC, 'k-o');
% 
% 
% 
% leg1 = sprintf('CV (AUC = %.4f)', mean(CV_AUC) ) ;
% leg2 = sprintf('NBAUC (AUC = %.4f)', mean(NBAUC));
% leg3 = sprintf('EBAUC (AUC = %.4f)', mean(EBAUC));
% leg4 = sprintf('PARAM (AUC = %.4f)', mean(PARAM_AUC));
% title(['AUC']);
% xlabel('Number of samples');
% ylabel('AUC');
% legend(leg1, leg2, leg3, leg4);
% grid on
%% Proof of accuracy


clear all
clc
samples = 10:5:95; % the percentage of samples.

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
            fileName  = sprintf('iteration/AUC/AUC_iter%d_perc%d',iter,k);
            accuracy = [accuracy load(fileName)];
        catch 
            disp(['Missing iteration ' num2str(iter) ' ...']); 
            %indata{1} = iter;
        end
        
        
        disp(['Data loaded successfully at Iteration: ' num2str(iter) ' for ' num2str(samples(k)) '% samples ...']); 
        
    end
    
    samplesAUC{k} = accuracy;
    
end

disp(['Finish loading data ...']);
save('samplesAUC.mat', 'samplesAUC');
%% Plot histograms
close all;
clear
clc
load samplesAUC.mat

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

    TEST = accuracies(:,1);
    diff_AUC_CV = abs(TEST - accuracies(:,2));
    diff_EBAUC = abs(TEST - accuracies(:,3));
    diff_CBAUC = abs(TEST - accuracies(:,4));
    differences = [accuracies(:, 1) - accuracies(:, 2), accuracies(:, 1) - accuracies(:, 3), accuracies(:, 1) - accuracies(:, 4)];
    %mean(abs(differences))
    %corr(accuracies)
    
    samples_mean_differences(k,:) = mean(abs(differences));
    samples_std_differences(k,:) = std(abs(differences));
    
    
%     figure(1); hold all;
%     hist(differences(:,1), linspace(-0.03, 0.03, 100));
%     
%     figure(2); hold all;
%     hist(differences(:,2), linspace(-0.03, 0.03, 100));
%     
%     figure(3); hold all;
%     hist(differences(:,3), linspace(-0.03, 0.03, 100));
    
    %hold all;
    %figure(3); hold all;
    %figure;
    %hist(differences(:,1), linspace(-0.6, 0.8, 1000) );
    %axis([-0.8 1 0 10e3]);
    %M_CV(k) = getframe(gcf);
    %legend('CV', 'EBAUC', 'CBAUC')
    
%     figure; hold all;
%     hist(differences, linspace(-0.03, 0.03, 100));
%     legend('CV', 'EBAUC', 'CBAUC')
    
    

end
% figure;
% plot(samples, samples_mean_differences, 'LineWidth', 2);
% legend('CV', 'EBAUC', 'CBAUC')
save('sample_differences.mat', 'samples', 'samples_mean_differences', ...
 'samples_std_differences');
%% ----------------------------------------------------------------------
% Plots for i) Mean of average AUC bias
%          ii) Standard deviation of average AUC bias
% Change Log: 20.06.2017
% -----------------------------------------------------------------------
close all
clear 
clc

load sample_differences.mat
samples = 10:5:95;
N = 216; % The total number of samples.
K = floor(N .* samples ./ 100); % Compute the number of samples at each percentage.



% cross-validation: 'ro:'  
% emperical BEE: 'b+-.'
% closed form BEE: 'gx--'
% 


% cross-validation: 'bo:'  
% closed form BEE: 'r+-'

figure;
hold on;

p1 = plot(K, samples_mean_differences(:,1), 'bo:', 'LineWidth', 1); 
%p2 = plot(K, samples_mean_differences(:,2), 'b+-.', 'LineWidth', 2);
p3 = plot(K, samples_mean_differences(:,3), 'r+-', 'LineWidth', 1);
alpha(0.5)
xlabel('Number of training samples');
ylabel('Mean of (Average AUC - True AUC)');

%legend([p1 p2 p3], 'CV', 'EBAUC', 'CBAUC')
legend([p1 p3], 'CV', 'CBAUC')
grid on

figure;
hold on;

p1 = plot(K, samples_std_differences(:,1), 'bo:', 'LineWidth', 1); 
%p2 = plot(K, samples_std_differences(:,2), 'b+-.', 'LineWidth', 2);
p3 = plot(K, samples_std_differences(:,3), 'r+-', 'LineWidth', 1);

xlabel('Number of training samples');
ylabel('Standard deviation of (Average AUC - True AUC)');

legend([p1 p3], 'CV', 'CBAUC')
%legend([p1 p2 p3], 'CV', 'EBAUC', 'CBAUC')
grid on


% figure;
% hold on
% errorbar(K, samples_mean_differences(:,1), samples_std_differences(:,1), 'bo:');
% errorbar(K, samples_mean_differences(:,3), samples_std_differences(:,3), 'r+-');
% grid on


% figure;
% f = [samples_mean_differences+2*sqrt(samples_std_differences); flipdim(samples_mean_differences-2*sqrt(samples_std_differences),1)];
% fill([K'; flipdim(K',1)], f(:,1), 'r', 'LineStyle', ':')
% hold on;
% fill([K'; flipdim(K',1)], f(:,2), 'b', 'LineStyle', '-.')
% fill([K'; flipdim(K',1)], f(:,3), 'g', 'LineStyle', '--')
% 
% p1 = plot(K, samples_mean_differences(:,1), 'ro:', 'LineWidth', 1); 
% p2 = plot(K, samples_mean_differences(:,2), 'b+-.', 'LineWidth', 1);
% p3 = plot(K, samples_mean_differences(:,3), 'gx--', 'LineWidth', 1);
% alpha(0.5)
% xlabel('Number of training samples');
% ylabel('Average AUC deviation from true AUC');
% 
% legend([p1 p2 p3], 'CV', 'EBAUC', 'CBAUC')
% grid on




% 
%  d = 0.1;
%  x = linspace(0,1,20);
%  z = d*x.^2;
% 
%  fill([x flip(x)],[z zeros(size(z))],'k','LineStyle','none')
%  hold on
%  plot(x,z,'k-');
%  alpha(0.25)
% 
%save('movies_CV.mat', 'M_CV');
%%
% clear all;
% close all;
% clc;
% load movies.mat
% load movies_CV.mat
% load movies_EBAUC.mat
% figure;movie(M_CBAUC);
% figure; movie(M_CV);
% figure; movie(M_EBAUC);