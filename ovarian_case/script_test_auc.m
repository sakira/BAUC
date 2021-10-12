%% Test the idea of perfcurve and CV
close all
clear all
clc

% The data taken from: ROC graphs: Notes and practical considerations for
% Researchers by Tom Fawcett, 2004

% scores 
p1 = [.9, .8, .7, .6, .55, .54, .53, .52, .51, .505]';
p2 = [.4, .39, .38, .37, .36, .35, .34, .33, .3, .1]';

% class labels
y1 = [2, 2, 1, 2, 2, 2, 1, 1, 2, 1 ]';
y2 = [2, 1, 2, 1, 1, 1, 2, 1, 2, 1 ]';
[X, Y, T, auc_score_perf] = perfcurve([y1; y2], [p1;p2], 2);
figure; plot(X,Y); title('Perfcurve'); xlabel('FPR'); ylabel('TPR');

%% with cv

% Calculate ROC and AUC for the test sample:

th = sort([p1;p2]);

y = [y1; y2];
c1_idx = (y == 1);
c2_idx = (y == 2);

p = [p1; p2];
score1 = p(c1_idx);
score2 = p(c2_idx);

% 1. CV-ROC

ROC = zeros(length(th), 2);

for k = 1:length(th)
    FPR = nnz(score1 >= th(k)) / length(score1);
    TPR = nnz(score2 >= th(k)) / length(score2);
    ROC(k,:) = [FPR, TPR];
end

figure; plot(ROC(:,1), ROC(:,2)); title('cv'); xlabel('FPR'); ylabel('TPR');


% CV_AUC
ROC = flipud(ROC);
auc_score_cv   = trapz(ROC(:,1), ROC(:,2));

figure; plot(X, Y, '-ro', ROC(:,1), ROC(:,2), '-gx');
legend('perfcurve', 'cv');