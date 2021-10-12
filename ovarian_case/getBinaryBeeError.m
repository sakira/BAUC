function [err, E1, E2] = getBinaryBeeError (X, y, a, b, varargin)
%
% Estimate the error of a linear classifier using the
% Bayesian MMSE error estimate.
%
% Usage: err = getBinaryBeeError (X, y, a, b, covMode)
%
% where X is the NxP data matrix, y is a Nx1 vector containing
% binary class labels, and a and b define the discriminant
% function: g(x) = ax + b. covMode is either 'identity' or
% 'general' corresponding to assumption of scaled identity
% covariance or general covariance.
%
% For details, see:
%
% L. A. Dalton and E. R. Dougherty, "Bayesian Minimum Mean-Square Error
% Estimation for Classification Error�Part II: The Bayesian MMSE Error
% Estimator for Linear Classification of Gaussian Distributions,"
% IEEE Transactions on Signal Processing, vol. 59, no. 1, pp. 130�144,
% January 2011.
%
% (c) 4.6.2012 Heikki.Huttunen@tut.fi
%

if (any(~isfinite(a)) || any(~isfinite(b)))
    err = inf;
    E1 = 0.5;
    E2 = 0.5;
    return
end

if nargin < 5 || isempty(varargin{1})
    beeOptions = struct('covMode', 'identity', 'prior', 'improper', 'std', 1);
else
    beeOptions = varargin{1};
end

% Remove columns of X with zero coefficient

X = X(:, a ~= 0);
a = a(a ~= 0);

if size(X,2) == 0
    err = 0.5;
    E1 = 0.5;
    E2 = 0.5;
    return
end

covMode = beeOptions.covMode;

% Define the basic constants

uniqClasses = unique(y);

D = size(X,2);
n1 = nnz(y == uniqClasses(1));
n2 = nnz(y == uniqClasses(2));

priorType = beeOptions.prior;

if strcmpi(priorType, 'improper')
    kappa = 0;
    nu = 0;
    S = zeros(D);
    m = zeros(D, 1);
elseif strcmpi(priorType, 'proper')
    kappa = D+2;
    nu = 0.5;
    S = eye(D) * beeOptions.std;
    m = zeros(D, 1);
end

% c = 0.5;
c = nnz(y == uniqClasses(1)) / length(y);

% Calculate class means and covariances

mu1 = mean(X(y == uniqClasses(1), :))';
mu2 = mean(X(y == uniqClasses(2), :))';

cov1 = cov(X(y == uniqClasses(1), :));
cov2 = cov(X(y == uniqClasses(2), :));

% Calculate star-quantities

Sx1 = (n1 - 1) * cov1 + S + (n1*nu / (n1 + nu)) * (mu1 - m) * (mu1 - m)';
Sx2 = (n2 - 1) * cov2 + S + (n2*nu / (n2 + nu)) * (mu2 - m) * (mu2 - m)';

nux1 = nu + n1;
nux2 = nu + n2;

kappax1 = kappa + n1;
kappax2 = kappa + n2;

mx1 = (n1 * mu1 + nu * m) / (n1 + nu);
mx2 = (n2 * mu2 + nu * m) / (n2 + nu);

beta1 = 0.5 * trace(Sx1);
beta2 = 0.5 * trace(Sx2);

alpha1 = ((kappax1 + D + 1)*D) / 2 - 1;
alpha2 = ((kappax2 + D + 1)*D) / 2 - 1;

% Calculate error estimates for the two classes

if strcmpi(covMode, 'IDENTITY') % Assume identity covariance matrix
    
    A1 = (-1)^0 * (a'*mx1 + b) * sqrt((nux1 / (nux1 + 1))) / norm(a);
    A2 = (-1)^1 * (a'*mx2 + b) * sqrt((nux2 / (nux2 + 1))) / norm(a);
    
    E1 = 0.5 + 0.5*sign(A1) .* ...
        betainc(A1.^2 ./ (A1.^2 + 2*beta1), 0.5, alpha1);
    E2 = 0.5 + 0.5*sign(A2) .* ...
        betainc(A2.^2 ./ (A2.^2 + 2*beta2), 0.5, alpha2);

elseif strcmpi(covMode, 'GENERAL') % Assume general covariance matrix
    
    A1 = (-1)^0 * (a'*mx1 + b) * sqrt((nux1 / (nux1 + 1)));
    A2 = (-1)^1 * (a'*mx2 + b) * sqrt((nux2 / (nux2 + 1)));
    
    try
        E1 = 0.5 + 0.5*sign(A1) .* ...
            betainc(A1.^2 ./ (A1.^2 + a'*Sx1*a), 0.5, (kappax1 - D + 1) / 2);
        E2 = 0.5 + 0.5*sign(A2) .* ...
            betainc(A2.^2 ./ (A2.^2 + a'*Sx2*a), 0.5, (kappax2 - D + 1) / 2);
    catch 
        warning('Unable to calculate error estimate: not enough samples.');
        E1 = 0.5;
        E2 = 0.5;
    end
else
    error('Unknown covariance mode.');
end

% Final estimate is the weighted sum of the class-errors.

err = c * E1 + (1-c) * E2;
