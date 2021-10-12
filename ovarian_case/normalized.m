function normalizedX = normalized (X,mu,sigma)
normalizedX = bsxfun(@minus, X, mu);
normalizedX = bsxfun(@rdivide, normalizedX, sigma);