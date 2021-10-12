function [x,y] = shuffling(X,Y)

    N = size(X,1);
    idx = randperm(N);
    x = X(idx,:);
    y = Y(idx,:);
end