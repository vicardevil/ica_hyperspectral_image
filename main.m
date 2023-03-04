% read hyperspectral image
clear, close all;
k = 6;
[him, ~] = HSIReader('./data/yyc200');

% normalize the hyperspectral image
him = double(him) / 4095;
% get the size of the hyperspectral image
[m, n, l] = size(him);

% reshape the hyperspectral image to a matrix
X = reshape(him, [], l);
% perform PCA
[P, ~, ~] = svds(X' * X, k);
X_pca = X * P * P(1:k, :)';
him_pca = reshape(X_pca, m, n, k);

% show the results
figure;
for i = 1:k
    subplot(2, ceil(k / 2), i);
    imshow(him_pca(:, :, i), []);
end
sgtitle('PCA')

% perform PSA
X_psa = PSA(X_pca, k);
him_psa = reshape(X_psa, m, n, k);

% show the results
figure;
for i = 1:k
    subplot(2, ceil(k / 2), i);
    imshow(him_psa(:, :, i), []);
end
sgtitle('PSA')
