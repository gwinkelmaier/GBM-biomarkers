%%% Interactive Data Augmentation
clear; clc

cwd = cd;
DATA_DIR = fullfile(cwd, '..', '..', 'data', '*.*');

%% UI input
% Read Histology image
[file, path] = uigetfile(DATA_DIR);
I = imread(fullfile(path, file));

% Read nuclear mask
mask_path = strrep(path, 'images', 'masks');
mask_file = strrep(file, 'image', 'mask');
M = imread(fullfile(mask_path, mask_file));

%% Display original, 75% and 50%
mask_75 = ones(size(M));
mask_75(M) = 0.75;
rgb_75 = I;
% rgb_75(:, :, 3) = uint8(double(rgb_75(:, :, 3)) .* mask_75);
rgb_75 = uint8(double(rgb_75) .* mask_75);

mask_50 = ones(size(M));
mask_50(M) = 0.5;
rgb_50 = I;
% rgb_50(:, :, 3) = uint8(double(rgb_50(:, :, 3)) .* mask_50);
rgb_50 = uint8(double(rgb_50) .* mask_50);

%% Visualize
figure(1), imshow(I, 'border', 'tight');
figure(2), imshow(rgb_75, 'border', 'tight');
figure(3), imshow(rgb_50, 'border', 'tight');
