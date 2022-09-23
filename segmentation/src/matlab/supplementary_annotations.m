%%%
% Interactive Human Segmentation
% Author: Garrett Winkelmaier
% Date: 11/02/21
% Description: Interactively annotate supplementary images to boot-strap 
%              ML model
%%%
close all; clear all; clc

% GLOBALS
PROJECT_DIR = '/home/gwinkelmaier/parvin-labs/GBM/segmentation'; 
DATA_DIR = fullfile(PROJECT_DIR, 'data', 'supp-snapshots', 'images');
SAVE_DIR = fullfile(PROJECT_DIR, 'data', 'supp-snapshots', 'masks');
FILE_TYPE = '*.tif';

BOUNDARY = 25;  % Boundary of seeds to remove around the boarder
AREA_THRESH = 35;

% MAIN
% get a list of files
files = dir([DATA_DIR '/0' FILE_TYPE]);

% cycle through files
% for i=1:numel(files)
for i = 7
    % read file
    Irgb = imread([files(i).folder '/' files(i).name]);
    Irgb_double = double(Irgb);
%     I = double(rgb2gray(Irgb));
    I = (100*squeeze(Irgb_double(:, :, 3))) ./ (squeeze(Irgb_double(:, :, 1)) + squeeze(Irgb_double(:, :, 2)));
%     I(isinf(I)) = 0;
%     I(isnan(I)) = 0;
    
%     % Crop out scalebar
    [m, n] = size(I);
    n = n - 10;
    m = m - 100;
    I = imcrop(I, [0, 0, n, m]);
    Irgb = imcrop(Irgb, [0, 0, n, m]);
       
    % apply LoG filter and get seeds
    [BW, P] = log_segmentation(I, 6, AREA_THRESH);
%     figure(5), imshow(BW, []);
    [seeds.x, seeds.y] = ind2sub(size(BW), find(P));
   
    % Visulaize Seeds
    figure(1), imshow(Irgb, []);
    hold on, plot(seeds.y, seeds.x, 'y*'); hold off
    
    % Distance Map
    D = bwdist(P);
    try
        L = watershed(D) .* uint16(BW);
    catch
        L = uint16(watershed(D)) .* uint16(BW);
    end
       
    
%     break
    figure(2), imshow(Irgb, []);
    hold on; visboundaries(L, 'Color', 'g'); hold off;
    % Save Masks
%     imwrite(Irgb, fullfile(files(i).folder, ['cropped-' files(i).name]));
%     imwrite(logical(L), fullfile(SAVE_DIR, ['cropped-' files(i).name]));
end

% FUNCTIONS
function [L, P] = log_segmentation(image, center_sigma, area_thresh)
%%% Apply a Laplacian-of-Gaussian Filter 
%%% Returns Thresholded LoG mask and Peaks locations

H_total = zeros(size(image));
for sigma=[center_sigma-1:center_sigma+1]
% for sigma=[4,7]
% Create a filter
h_size = 9*sigma;
if mod(h_size, 2) == 0
    h_size = h_size + 1;
end

% Apply Filter
h = fspecial('log', h_size, sigma);
H = imfilter(image, -1*h, 'symmetric');

% Normalize Image
H = H - min(H(:));
H = H / max(H(:));

% Maximum Projection
H_total = max(H_total, H);
end

figure(3), imshow(H_total, []);

% Create Seeds
P = imregionalmax(H_total);
BW = H_total > 0.65;
% BW = adaptthresh(H_total, 0.8);
BW = bwareafilt(BW, [area_thresh, Inf]);

% Return 
L = BW;
P = P & L;
end
