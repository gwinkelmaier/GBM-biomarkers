close all; clear all; clc

%% Get Images
img_number = 1;
I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);

%% Mina Decomposition with Mask Initialization
[~, ch1, ch2] = decompose(I,M);

%% Display
figure(1), imshow(uint8(I));
figure(2), imshow(ch1);
figure(3), imshow(ch2);