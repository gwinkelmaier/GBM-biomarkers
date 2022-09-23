close all; clear all; clc

%% Get Images
img_number = 2;
I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
imshowpair(uint8(I),M,'montage')

%% Convert the RGB space to optical density
% I0 = max(I(:))+1;
OD = -log(I./max(I(:)) + 0.001);
% n = find(OD<0);
% OD(OD<0) = 0;

%% Create W
W1 = double(M(:));
W2 = double(~M(:));
W = cat(2, W1, W2);

%% NMF
[m,n,c] = size(OD);
A = reshape(OD, [m*n,3]);
A = double(A);

[B, X] = nnmf(A, 2,'algorithm','als','w0',W);

%% Display
B = reshape(B, [m,n,2]);
ch1 = squeeze(B(:,:,1));
ch2 = squeeze(B(:,:,2));

figure(1), imshow(uint8(I))
figure(2), imshow(ch1,[])
figure(3), imshow(ch2,[])
figure(4), imshowpair(ch1,ch2)