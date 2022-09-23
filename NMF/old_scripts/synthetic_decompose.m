close all; clear all; clc

%% Synthetic Image
R = cat(3, ones(100,100), zeros(100,100), zeros(100,100));
B = cat(3, zeros(100,100), zeros(100,100), ones(100,100));

I = cat(2, R, B);

H = [1 0 0; 0 0 1];
figure(1), imshow(I);

%% NMF
[m,n,c] = size(I);
A = reshape(I, [m*n,3]);
A = double(A);

opt = statset('Display', 'final');
[B, X] = nnmf(A, 2,'algorithm','mult','h0',H,'Options',opt);

% %% Display
% B = reshape(B, [m,n,2]);
% ch1 = squeeze(B(:,:,1));
% ch2 = squeeze(B(:,:,2));
% C = cat(2, ch1, ch2);
% 
% figure(2), 
% subplot(1,3,1)
% imshow(I, []); 
% xlabel('Original');
% subplot(1,3,2)
% imshow(ch1, []);
% xlabel('Red Channel');
% subplot(1,3,3)
% imshow(ch2, []);
% xlabel('Blue Channel');
% 
% %%
% CC = medfilt2(C, [5 5]);
% imtool(CC, []);
% 
% %%
% Z = zeros(size(ch1));
% rgb = cat(3, ch1/max(ch1(:)), Z, ch2/max(ch2(:)));
% imtool(rgb,[])
% 
% imshow(Blue, Pink, 'montage')