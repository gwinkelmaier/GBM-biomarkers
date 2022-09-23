%==========================================================================
% Mina Khoshdeli
% Data : June 2016
% This fuction decompose an H&E stained histology image into two channels.
% Input : I is the RGB image
% Outputs : C1 and C2 are two color channels
% The basis for nnmf is initialized using w0.
%==========================================================================
function [C1, C2] = decompose_threshold(I,M)
l = [455.237613, 397.007634];

% Compute Seeds
I = double(I);
% I = log(255) - log(I+1);
% Ir = I(:,:,1);
% Ir = medfilt2(Ir,[5,5]);
% Ir = medfilt2(Ir,[5,5]);
% Ir = medfilt2(Ir,[5,5]);
% Ig = I(:,:,2);
% Ig = medfilt2(Ig,[5,5]);
% Ig = medfilt2(Ig,[5,5]);
% Ig = medfilt2(Ig,[5,5]);
% Ib = I(:,:,3);
% Ib = medfilt2(Ib,[5,5]);
% Ib = medfilt2(Ib,[5,5]);
% Ib = medfilt2(Ib,[5,5]);
% I = cat(3, Ir, Ig, Ib);


% I = -log((I+1)/255);
[h0 ] = NMF_Initial_medianThresh(I,M);
% h0 = [h0; 0 0 0];
h0 = h0 / norm(h0);

% NMF Algorithm (decomposition)
s = size(I);
D = reshape(I, s(1)*s(2), 3);

[W,H] = nnmf(D, 2,'algorithm','als','h0',h0);
% [W,H] = nnmf(D, 2,'algorithm','als');

% Reconstruct Original Image
% Y = W*H;
% Y = reshape(Y, size(I));
% imtool(Y, [])

% Reformat decomposed layers
% C1 = reshape(W(:,1), s(1), s(2));
% C2 = reshape(W(:,2), s(1), s(2));

% Swap cytoplasm and nuclear if needed
% fprintf("%f, %f, %f, %f\n", H(1,2), H(2,2), mean(C1(M)), mean(C2(M)));
% if H(1,2) < H(2,2) & mean(C1(M)) > mean(C2(M))
%     fprintf("Switch!")
%     C3 = C1;
%     C1 = C2;
%     C2 = C3;
% end

% Rescale
C1 = 255 / (max(W(:,1)) - min(W(:,1))) * (W(:,1) - min(W(:,1)));
C2 = 255 / (max(W(:,2)) - min(W(:,2))) * (W(:,2) - min(W(:,2)));
% C3 = 255 / (max(W(:,3)) - min(W(:,3))) * (W(:,3) - min(W(:,3)));
C1 = reshape(C1, s(1), s(2));
C2 = reshape(C2, s(1), s(2));
% C3 = reshape(C3, s(1), s(2));
% C1 = 255 / l(1) * C1;
% C2 = 255 / l(2) * C2;


% Median Filter
C1 = medfilt2(C1, [2,2]);
C2 = medfilt2(C2, [2,2]);
% C3 = medfilt2(C3, [2,2]);

% fprintf("C1: [%f, %f]\t", min(C1(:)), max(C1(:)));
% fprintf("C2: [%f, %f]\n", min(C2(:)), max(C2(:)));
% mask_1 = C1(M);
% Ib = I(:,:,3);
% mask_2 = Ib(M);
% mask_1 = mask_1 / norm(mask_1);
% mask_2 = mask_2 / norm(mask_2);
% dot(mask_1, mask_2);
mask_1 = C2(M);
m1 = median(mask_1);
mask_2 = C2(~M);
m2 = median(mask_2);
fprintf("%f\t%f\n", m1, m2);
end
