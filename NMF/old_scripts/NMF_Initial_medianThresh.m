% =========================================================================
% Mina Khoshdeli
% Date: June 2016
% =========================================================================
function [w0 ] = NMF_Initial_medianThresh(I,M)
% I = double(I);
% this function gives the basis for NMF color deconvolution
% w0 : is the basis matrix or the stain matrix
% h0 : is the coefficients matrix for each stain
%-----------------------------------------------------
% threshold = [70 51 41];

mask = zeros(size(I));
mask(I>200) = 1;
mask = sum(mask, 3);
mask = mask ~= 3;

R = I(:,:,1);
G = I(:,:,2);
B = I(:,:,3);
fg_r = median(R(M & mask));
fg_g = median(G(M & mask));
fg_b = median(B(M & mask));

bg_r = median(R(~M & mask));
bg_g = median(G(~M & mask));
bg_b = median(B(~M & mask));

% if abs(fg_b - bg_b) < threshold(3)
%     bg_b = fg_b - threshold(3);
% end

% s = size(I);
% figure, 
% subplot(1,2,1)
% imshow(uint8(I.*(~M&mask)));
% subplot(1,2,2)
% imshow(uint8(I.*(M&mask)));


w0 = [fg_r, fg_g, fg_b; bg_r, bg_g, bg_b]';
