% Mina Khoshdeli
% Date: June 2016
% =========================================================================
function [w0, mask] = he_seeds(I,M)
% I = double(I);
% this function gives the basis for NMF color deconvolution
% w0 : is the basis matrix or the stain matrix
% h0 : is the coefficients matrix for each stain
%-----------------------------------------------------
mask = zeros(size(I));
mask(I<0.1) = 1;
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

w0 = [fg_r, fg_g, fg_b;
      bg_r, bg_g, bg_b]';
w0(:,1) = w0(:,1) / norm(w0(:,1));
w0(:,2) = w0(:,2) / norm(w0(:,2));
end
