%==========================================================================
% Mina Khoshdeli
% Data : June 2016
% This fuction decompose an H&E stained histology image into two channels.
% Input : I is the RGB image
% Outputs : C1 and C2 are two color channels
% The basis for nnmf is initialized using w0.
%==========================================================================
function [w0, C1, C2] = decompose_threshold(I,M)
I = double(I);
[w0 ] = NMF_Initial_medianThresh(I,M);
tmp = w0(:,1);
w0(:,1) = w0(:,2);
w0(:,2) = tmp;

s = size(I);
l = [0.003865, 0.005774];

% Apply NMF
for i = 1:3
%     t = OD(:,:,i);
    t = I(:,:,i);
    D(i,:) = t(:);
end

[B, X] = nnmf(D, 2,'algorithm','als','w0',w0);
Y = B*X;
Y = permute(Y, [2,3,1]);
imshowpair(uint8(I), uint8(reshape(Y, size(I))), 'montage');
% fprintf("%f\t%f\n", norm(w0(:,1)), norm(w0(:,2)));
% 
% if (norm(w0(:,1)) < norm(w0(:,2))) && (norm(B(:,1)) < norm(B(:,1)))
%     fprintf("Matched\n")
% else
%     fprintf("Switch\n")
    
% Blue = 255/l(1) * X(1,:);
% Pink = 255/l(2) * X(2,:);
Blue = 255/max(X(1,:)) * X(1,:);
Pink = 255/max(X(2,:)) * X(2,:); 


C1 = reshape(Blue, s(1), s(2));
C2 = reshape(Pink, s(1), s(2));

C1 = medfilt2( C1,[2 2]);
C2 = medfilt2( C2,[2 2]);
end
