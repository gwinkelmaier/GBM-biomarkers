% Function Name: he_decompose
% author: Garrett Winkelmaier
% date: 08.03.2021
%
% Custom NMF Solver for H&E tissue samples
% 
% min || A - W*H ||
%    A(3xN) - RGB (or O.D.) vectorized image
%    W(3x2) - Stain mixing matrix
%    H(2xN) - H&E deconvolved image
% 
% Inputs:  I - Image (rgb or o.d.)
%          M - Mask (nuclear mask) binary
%          n_iter - number of iterations (defaults to 5)
% Outputs: D - Deconvolved Image (H&E channels)
%          w - decomposition matrix

function [D, w, win] = he_decompose(I, M, n_iter)

I = double(I);
M = double(M);

% Create Seeds (intialize w matrix)
w = he_seeds(I, M);
win = w;

% Initialize Formulation
s = size(I);
A = permute(I, [3,1,2]);
A = reshape(A, [3, s(1)*s(2)]);

if ~exist('n_iter')
    n_iter = 5;
end

% % lslin() parameters
% C = w;
% d = A;
% 
% lsqlin(C,d,[],[]);

% Iterate
for i=1:n_iter
    % Solve For H
    H = w\A;
    
    % Solve For W
    w = (H' \ A')';
    
    % Compute error
    % e = A-w*H; 
    % E = norm(e);
end

% Format the output
D = permute(H, [2,1]);
D = reshape(D, s(1), s(2), 2);
C1 = medfilt2(D(:,:,1), [5,5]);
C2 = medfilt2(D(:,:,2), [5,5]);

% Shift Channels to positive range
C1 = C1 - min(C1(:));
C2 = C2 - min(C2(:));

% Return
D = cat(3, C1, C2);
end
