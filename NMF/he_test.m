close all; clear all;

%% Synthetic Data
% red = cat(3, ones(100,100), zeros(100,100), zeros(100,100));
% blue = cat(3, zeros(100,100), zeros(100,100), ones(100,100));
% I = cat(2, red, blue);
% M = zeros(100, 200);
% x0 = 50; y0 = 50;
% M(x0, y0) = 1;
% for r=1:10
%     for theta=1:360
%         x = round(x0 + r*cos(theta));
%         y = round(y0 + r*sin(theta));
%         M(x,y) = 1;
%     end
% end
% imshowpair(I,M,'montage')

%% Real Data
% img = 31;
% for img=12
%     if img==26
%         continue
%     end
%     fprintf("Image %d\n", img);
%     
%     % Read Data
%     I = imread(['data/Images/image' num2str(img) '.bmp']);
%     M = imread(['data/Masks/mask' num2str(img) '.bmp']);
%     
%     % Call he_decompose
%     D = he_decompose(I, M, 10);
%     C1 = D(:,:,1); C2 = D(:,:,2);
%     fprintf("Ch1: [%0.2f, %0.2f]\tCh2: [%0.2f, %0.2f]\n",  ...
%         min(C1(:)), max(C1(:)), min(C2(:)), max(C2(:)));
%     
%     figure, imshowpair(D(:,:,1), D(:,:,2), 'montage');
% %     x = input("continue? ", 's');
% end

%%
d_min = Inf; d_max = 0;
dd_min = Inf; dd_max = 0;
for img=1:31
    if img==26
        continue
    end
    % Decompose H&E
    I = imread(['data/Images/image' num2str(img) '.bmp']);
    M = imread(['data/Masks/mask' num2str(img) '.bmp']);
    V = log(255) - log(double(I)+1);
    D = he_decompose(V, M, 5);
    
    % NMF
    w = he_seeds(V, M);
    s = size(V);
%     V = permute(V, [3,1,2]);
%     V = reshape(V, [3, s(1)*s(2)]);
    V = reshape(V, [s(1)*s(2), 3]);
    [W,H] = nnmf(V, 2, 'H0', w');
    
    W1 = reshape(W(:,1), [s(1), s(2)]);
    W1 = medfilt2(W1, [5, 5]);
    W2 = reshape(W(:,2), [s(1), s(2)]);
    W2 = medfilt2(W2, [5 5]);
       
    C1 = D(:,:,1);
    C2 = D(:,:,2);
%     if d_min > min(C1(:))
%         d_min = min(C1(:));
%     end
%     if d_max < max(C1(:))
%         d_max = max(C1(:));
%     end
%     if dd_min > min(C2(:))
%         dd_min = min(C2(:));
%     end
%     if dd_max < max(C2(:))
%         dd_max = max(C2(:));
%     end
    
    % Save Image
%     imshowpair(D(:,:,1), D(:,:,2), 'montage');
    imshowpair(W1, W2, 'montage');
    saveas(gcf, ['exp/od_nmf_normalizedW/montage' num2str(img) '.png']);
%     save(['exp/decomp/decomp' num2str(img) '.mat'], 'C1', 'C2');

    fprintf("image %d:\n", img)
    H
end
figure(1), imshowpair(C1, C2, 'montage');
figure(2), imshowpair(W1, W2, 'montage');
% 
% fprintf("Channel 1: [%0.4f, %0.4f]\n", d_min, d_max);
% fprintf("Channel 2: [%0.4f, %0.4f]\n", dd_min, dd_max);