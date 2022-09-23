% files = dir('Images/*.bmp');
close all; imtool close all
folder = '/home/gwinkelmaier/parvin-labs/Histology/mina-data/Images/';
fid = fopen('min-max.log','w');

c1_max = 0;
c1_min = Inf;
c2_max = 0;
c2_min = Inf;

% for i = [18,31]
% for i = 1:31
for i = 5
    if i==26
        continue
    end
    fprintf("Image %02d\t", i);
    
    I = imread([folder 'image' num2str(i) '.bmp' ]);
    M = imread([strrep(folder, 'Images', 'Masks') 'mask' num2str(i) '.bmp']);
    
    % NMF
%     Y = double(I) * [0.55 0.57 0.6; 0.8 0 0.5]';
    [c1, c2] = decompose_threshold(I,M);

    % PLOS decomposition
%     [x,y] = SCD_MA(I);
%     s = size(I);
%     c1 = 255 / (max(x(1,:)) - min(x(1,:))) * (x(1,:) - min(x(1,:)));
%     c2 = 255 / (max(x(2,:)) - min(x(2,:))) * (x(2,:) - min(x(2,:)));
%     c1 = reshape(c1, s(1), s(2));
%     c2 = reshape(c2, s(1), s(2));
%     c1 = medfilt2(c1, [2,2]);
%     c2 = medfilt2(c2, [2,2]);

    % KUMAR decomposition
%     lambda = 0.1;
%     nstains = 2;
%     [Wi, Hi,Hiv,stains]=stainsep(I,nstains,lambda);
%     c1 = Hi(:,:,1);
%     c2 = Hi(:,:,2);
%     c1 = 255 / (max(c1(:)) - min(c1(:))) * (c1 - min(c1(:)));
%     c2 = 255 / (max(c2(:)) - min(c2(:))) * (c2 - min(c2(:)));
%     c1 = medfilt2(c1, [2,2]);
%     c2 = medfilt2(c2, [2,2]);
%     
%     fprintf(fid, "Image %d:\tmin:%0.4f\tmax:%0.4f\tmin:%0.4f\tmax:%0.4f\n", ...
%             i, min(c1(:)), max(c1(:)), min(c2(:)), max(c2(:)));
        
%      x = c1;
%      save(['exp/orig_kumar/decomp' num2str(i) '_ch1.mat'], 'x');
%      x = c2;
%      save(['exp/orig_kumar/decomp' num2str(i) '_ch2.mat'], 'x');

%     figure, imshow(I),
%     hold on, visboundaries(M, 'color', 'y'); hold off
%     figure, imshowpair(c1,c2,'montage');
%     % Get Max and Min
%     if max(c1(:)) > c1_max
%         c1_max = max(c1(:));
%     end
%     if max(c2(:)) > c2_max
%         c2_max = max(c2(:));
%     end
%     if min(c1(:)) < c1_min
%         c1_min = min(c1(:));
%     end
%     if min(c2(:)) < c2_min
%         c2_min = min(c2(:));
%     end
    
%     imshowpair(uint8(c1),uint8(c2),'montage')
%     imwrite(c1, ['exp/orig_kumar/decomp' num2str(i) '_ch1.png']);
%     imwrite(c2, ['exp/orig_kumar/decomp' num2str(i) '_ch2.png']);
end

% fprintf("Global Min/Max:\n");
% fprintf("C1: [%f, %f]\n", c1_min, c1_max);
% fprintf("C2: [%f, %f]\n", c2_min, c2_max);