close all; clear all; clc

my_dirs = dir('data/Images/image*.bmp');
bins = [0:10:260];
for img_number=[1,5,9,20,23]
     I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
    M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
    
    [w0, ch1, ch2] = decompose(I,M);
    [w0_2, ch1_2, ch2_2] = decompose_threshold(I,M);
    
    % Plot Figure and Mask
    figure(1), imshow(uint8(I), [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_image.png'], '-dpng', '-r300')
    figure(1), clf, imshow(uint8(M), [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_mask.png'], '-dpng', '-r300')
    
    % Create Histogram plot
    mask = zeros(size(I));
    mask(I>200) = 1;
    mask = sum(mask, 3);
    mask = mask ~= 3;
    R = I(:,:,1);
    G = I(:,:,2);
    B = I(:,:,3);
    fg_r = R(M & mask);
    fg_g = G(M & mask);
    fg_b = B(M & mask);
    bg_r = R(~M & mask);
    bg_g = G(~M & mask);
    bg_b = B(~M & mask);
    figure(1), clf
    subplot(1,2,1), hold on
    histogram(fg_r, bins, 'FaceColor','r','Normalization','probability');
    histogram(fg_g, bins, 'FaceColor','g','Normalization','probability');
    histogram(fg_b, bins, 'FaceColor','b','Normalization','probability');
    hold off
    subplot(1,2,2), hold on
    histogram(bg_r, bins, 'FaceColor','r','Normalization','probability');
    histogram(bg_g, bins, 'FaceColor','g','Normalization','probability');
    histogram(bg_b, bins, 'FaceColor','b','Normalization','probability');
    hold off
    print(gcf, ['exp/' num2str(img_number) '_hist.png'], '-dpng', '-r300');
    
    % Plot decompose before
    [~, ch1, ch2] = decompose(I,M);
    figure(1), clf
    figure(1), clf, imshow(ch1, [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_decompose1_prior_ch1.png'], '-dpng', '-r300');
    figure(1), clf, imshow(ch2, [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_decompose1_prior_ch2.png'], '-dpng', '-r300');
    
    % Plot decompose after
    [~, ch1, ch2] = decompose_threshold(I,M);
    figure(1), clf
    figure(1), clf, imshow(ch1, [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_decompose2_prior_ch1.png'], '-dpng', '-r300');
    figure(1), clf, imshow(ch2, [], 'border', 'tight');
    print(gcf, ['exp/' num2str(img_number) '_decompose2_prior_ch2.png'], '-dpng', '-r300');
end

% %%
% % for img_number=[9:length(my_dirs)]
% count = 1;
% for img_number=[9,22]
%     I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
%     M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
%     
%     [w0, ch1, ch2] = decompose(I,M);
%     broken_class{count} = w0;
%     w0
%     abs(w0(:,1) - w0(:,2))
%     count = count+1;
%     
%     %
%     %     subplot(221)
%     %     imshow(uint8(I), [])
%     %     title('Original Image');
%     %     subplot(222)
%     %     imshow(M, [0 1])
%     %     title('Mask');
%     %
%     %     subplot(223)
%     %     imshow(ch1, [])
%     %     title(['min: ' num2str(min(ch1(:))) ' | max: ' num2str(max(ch1(:))) ])
%     %     subplot(224)
%     %     imshow(ch2, [], 'border','tight')
%     %     title(['min: ' num2str(min(ch2(:))) ' | max: ' num2str(max(ch2(:))) ])
%     %     saveas(gcf, ['output/decomposed_' num2str(img_number) '.fig'])
%     %     saveas(gcf, ['output/decomposed_' num2str(img_number) '.png'])
% end
% 
% count = 1;
% for img_number=[1,2]
%     if ismember( img_number, [9,23,26] )
%         continue
%     end
%     I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
%     M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
%     
%     [w0, ch1, ch2] = decompose(I,M);
%     working_class{count} = w0;
%     w0
%     abs(w0(:,1) - w0(:,2))
%     count = count + 1;
%     
%     %     subplot(221)
%     %     imshow(uint8(I), [])
%     %     title('Original Image');
%     %     subplot(222)
%     %     imshow(M, [0 1])
%     %     title('Mask');
%     %
%     %     subplot(223)
%     %     imshow(ch1, [])
%     %     title(['min: ' num2str(min(ch1(:))) ' | max: ' num2str(max(ch1(:))) ])
%     %     subplot(224)
%     %     imshow(ch2, [], 'border','tight')
%     %     title(['min: ' num2str(min(ch2(:))) ' | max: ' num2str(max(ch2(:))) ])
%     %     saveas(gcf, ['output/decomposed_' num2str(img_number) '.fig'])
%     %     saveas(gcf, ['output/decomposed_' num2str(img_number) '.png'])
% end
% 
% %%
% for img_number = [1,2,9,22,23]
%     I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
%     M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
%     
%     
%     w0 = NMF_Initial(I,M);
%     mask = zeros(size(I));
%     mask(I>200) = 1;
%     mask = sum(mask, 3);
%     mask = mask ~= 3;
%     
%     R = I(:,:,1);
%     G = I(:,:,2);
%     B = I(:,:,3);
%     fg_r = R(M & mask);
%     fg_g = G(M & mask);
%     fg_b = B(M & mask);
%     bg_r = R(~M & mask);
%     bg_g = G(~M & mask);
%     bg_b = B(~M & mask);
%     
%     figure(1), clf
%     subplot(1,2,1), hold on
%     histogram(fg_r, 'FaceColor','r','Normalization','pdf');
%     histogram(fg_g, 'FaceColor','g','Normalization','pdf');
%     histogram(fg_b, 'FaceColor','b','Normalization','pdf');
%     hold off
%     subplot(1,2,2), hold on
%     histogram(bg_r, 'FaceColor','r','Normalization','pdf');
%     histogram(bg_g, 'FaceColor','g','Normalization','pdf');
%     histogram(bg_b, 'FaceColor','b','Normalization','pdf');
%     hold off
%     
%     saveas(gcf, ['exp/hist_' num2str(img_number) '.fig']);
% end
% 
% %% Get median separation statistic
% diff = zeros(numel(my_dirs), 3);
% for img_number=[1:numel(my_dirs)]
%     try
%         I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
%         M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
%     catch
%         continue
%     end
%     
%     [w0, ch1, ch2] = decompose(I,M);
%     d = w0(:,1) - w0(:,2);
%     diff(img_number,:) = d;
% end
% diff(26, :) = [];
% median(diff, 1)
% 
% %% Compare methods
% img_number = 23;
% I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
% M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
% 
% [w0_1, ch1_1, ch2_1] = decompose(I,M);
% [w0_2, ch1_2, ch2_2] = decompose_threshold(I,M);
% 
% figure(1), clf
% subplot(2,2,1)
% imshow(ch1_1);
% subplot(2,2,2)
% imshow(ch1_2);
% subplot(2,2,3)
% imshow(ch2_1);
% subplot(2,2,4)
% imshow(ch2_2);
% 
% figure(2), clf
% imshow(uint8(I));