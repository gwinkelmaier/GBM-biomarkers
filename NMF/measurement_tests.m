% Test if Image Normalization makes a difference on Chromatin Measurements
clear all; close all; imtool close all

folder = '/home/gwinkelmaier/parvin-labs/Histology/tests/NMF/data';

% Target
I = imread([folder '/target1.png']);
[Wt, Ht, ~, ~] = stainsep(I, 2, 0.2);

% Source Transforms
% for i=1:31
for i=14
    if i==26
        continue
    end
    fprintf("Image %02d\n", i);
    
    I = imread([folder '/Images/image' num2str(i) '.bmp' ]);
    M = imread([folder '/Masks/mask' num2str(i) '.bmp']);
    V = log(255) - log(double(I)+1);    
    [w0, mask] = he_seeds(V, M);
    [Wi, Hi]=stainsep(I, 2, 0.2, w0);
    [our]=SCN(I,Ht,Wt,Hi);
    V2 = log(255) - log(double(our)+1);
    [w0_prime] = he_seeds(V2, M);
    [Wi_prime, Hi_prime]=stainsep(our, 2, 0.2, w0_prime);
    
    g_min = min( min(Hi(:)), min(Hi_prime(:)));
    g_max = max( max(Hi(:)), max(Hi_prime(:)));
    figure(1)
    subplot(2,2,1)
    imshow(Hi(:,:,1), [g_min, g_max]);
    subplot(2,2,2)
    imshow(Hi(:,:,2), [g_min, g_max]);
    subplot(2,2,3)
    imshow(Hi_prime(:,:,1), [g_min, g_max]);
    subplot(2,2,4)
    imshow(Hi_prime(:,:,2), [g_min, g_max]);
%     saveas(gcf, ['exp/normalize_reconstruct/measurement_' num2str(i) '.bmp']);
%     
%     figure(2)
%     subplot(2,2,1)
%     imshow(Hi(:,:,1), []);
%     subplot(2,2,2)
%     imshow(Hi(:,:,2), []);
%     subplot(2,2,3)
%     imshow(Hi_prime(:,:,1), []);
%     subplot(2,2,4)
%     imshow(Hi_prime(:,:,2), []);
    
    c=squeeze(Hi(:,:,1));
    meas.original_nuc(i)=mean(c(M & mask));
    meas.original_cyto(i)=mean(c(~M & mask));
    c=squeeze(Hi_prime(:,:,1));
    meas.reconstructed_nuc(i)=mean(c(M & mask));
    meas.reconstructed_cyto(i)=mean(c(~M & mask));
    
    subplot(2,2,1), title(num2str(meas.original_nuc(i)));
    subplot(2,2,2), title(num2str(meas.original_cyto(i)));
    subplot(2,2,3), title(num2str(meas.reconstructed_nuc(i)));
    subplot(2,2,4), title(num2str(meas.reconstructed_cyto(i)));
    
%     saveas(gcf, ['exp/normalize_reconstruct/measurement_' num2str(i) '.bmp']);
end

%%
[~,nuc_bins] = histcounts(meas.original_nuc);
[~,cyto_bins] = histcounts(meas.original_cyto);
figure(1), clf
subplot(1,2,1)
histogram(meas.original_nuc, nuc_bins);
hold on, histogram(meas.reconstructed_nuc, nuc_bins); hold off
[~,p] = ttest(meas.original_nuc, meas.reconstructed_nuc);
title(sprintf('p-value: %0.2e', p));
legend({'original'; 'reconstructed'});

subplot(1,2,2)
histogram(meas.original_cyto, cyto_bins);
hold on, histogram(meas.reconstructed_cyto, cyto_bins); hold off
[~,p] = ttest(meas.original_cyto, meas.reconstructed_cyto);
title(sprintf('p-value: %0.2e', p));
legend({'original'; 'reconstructed'});