close all; clear all; clc

my_dirs = dir('data/Images/image*.bmp');

for img_number=[27:length(my_dirs)]
    try
        I = double(imread(['data/Images/image' num2str(img_number) '.bmp']));
        M = imread(['data/Masks/mask' num2str(img_number) '.bmp']);
    catch
        disp(['Skipping ' num2str(img_number)]);
        continue
    end
    
    [w0, ch1, ch2] = decompose_threshold(I,M);
    
    subplot(221)
    imshow(uint8(I), [])
    title('Original Image');
    subplot(222)
    imshow(M, [0 1])
    title('Mask');
    
    subplot(223)
    imshow(ch1, [])
    title(['min: ' num2str(min(ch1(:))) ' | max: ' num2str(max(ch1(:))) ])
    subplot(224)
    imshow(ch2, [], 'border','tight')
    title(['min: ' num2str(min(ch2(:))) ' | max: ' num2str(max(ch2(:))) ])
    saveas(gcf, ['output-thresh/decomposed_' num2str(img_number) '.fig'])
    saveas(gcf, ['output-thresh/decomposed_' num2str(img_number) '.png'])
end