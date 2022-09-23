% files = dir('Images/*.bmp');
close all;
folder = '/home/gwinkelmaier/parvin-labs/Histology/mina-data/Images/';

% for i = 1:31
% for i=[1,3,18]
for i = 12
    if i==26
        continue
    end
    I = imread([folder 'image' num2str(i) '.bmp' ]);
    M = imread([strrep(folder, 'Images', 'Masks') 'mask' num2str(i) '.bmp']);
%     C1 = imread(['exp/4/decomp' num2str(i) '_ch1.png']);
%     mean(C1(M))
    [~, c1, c2] = decompose_threshold(I,M);
    imshowpair(uint8(c1),uint8(c2),'montage')
%     imwrite(uint8(c1), ['exp/4/decomp' num2str(i) '_ch1.png']);
%     imwrite(uint8(c2), ['exp/4/decomp' num2str(i) '_ch2.png']);
end