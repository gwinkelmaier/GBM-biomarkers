data_dir = '/home/gwinkelmaier/MILKData/NCI-GDC/GBM/Tissue_slide_image/';
% wsi = '0ace7b57-b721-4fd0-b84a-40ed44a7fc6b';
% wsi = '8f0c5240-f556-49f1-aeb9-64d0e6ea7874';
wsi = 'ebcfcaa3-3703-4998-bc0f-9fe42e6e84d9';
patches_subdir = '2021/patches/**/*_tiles';
mask_subdir = '2021/20220110-162436';
decompose_dir = '2021/decompose-navab';

files = dir([data_dir wsi '/' patches_subdir '/*.png']);
% i=1156;
i=1161;
% i=1163;
% base=2911;
% for j=1:10
    i = base+j;
    M = imread([data_dir wsi '/' mask_subdir '/' files(i).name]);
    M = M>0.5*255.0;
    M = bwareafilt(M, 30);
    I = imread([files(i).folder '/' files(i).name]);
    load([data_dir wsi '/' decompose_dir '/' strrep(files(i).name, 'png', 'mat')]);
    
    figure(1), imshowpair(I, M, 'montage');
    figure(2), imshowpair(C1, C2, 'montage');
    
    figure(3), histogram(C1(M));
    figure(4), histogram(C2(M));
    
    left_sample = median(C1(M))
    right_sample = median(C2(M))
    
%     pause;
% end