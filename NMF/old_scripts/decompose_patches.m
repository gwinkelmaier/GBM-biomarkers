function decompose_patches( folder )
addpath(pwd);
cd(folder);
files = dir('predictions/*.png');
files = dir(folder)
for i=1:numel(files)
    M = imread([files(i).folder '/' files(i).name]);
    M = M>0.5*255.0;
    I = imread([strrep(files(i).folder,'predictions','patches_20x_224') '/' files(i).name]);

    [~,c1,c2] = decompose_threshold(I,M);
    imwrite(c1, ['decompose_ch1/' files(i).name]);
    imwrite(c2, ['decompose_ch2/' files(i).name]);
end
end
