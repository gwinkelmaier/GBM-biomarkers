function decompose_patches_ruifrok( folder )
addpath(pwd);
cd(folder);
files = dir('20220110-162436/*.png');
length(files)
base_name = split(files(1).name, '_');
base_name = base_name{1};
for i=1:numel(files)
%     M = imread([files(i).folder '/' files(i).name]);
%     M = M>0.5*255.0;
    I = imread(['patches/', base_name, '/', base_name, '_tiles/', files(i).name]);

    try
%         D = he_decompose(I,M);
        D = ruifrok_decompose(I);
        C1 = D(:,:,1);
        C2 = D(:,:,2);
        C3 = D(:,:,3);
        save(['decompose-ruifrok/' strrep(files(i).name, 'png','mat')], 'C1','C2','C3');
%         imwrite(c1, ['decompose_ch1/' files(i).name]);
%         imwrite(c2, ['decompose_ch2/' files(i).name]);
    catch
        continue
    end
end
end

function D=ruifrok_decompose(I)
I = double(I);
W = [[1.88 -0.07 -0.60];
     [-1.02 1.13 -0.48];
     [-0.55 -0.13 1.57]];
[m, n, k] = size(I);
D = reshape(I, [m*n, 3]) * W;
D = reshape(D, [m, n, k]);
end
