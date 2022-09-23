function decompose_patches( folder )
% Set up env
addpath(genpath(pwd));
addpath(genpath('SNMF stain separation and color normalization'));
cd spams-matlab-v2.6
start_spams
cd ..


% Set-up parameters
nstains=2;
lambda=0.02;
norm_count = 0;
decomp_count = 0;

% Target Images
target = imread('data/target1.png');
[Wi, Hi,Hiv]=stainsep(target,nstains,lambda);

cd(folder);
files = dir('20220110-162436/*.png');
base_name = split(files(1).name, '_');
base_name = base_name{1};
% for i=1:numel(files)
for i=300
    M = imread([files(i).folder '/' files(i).name]);
    M = M>0.5*255.0;
    I = imread(['patches/', base_name, '/', base_name, '_tiles/', files(i).name]);

    try
        % Normalize the image
%         [Wis, His, Hivs]=stainsep(I,nstains,lambda);
%         [norm]=SCN(I,Hi,Wi,His);
%         imwrite(norm, ['normalized/' files(i).name])
%         norm_count = norm_count + 1;
        [D, w, w_start] = he_decompose(I, M);
        [w2, H2, Hv2] = stainsep(I, nstains, lambda);
        [Wis, His, Hivs]=stainsep(I,nstains,lambda);
        [norm]=SCN(I,Hi,Wi,His);
        [WWis, HHis, HHivs]=stainsep(norm,nstains,lambda);
        
        
        %%%%%%%%%%%%%%%%
        D_old = load(['decompose/',  strrep(files(i).name, 'png', 'mat')]);
        C_1 = D(:, :, 1);
        C_2 = D(:, :, 2);
        
        CC_1 = H2(:, :, 1);
        CC_2 = H2(:, :, 2);
        
        CCC_1 = D_old.C1;
        CCC_2 = D_old.C2;
        
        CCCC_1 = HHis(:, :, 1);
        CCCC_2 = HHis(:, :, 2);
        
        figure(1), imshowpair(C_1, C_2, 'montage');
        figure(2), imshowpair(CC_1, CC_2, 'montage');
        figure(3), imshowpair(CCC_1, CCC_2, 'montage');
        figure(4), imshow(I, []);
        figure(5), imshow(M, []);
        figure(6), imshowpair(CCCC_1, CCCC_2, 'montage');
        %%%%%%%%%%%%%%%%
        
        D_flat = reshape(D, [224*224, 2]);
        norm = D_flat * Wi';
        norm = reshape(norm, [224, 224, 3]);
%         norm = norm / max(norm(:));

        % Decompose Normalized Image
        D = he_decompose(norm,M);
        C1 = D(:,:,1);
        C2 = D(:,:,2);
%         save(['decompose-norm/' strrep(files(i).name, 'png','mat')], 'C1','C2');
        decomp_count = decomp_count + 1;
    catch
        continue
    end
end

size(files)
norm_count
decomp_count
end
