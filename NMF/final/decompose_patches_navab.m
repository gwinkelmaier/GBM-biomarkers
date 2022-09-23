function decompose_patches_navab( folder )
% Set up env
cur_dir = pwd;

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
[Wtarget, Htarget,Htargetv]=stainsep(target,nstains,lambda);

cd(folder);
files = dir('20220110-162436/*.png');
base_name = split(files(1).name, '_');
base_name = base_name{1};
tic
parfor i=1:numel(files)
    
    
    try
        
        %for i=202
        I = imread(['STBpatches/', base_name, '/', base_name, '_tiles/', files(i).name]);
        
        if std(double(I(:))) < 5
            continue
        end
        
        [Wsource, Hsource, Hsourcev] = stainsep(I, nstains, lambda);
        [norm]=SCN(I,Htarget,Wtarget,Hsource);
        
        [WsourceFinal, HsourceFinal, HsourceFinalv] = stainsep(norm,nstains,lambda);
        
        C1 = HsourceFinal(:, :, 1);
        C2 = HsourceFinal(:, :, 2);
        save_name = ['decompose-navab/' strrep(files(i).name, 'png','mat')];
        custom_save(save_name, C1, C2);
        decomp_count = decomp_count + 1;
    catch
        continue
    end
    
end
toc
size(files)
decomp_count

cd(cur_dir)
end

function custom_save(save_name, C1, C2)
save(save_name, 'C1','C2');
end
