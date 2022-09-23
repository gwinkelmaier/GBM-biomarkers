% List of rgb images
f = dir('data/Images/*.bmp');

% Set-up parameters
nstains=2;
lambda=0.02;

% Target Images
target = imread('data/target1.png');
[Wi, Hi,Hiv]=stainsep(target,nstains,lambda);

% Normalize and save each rgb image
for i=1:numel(f)
    source = imread([f(i).folder '/' f(i).name]);
    [Wis, His,Hivs]=stainsep(source,nstains,lambda);
    [norm]=SCN(source,Hi,Wi,His);
    imwrite(norm, ['data/Normalized/' f(i).name])
end
