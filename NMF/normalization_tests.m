for i = 1:31
  image_name = ['image' num2str(i) '.bmp'];
  
  % Read Image Examples
  try
  I = imread(['data/Images/' image_name]);
  M = imread(['data/Masks/' strrep(image_name, 'image', 'mask')]);
  N = imread(['data/Normalized/' image_name]);
  catch
    continue
  end

  f = figure('visible', 'off');
  f.Position = [100 100 540 400];
  
  % Get W matrix of I, M pair
  [D, w] = he_decompose(I, M);
  C1 = squeeze(D(:, :, 1));
  C2 = squeeze(D(:, :, 2));
  subplot(2,3,1)
  imshow(I, [], 'border', 'tight')
  subplot(2,3,2)
  imshow(C1, [], 'border', 'tight')
  subplot(2,3,3)
  imshow(C2, [], 'border', 'tight')
  
  % Get W matrix of N, M pair
  [D2, w2] = he_decompose(N, M);
  C1 = squeeze(D2(:, :, 1));
  C2 = squeeze(D2(:, :, 2));
  subplot(2,3,4)
  imshow(N, [], 'border', 'tight')
  subplot(2,3,5)
  imshow(C1, [], 'border', 'tight')
  subplot(2,3,6)
  imshow(C2, [], 'border', 'tight')

  % Save figure
  set(gcf,'PaperUnits','inches');
  set(gcf,'PaperSize', [16 8]);
  set(gcf,'PaperPosition',[-1 0 18 8]);
  set(gcf,'PaperPositionMode','Manual');
  saveas(gcf, ['norm_test/test' num2str(i) '.jpg'])
  
  % Compare
  wList{i}.w = w;
  wList{i}.w2 = w2;
end

% Save results
txt = jsonencode(wList);
fprintf(fopen('w_comparison.json', 'w'), txt);
fclose('all')


