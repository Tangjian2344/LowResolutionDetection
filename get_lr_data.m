close all;
clc;
scale = 4;
src_path = '/home/tangjian/Codes/VOC/images512/';
datasets = {'test2007', 'train2007', 'train2012'};

for idx_set = 1:length(datasets)
    fprintf('Processing %s ....\n', datasets{idx_set});
	files = dir(fullfile(src_path, datasets{idx_set}, '*.jpg'));
    c = 123;
    for idx_file = 1: length(files)
        file_name = files(idx_file).name;
        img_path = fullfile(src_path, datasets{idx_set}, file_name);
        img = imread(img_path);
        lr_img = imresize(img, 1/scale,'bicubic');
        save_path =  replace(img_path, datasets{idx_set}, [datasets{idx_set}, '_lr']);
        imwrite(lr_img, save_path);
    end
end










