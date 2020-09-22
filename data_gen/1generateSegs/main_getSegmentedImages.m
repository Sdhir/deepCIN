% Compute the time requred to extract features and store it in xlsx file

startime = tic;

% **NOTE: if generating overlapping segs, uncomment code from line 126 in createSegments.m file

seg_width = 64;
factor_distance = 1; % to factor segment widths
segs_with_noBg = 1;
seg_folder = 'sz64Segs_';
display = 0;
saving = 1;

do_resize = 0;
numrows = 704;
numcols = 64;

out_DIR = '/usr/local/home/ssbw5/classification/deep-cin_expts/data';

rootdir_62imags = '/usr/local/home/ssbw5/R_DRIVE/dataSets/Images62';% root directory 62 images
rootdir_72imags = '/usr/local/home/ssbw5/R_DRIVE/dataSets/Images72';% root directory 71 images
rootdir_50_imagesApril2015='/usr/local/home/ssbw5/R_DRIVE/dataSets/Images50';% root directory 50 images cropped in April 2015
rootdir_74='/usr/local/home/ssbw5/R_DRIVE/dataSets/Images74';% root directory images cropped in April 2016
rootdir_202imags = '/usr/local/home/ssbw5/R_DRIVE/dataSets/Images202';

imnames_62 = dir([rootdir_62imags filesep '*.jpg']); % old DB 62 images
imnames_72 = dir([rootdir_72imags filesep '*.tif']); % New DB 71 images
imnames_50 = dir([rootdir_50_imagesApril2015 filesep '*.tif']); % New DB 50 images of April
imnames_74 = dir([rootdir_74 filesep '*.tif']); % New DB 76 images of April 2016
imnames_202 = dir([rootdir_202imags filesep '*.tif']);
%image_set = 0;
% set the image dataset to be used
 
% image_set = 62;
% image_set = 72;
% image_set = 50;
% image_set = 74;
%image_set = 202;

tic
image_sets = [62, 72, 50, 74, 202];

for image_set = image_sets
    
    switch image_set
        case 62
            % The 62 images dataset
            imNos = [1:7 9:62];
            rootd = rootdir_62imags;
            imnames = imnames_62;
            filePrefix = '62Set';
        case 72
            % The 71 images dataset
            imNos = 1:length(imnames_72); %[1:44 46:72]; %71 imageset image 45 was having problem have problems 
            rootd = rootdir_72imags;
            imnames = imnames_72;
            filePrefix = '72Set';
        case 50
            % The 50 images dataset
            imNos = 1:length(imnames_50);
            rootd = rootdir_50_imagesApril2015;
            imnames = imnames_50;
            filePrefix = '50Set';

        case 74
            % The 74 images dataset
            imNos =[1:54 56:length(imnames_74)]; % bad image - 55 
            rootd = rootdir_74;
            imnames = imnames_74;
            filePrefix = '74Set';
        case 202
            % The 202 images dataset
            imNos = [1 3:5 7:48 50:162 164:202]; 
            rootd = rootdir_202imags;
            imnames = imnames_202;
            filePrefix = '202Set';
        % if image set is not set correctly the old 62 image set will be used
        otherwise
            % The 62 images dataset
            imNos = 1:62;
            rootd = rootdir_62imags;
            imnames = imnames_62;
            filePrefix = '62Set';
    end

    %% save all segments
    parfor idx = 1:numel(imNos)
        nn = imNos(idx);
        imname = imnames(nn).name;
        %disp(['Getting ' num2str(num_blk) ' segments of image: ',num2str(nn),' from the ',num2str(image_set) ' image set'])
        % Use appropriate mask (xml or tiff) depending on what image set is passed
        if image_set == 62
            [original_img_rotated,Img,final_image,Eccentricity,~] = getMaskedImageXML(rootd,rootd,imname);
        else
            [original_img_rotated,Img,final_image,Eccentricity,~] = getMaskedImage(rootd,rootd,imname);
        end
        % the [~] will not return the segments just save them in disk
        if display
            figure(nn);imshow(original_img_rotated); hold all;
        end
        if segs_with_noBg
            % without background blacked out
            [~] = segmentImages_noBg(original_img_rotated,Img,final_image,nn,Eccentricity,seg_width,out_DIR,imname,do_resize,numrows,numcols,factor_distance,display,saving);
        else
            % with background blacked out
            [~] = segmentImages(original_img_rotated,final_image,nn,Eccentricity,seg_width,out_DIR,seg_folder,imname,do_resize,numrows,numcols,factor_distance,display,saving);
        end

    end
end
toc

% %% display an images
% nn = 1; % image to be displayed from current set 
% if 1  % 0 will not execute 1 will execute
%     close all
% %     iptsetpref('ImshowBorder','tight');
%     iptsetpref('ImtoolInitialMagnification','fit');
%     saving = 0; % don't save segment, for display only 
%     imname = imnames(nn).name;
% 
%     % Use appropriate mask (xml or tiff) depending on what image set is passed
%     [original_img_rotated,final_image,Eccentricity,elapsed_time_mask] = getMaskedImage(rootd,imname);
% 
%     segments = segmentImages(original_img_rotated,final_image,Eccentricity,rootd,imname,num_blk,saving);
% 
%     figure
%     subplot(2,num_blk,1:num_blk)
%     imshow(original_img_rotated)
%     title(imname)
% 
%     for i =1:num_blk
%         subplot(2,num_blk,num_blk+i)
%         imshow(uint8(segments{i}),[]);
%     end
% end 