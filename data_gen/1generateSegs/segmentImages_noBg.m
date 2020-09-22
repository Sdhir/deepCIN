% Missouri University of Science and Technology
% June 28 2017
% This function takes 6 arguments an image that have been masked, the
% binary mask of the image, the eccentricity, the directory where the image
% is located, the filename of the image, and number of segments.
% it produces 3 outputs 
% segmentedImages: a cell array containing the 5 segments of the image
% final_features1: a matrix containing all the extracted features of the
% image
% MedialAxis: the medial axis of the image
%
% This function assumes a subfolder named segments is located in the image 
% folder it will store an excel file of the extracte and will also save the
% image segments
%



%%

function outputsegments = segmentImages_noBg(original_img_rotated,Img,final_image,nn,Eccentricity,seg_width,out_DIR,imname,do_resize,numrows,numcols,factor_distance,display,saving)

tic
%clc
if nargin < 13
    saving = 1;
end
warning off  

set_span=0.73; % this is the only parameter in the code. Usually the range was found to be optimal between 0.7 and 0.8
shift_val=250; % the zero padding on top and bottom of segment to make sure that all the region gets covered due to rotation
seg_folder = 'sz64Segs_new';
I_class=class(original_img_rotated);
% outputsegments store the segments of the images to be displayed in the
% image 


% root directory where the vertical segments are stored
rootdir_segmented_images=[out_DIR filesep seg_folder filesep]; 

if isequal(exist(rootdir_segmented_images,'dir'),7) % 7 means it's a dir.
  
else % create a new directory to store  the images
   mkdir(rootdir_segmented_images)
end
% imgfn the image file name
imgfn = imname;
    %% This part finds the medial axis of image 
    
       
    if (Eccentricity)<set_span
        % find the medial axis using the ratio of nuclei approach if
        % Eccentricity is less than set_span threshold
        % This also segments the images and return them in final_box
        num_method=1;
        [newMedialAxis]=controlpointsrotation(final_image,original_img_rotated,nn,set_span); % find the medial axis using the ratio of nuclei approach
        %num_blk = ceil(size(newMedialAxis,1)/64);
        %disp(num_blk);
        [final_box,seg_num]=createSegment(newMedialAxis,seg_width,nn,final_image,shift_val,num_method,factor_distance,display); % create the vertical segments. The outputs contain the coordinates of the ten vertical segments.
        
      
    else
        % find the medial axis using the distance transform approach if
        % Eccentricity is greater than set_span threshold 
        % This also segments the images and return them in final_box
        num_method=2;
        [newMedialAxis]=BoundingBoxProcessnew(final_image,nn);
        %num_blk = ceil(size(newMedialAxis,1)/64);
        %disp(num_blk);
        [final_box,seg_num]=createSegment(newMedialAxis,seg_width,nn,final_image,shift_val,num_method,factor_distance,display); % create the vertical s
        
    end
    
    outputsegments = cell(seg_num);
    
    % This loop will extract and save all segments
    seg_label=zeros(size(original_img_rotated,1),size(original_img_rotated,2));
    for save_arr=1:seg_num-1
        temp_im_loc=cell2mat(final_box(save_arr));
        BW_small = poly2mask(temp_im_loc(1,:), temp_im_loc(2,:), size(original_img_rotated,1), size(original_img_rotated,2));

        if (size(original_img_rotated,3)==3)
            BW_small(:,:,2)=BW_small(:,:,1);
            BW_small(:,:,3)=BW_small(:,:,1);
        end

        if I_class=='uint8'
            BW_small=uint8(BW_small);
        elseif I_class=='double'
            BW_small=double(BW_small);
        elseif I_class=='uint16'
            BW_small=uint16(BW_small);
        end
        
        BW_small(BW_small==1)=original_img_rotated(BW_small==1);
        BW_small_1=zeros(size(original_img_rotated,1),size(original_img_rotated,2));
        BW_small_1(BW_small(:,:,1)>0)=1;
       
        if (sum(sum(sum(BW_small)))~=0)
            
            stats_small=regionprops(BW_small_1,'Orientation');
            img_small_1 = original_img_rotated .* uint8(repmat(BW_small_1, [1 1 3])); %get titled segment region
            img_small_1_rotate = imrotate(img_small_1,90-stats_small.Orientation);
            Img_rotate = imrotate(Img,90-stats_small.Orientation);
            stats_img_small=regionprops(img_small_1_rotate(:,:,1)>0,'BoundingBox','Area');
            small_area = cat(1, stats_img_small.Area);
            [~,argmax] = max(small_area);
            small_BoundingBox=  round(cat(1, stats_img_small.BoundingBox));
            small_BoundingBox_1 = small_BoundingBox(argmax,:);
            small_img=Img_rotate(small_BoundingBox_1(2):small_BoundingBox_1(2)+small_BoundingBox_1(4),small_BoundingBox_1(1):small_BoundingBox_1(1)+small_BoundingBox_1(3),:);
            
            
        else
            small_img = magic(2);
            small_img(:,:,2) = magic(2);
            small_img(:,:,3) = magic(2);
        end
%         disp(save_arr);
        [a,b,~] = size(small_img);
        [c,d,~] = size(BW_small);
        % check for edge black image and very small image
        % return zeros for the features 
        if sum(sum(sum(BW_small)))> 5000 % (((a*b)/(c*d)>0.005) && sum(sum(sum(BW_small)))~=0)
            % Save the segments into the segments subfolder
            if (saving == 1)    
%                 disp(save_arr);
                mkdir(num2str(rootdir_segmented_images),imgfn(1:end-4))
                imgFileName = sprintf(strcat(imgfn(1:end-4),'_seg_%03d.tif'),save_arr);
                if do_resize
                    resized_small_img = imresize(small_img,[numrows numcols],'bicubic');
                    imwrite(resized_small_img,[[num2str(rootdir_segmented_images),imgfn(1:end-4)],filesep,imgFileName],'tif')
                else
                    imwrite(small_img,[[num2str(rootdir_segmented_images),imgfn(1:end-4)],filesep,imgFileName],'tif')
%                 imwrite(BW_small_1,[[num2str(rootdir_segmented_images),imgfn(1:end-4)],filesep,imgfn(1:end-4),'_mask_',num2str(save_arr),'.jpg'],'jpg')
                end
            outputsegments{save_arr} = small_img;
            end
        BW_small_1(BW_small_1(:,:,1)>0)=save_arr;
        seg_label = seg_label+BW_small_1;
        end
    if isequal(exist([num2str(rootdir_segmented_images),imgfn(1:end-4)],'dir'),7) % 7 means it's a dir.
    else % create a new directory to store  the images
       mkdir(num2str(rootdir_segmented_images),imgfn(1:end-4))
    end
    % save([[num2str(rootdir_segmented_images),imgfn(1:end-4)],filesep,imgfn(1:end-4),'_seg_',num2str(seg_num-1),'labels.mat'], 'seg_label');
    %figure,imagesc(seg_label);
   
      
    
    clear arr_new
    clear v
    clear arr
    clear x_rotated y_rotated p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 ind_ar final_Or final_features
 

end
