% Missouri University of Science and Technology
% October 15 2014
% This function takes 2 arguments 
%       - the directory where the image is located 
%       - image filename
%
% returns 4 variables
%       - original_img_rotated: the original image after applying the mask 
%         and rotating it for proper segmentation
%
%       - final_image: binary mask which will be used in segmenting the 
%         image and extracting the features
%
%       - Eccentricity: the eccentricity of the image which will be used to 
%         determaine the medial axis method to apply
%       - elapsed_time: the time it take the function to run
%
% This function assumes a subfolder named Mask is located in the image 
% folder and contans the mask of the image that have name as the image file
% name with _mask at the end
%
% last updated: May 17, 2015
%               using the new method for applying the mask which is faster



function [original_img_rotated,Img,final_image,Eccentricity,elapsed_time] = getMaskedImage(rootdir,rootd_m,imgfn)
%clear all
%close all
tic
%clc
warning off

    % set the path and file names of the original image and the mask this
    maskdir='Mask';
    % Get mask file name
    maskfn =[imgfn(1:end-4),'_mask.tif'];
    % Set the full path for for the image and maske
    imgpath = [rootdir filesep  imgfn];
    maskpath = [rootd_m filesep  maskdir filesep  maskfn];

    % Read the image and the mask    
    Mask=(imread(maskpath));
    Img=(imread(imgpath));    
    
    final_image  = Mask>0;
    img_color_output = Img .* repmat(uint8(final_image), [1 1 3]);
    
 
    % create a new image with only the segmented area

    img= img_color_output;
    
    rotAngle=regionprops(final_image,'Orientation','Area', 'Eccentricity');
    %disp(rotAngle.Orientation);
    [~, ind_ar] = sort([rotAngle.Area],'descend');
    %disp(ind_ar);

    final_image=imrotate(final_image,-rotAngle(ind_ar(1)).Orientation); % rotate mask
    final_image = final_image > 0;
%     imshow(final_image);
    original_img_rotated=imrotate(img,-rotAngle(ind_ar(1)).Orientation); % rotate original image
    Img = imrotate(Img,-rotAngle(ind_ar(1)).Orientation);
%     figure;
%     imshow(original_img_rotated);
    Eccentricity = rotAngle(ind_ar(1)).Eccentricity;
    %disp(Eccentricity);
    elapsed_time = toc;
    elapsed_time = num2str(elapsed_time);
