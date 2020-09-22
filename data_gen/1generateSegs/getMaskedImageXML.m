% Missouri University of Science and Technology
% May 17 2015
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
% This function assumes a subfolder named refinedboundaries is located in 
% the image folder and contans the mask of the image that have name as the 
% image file name with with XML extension
% it uses cebReadITSCoords function to read the XML file containing
% polygon coordinates of the epithelium region.


function [original_img_rotated,Img,final_image,Eccentricity,elapsed_time] = getMaskedImageXML(rootdir,rootd_m,imgfn)
%clear all
%close all
tic
%clc
warning off


 % set the path and file names of the original image and the mask this
    maskdir='refinedboundaries';
    % Get mask file name
    xmlfn =[imgfn(1:end-4),'.xml'];
    % Set the full path for for the image and maske
    imgpath = [rootdir filesep imgfn];
    xmldir = [rootd_m filesep maskdir];
    % Read the image and the mask    
    I0=(imread(imgpath));
    [x,y]=cebReadITSCoords(xmldir,xmlfn);
    I_class=class(I0);
    BW1 = poly2mask(x, y, size(I0,1), size(I0,2));

    I_c=I0;
    I=rgb2gray(I0);
    
    A=BW1;
    if I_class=='uint8'
        A=uint8(A);
    elseif I_class=='double'
        A=double(A);
    elseif I_class=='uint16'
        A=uint16(A);
    else
        disp('unknown Image format.. continuing with next image')
        system.exit(1)
    end
    
    A=A.*I;
    AT=A;
%    imshow(AT);
    AT(AT>0)=I_c(AT>0);
%     imshow(AT);
    AT(:,:,2)=AT(:,:,1);
%     imshwo(AT);
    AT(:,:,3)=AT(:,:,1);
%     imshow(AT);
    AT(AT>0)=I_c(AT>0);
    
    img=AT;
    
    final_image=BW1;
    
    BW1=imfill(BW1,'holes');
%     imshow(BW1);
    
    rotAngle=regionprops(final_image,'Orientation','Area','Eccentricity');
    [~, ind_ar] = sort([rotAngle.Area],'descend');
    [L,num]=bwlabel(final_image);
    L(L~=ind_ar(1))=0;
    final_image1=im2bw(L);
    mask_original=final_image;
    mask_index=ind_ar(1);
    final_image=imrotate(final_image,-rotAngle(ind_ar(1)).Orientation); % rotate mask
    original_img_rotated=imrotate(img,-rotAngle(ind_ar(1)).Orientation); % rotate original image
    Img = imrotate(I0,-rotAngle(ind_ar(1)).Orientation); % rotate original image
    Eccentricity = rotAngle(ind_ar(1)).Eccentricity;
    
    
 elapsed_time = toc;
 elapsed_time = num2str(elapsed_time);
    