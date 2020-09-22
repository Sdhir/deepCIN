function [newMedialAxis]= BoundingBoxProcessnew(final_image,nn) 

display = 0; % control if you want to display figures

bbox=regionprops(final_image,'BoundingBox','Area');
    maxAr=0;
    for kk=1:numel(bbox)
        maxAr_new=bbox(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    
    
    BoundingBox=  cat(1, bbox(ind_ar).BoundingBox);
    if display
        rectangle('Position',[BoundingBox],'EdgeColor','r');        
    end

    clear ind_ar
    dI1=bwdist(~final_image,'quasi-euclidean');
    
    
    [xx,yy]=(max(dI1));
    zz=zeros(size(dI1));
    for mm=1:size(yy,2)
        arr(mm,:)=[yy(mm),mm];
    end
    arr(find(arr(:,1)==1),:)=[];
    midBox=[BoundingBox(1) BoundingBox(2)+BoundingBox(4)/2 BoundingBox(1)+BoundingBox(3) BoundingBox(2)+BoundingBox(4)/2];
    MidLine_Box=zeros(size(arr));
    MidLine_Box(:,2)=arr(:,2);
    MidLine_Box(:,1)=BoundingBox(2)+BoundingBox(4)/2;

    
    %% 4 rectangle approach: this is used to check the orientation of the image if divided into 4 parts (four corners). The orientation of the corners are checked.
    p1=[BoundingBox(1) BoundingBox(2)];
    p2=[BoundingBox(1)+BoundingBox(3)/2 BoundingBox(2)];
    p3=[BoundingBox(1)+BoundingBox(3) BoundingBox(2)];
    p4=[BoundingBox(1)+BoundingBox(3) BoundingBox(2)+BoundingBox(4)/2];
    p5=[BoundingBox(1)+BoundingBox(3) BoundingBox(2)+BoundingBox(4)];
    p6=[BoundingBox(1)+BoundingBox(3)/2 BoundingBox(2)+BoundingBox(4)];
    p7=[BoundingBox(1) BoundingBox(2)+BoundingBox(4)];
    p8=[BoundingBox(1) BoundingBox(2)+BoundingBox(4)/2];
    p9=[BoundingBox(1)+BoundingBox(3)/2 BoundingBox(2)+BoundingBox(4)/2];
    
    im1=final_image(round(p1(2):p9(2)),round(p1(1):p9(1)));
    im2=final_image(round(p2(2):p4(2)),round(p2(1):p4(1)));
    im3=final_image(round(p8(2):p6(2)),round(p8(1):p6(1)));
    im4=final_image(round(p9(2):p5(2)),round(p9(1):p5(1)));
    or{1}=regionprops(im1,'Orientation','Area');
    or{2}=regionprops(im2,'Orientation','Area');
    or{3}=regionprops(im3,'Orientation','Area');
    or{4}=regionprops(im4,'Orientation','Area');
    for kk=1:4
        maxAr=0;
        for kkk=1:numel(or{kk})
            maxAr_new=or{kk}.Area;
            if maxAr_new>maxAr
                maxAr=maxAr_new;
                ind_ar=kkk;
            end
            
        end
        final_Or(kk)=or{kk}(ind_ar).Orientation;
    end
    final_Or1(nn,:)=final_Or;
    clear ind_ar im1 im2 im3 im4
    %% 5 rectangle approach: this is used to break the image into five vertical segments and then analyze the end two segments to augment the medial axis detection. This is because the medial axis seemed to deviate along the first and fifth rectangles
    %
    p1=[BoundingBox(1) BoundingBox(2)];
    p2=[BoundingBox(1) BoundingBox(2)+BoundingBox(4)];
    
    p3=[BoundingBox(1)+round(0.2*BoundingBox(3)) BoundingBox(2)];
    p4=[BoundingBox(1)+round(0.2*BoundingBox(3)) BoundingBox(2)+BoundingBox(4)];
    
    p5=[BoundingBox(1)+round(0.4*BoundingBox(3)) BoundingBox(2)];
    p6=[BoundingBox(1)+round(0.4*BoundingBox(3)) BoundingBox(2)+BoundingBox(4)];
    
    p7=[BoundingBox(1)+round(0.6*BoundingBox(3)) BoundingBox(2)];
    p8=[BoundingBox(1)+round(0.6*BoundingBox(3)) BoundingBox(2)+BoundingBox(4)];
    
    p9=[BoundingBox(1)+round(0.8*BoundingBox(3)) BoundingBox(2)];
    p10=[BoundingBox(1)+round(0.8*BoundingBox(3)) BoundingBox(2)+BoundingBox(4)];
    
    p11=[BoundingBox(1)+BoundingBox(3) BoundingBox(2)];
    p12=[BoundingBox(1)+BoundingBox(3) BoundingBox(2)+BoundingBox(4)];
    
    
    
    im1=final_image(round(p1(2):p4(2)),round(p1(1):p4(1)));
    im2=final_image(round(p3(2):p6(2)),round(p3(1):p6(1)));
    im3=final_image(round(p5(2):p8(2)),round(p5(1):p8(1)));
    im4=final_image(round(p7(2):p10(2)),round(p7(1):p10(1)));
    im5=final_image(round(p9(2):p12(2)),round(p9(1):p12(1)));
    or{1}=regionprops(im1,'Orientation','Area','Eccentricity');
    or{2}=regionprops(im2,'Orientation','Area','Eccentricity');
    or{3}=regionprops(im3,'Orientation','Area','Eccentricity');
    or{4}=regionprops(im4,'Orientation','Area','Eccentricity');
    or{5}=regionprops(im5,'Orientation','Area','Eccentricity');
    
    clear kk ind_ar
    
    for kk=1:5
        maxAr=0;
        for kkk=1:numel(or{kk})
            maxAr_new=or{kk}.Area;
            if maxAr_new>maxAr
                maxAr=maxAr_new;
                ind_ar=kkk;
            end
            
        end
        final_Or(kk)=or{kk}(ind_ar).Orientation;
    end
    final_Or2(nn,:)=final_Or;
    
    %% modify the earlier medial axis with shift
    
    
    X = arr(:,1);
    Y = arr(:,2);
    
    % create a matrix of these points, which will be useful in future calculations
    v = [X';Y'];
    
    %%  process first rectangle (segment)
    if abs(final_Or1(nn,1))<9
        
        % choose a point on first block which will be the center of rotation
        
        % tempArr=arr(:,1)-MidLine_Box(:,1);
        tempArr=arr(1:round(size(MidLine_Box,1)/5),1)-MidLine_Box(1:round(size(MidLine_Box,1)/5),1);
        tempArr=abs(tempArr);
        [minVal,minPos]=min(tempArr);
        if size(tempArr,1)== minPos % if the position is in the edge move it back
            minPos = minPos -1;
        end
        
        y_center=arr(minPos,1);
        x_center=arr(minPos,2);
        
        % Find the edge points
        
        ii1 = edge(im1);
        if display
            figure(100*nn);subplot(1,5,1);imshow(ii1,[]);hold on
        end
        [xx,yy]=find(ii1==1);
        yi=median(yy);
        %xi=median(xx);
        xi=round(median(xx)); % xi is median value
        [~,xt]= min(abs(xx-xi)); % xt is the index to the closest point to medain
        
        if display
            plot(yy(xt),xx(xt),'y*','LineWidth',2)
            hold off
        end
        x_edge=yy(xt);
        y_edge=xx(xt);
        
        
        % create a matrix which will be used later in calculations
        center = repmat([y_center; x_center], 1, length(X));
        
        
        if (y_edge+BoundingBox(2))>arr(1,1) && arr(1,1)<y_center
            theta = (atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
            orig_angle=atand((y_center-arr(1,1))/(x_center-arr(1,2)));
            theta=(theta+orig_angle);
        elseif (y_edge+BoundingBox(2))<arr(1,1) && arr(1,1)>y_center
            theta = -(atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
            orig_angle=atand((y_center-arr(1,1))/(x_center-arr(1,2)));
            theta=(theta+orig_angle);
        elseif (y_edge+BoundingBox(2))>arr(1,1)
            theta = (atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
        elseif (y_edge+BoundingBox(2))<arr(1,1)
            theta = -(atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
        end
        R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
        
        % do the rotation...
        s = v - center; % shift points in the plane so that the center of rotation is at the origin
        so = R*s; % apply the rotation about the origin
        vo = so + center; % shift again so the origin goes back to the desired center of rotation
        % this can be done in one line as:
        % vo = R*(v - center) + center
        
        % pick out the vectors of rotated x- and y-data
        x_rotated = vo(1,:); % rotate all data
        y_rotated = vo(2,:);
        
        x_rotated(minPos:end)= v(1,minPos:end);% restore the roatated data to original values for the other four sections of image
        y_rotated(minPos:end)= v(2,minPos:end);
        
%         if display
%             figure;plot(y, x, 'k-', y_rotated, x_rotated, 'r-', y_center, x_center, 'bo');
%         end
%         
        % process fifth rectangle
        % choose a point on first block which will be the center of rotation
        
        
        if std2(x_rotated(1:minPos))>11
            x_rotated(1:minPos)= MidLine_Box(1:minPos,1);% restore the roatated data to original values for the other four sections of image
            y_rotated(1:minPos)= MidLine_Box(1:minPos,2);
        end
        
        
        v=[x_rotated' y_rotated']';
    end
    
    %% process fifth rectangle
    if abs(final_Or1(nn,4))<9
        
        tempArr=arr(round(4*size(MidLine_Box,1)/5):end,1)-MidLine_Box(round(4*size(MidLine_Box,1)/5):end,1);
        tempArr=abs(tempArr);
        [minVal,minPos]=min(tempArr);
        if size(tempArr,1)== minPos % if the position is in the edge move it back
            minPos = minPos -1;
        end
        y_center=arr(round(4*size(MidLine_Box,1)/5)+minPos,1);
        x_center=arr(round(4*size(MidLine_Box,1)/5)+minPos,2);
        
        
        ii2 = edge(im2);
        ii3 = edge(im3);
        ii4 = edge(im4);
        ii5 = edge(im5);
        if display
            figure(100*nn);subplot(1,5,2);imshow(ii2,[]);subplot(1,5,3);imshow(ii3,[]);subplot(1,5,4);imshow(ii4,[]);
        end
        clear xx yy yi xi
        if display
            subplot(1,5,5);imshow(ii5,[]);hold on
        end
        [xx,yy]=find(ii5==1);
        yi=median(yy);
        xi=median(xx);
        xi=round(median(xx));
        xt=max(find(xi==xx));
        if display
            plot(yy(xt),xx(xt),'y*','LineWidth',2)
            hold off
        end
        
        
        
        x_edge=yy(xt);
        y_edge=xx(xt);
        
        % create a matrix which will be used later in calculations
        center = repmat([y_center; x_center], 1, length(X));
        
        
        if (y_edge+BoundingBox(2))>arr(end,1) && arr(end,1)<y_center
            theta = (atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
            orig_angle=atand((y_center-arr(end,1))/(x_center-arr(end,2)));
            theta=(theta+orig_angle);
        elseif (y_edge+BoundingBox(2))<arr(end,1) && arr(end,1)>y_center
            theta = -(atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
            orig_angle=atand((y_center-arr(end,1))/(x_center-arr(end,2)));
            theta=(theta+orig_angle);
        elseif (y_edge+BoundingBox(2))>arr(end,1)
            theta = (atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
        elseif (y_edge+BoundingBox(2))<arr(end,1)
            theta = -(atand(abs((y_edge+BoundingBox(2)-y_center)/(x_edge+BoundingBox(1)-x_center)))); % pi/3 radians = 60 degrees
        end
        % do the rotation...
        R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
        
        s = v - center; % shift points in the plane so that the center of rotation is at the origin
        so = R*s; % apply the rotation about the origin
        vo = so + center; % shift again so the origin goes back to the desired center of rotation
        % this can be done in one line as:
        % vo = R*(v - center) + center
        
        % pick out the vectors of rotated x- and y-data
        x_rotated = vo(1,:);
        y_rotated = vo(2,:);
        
        x_rotated(1:round(4*size(MidLine_Box,1)/5)+minPos)= v(1,1:round(4*size(MidLine_Box,1)/5)+minPos);
        y_rotated(1:round(4*size(MidLine_Box,1)/5)+minPos)= v(2,1:round(4*size(MidLine_Box,1)/5)+minPos);
        
        if std2(x_rotated(round(4*size(MidLine_Box,1)/5)+minPos:end))>11
            x_rotated(round(4*size(MidLine_Box,1)/5)+minPos:end)= MidLine_Box(round(4*size(MidLine_Box,1)/5)+minPos:end,1);% restore the roatated data to original values for the other four sections of image
            y_rotated(round(4*size(MidLine_Box,1)/5)+minPos:end)= MidLine_Box(round(4*size(MidLine_Box,1)/5)+minPos:end,2);
        end
        
        
        arr_new(:,1)=x_rotated';
        arr_new(:,2)=y_rotated';
        
               
    else
        arr_new=v';
    end
    
    arr_new(:,1)=smooth(arr_new(:,1),150);
    if display
        figure(nn);
        hold on;plot(arr_new(:,2),arr_new(:,1),'b-','LineWidth',2);
        legend('Medial Axis');
    end

%%%%algorithm to interpolate between missing points 
Medial_Axis=floor(arr_new);
% Medial_Axis=arr_new;

%%%trying to interpolate so that there is a value at each x point
p_prev=Medial_Axis(1,2);
p=0;%%make it iterate in 2nd loop for 1:2 and interpolate
for ite=2:size(Medial_Axis)
    pp1= Medial_Axis(ite,2);
    diff=pp1-p_prev;
    for itdiff=1:diff
        p=p+1;
        comp=p_prev+1;
    if(pp1==comp)
        storex(p,1)=pp1;
        storey(p,1)=Medial_Axis(ite,1);
       
    else
        storex(p,1)=comp;
        storey(p,1)=Medial_Axis(ite,1);
       
    end
    p_prev=comp;
    end
   
end
    storex=[Medial_Axis(1,2);storex];
    storey=[Medial_Axis(1,1);storey];
    
    Medial_Axis=[storey storex];
    
    Medial_Axis(find(Medial_Axis(:,2)<=0),:)=[];

%     MedialAxis=arr_new;

matt=zeros(size(final_image));
for it=1:size(Medial_Axis,1)
    p1=Medial_Axis(it,1);
    p2=Medial_Axis(it,2);
    matt(p1,p2)=1;
end

%     figure,imshow(matt)
    [row col]=size(final_image);
    matt=matt(1:row,1:col);
    
    image_edge1=edge(final_image);
se = strel('disk',2);
II1 = imclose(image_edge1,se);
%[XXX,YYY]=find(ii1==1);
% figure,imshow(II1)

axis_pts=matt&II1;
% figure,imshow(axis_pts)

[k,l]=find(axis_pts==1);

if(numel(k)>1)

p11 = [k(1); l(1)];
p22 = [k(end); l(end)];
d = norm(p11 - p22);

if(d>200)
    
if(k(1)==k(end))
    
pos1=find(Medial_Axis(:,2)== l(1));
pos2=find(Medial_Axis(:,2)== l(end));


elseif(l(1)==l(end))
[pos1]=find(Medial_Axis(:,1)== k(1));
[pos2]=find(Medial_Axis(:,1)== k(end));

else
[pos1]=find(Medial_Axis(:,2)== l(1));
[pos2]=find(Medial_Axis(:,2)== l(end));

end    
% % (1)l2=l(2);
Medial_Axis=Medial_Axis(pos1:pos2,:);
% 
end
% 
% 
else
[pos1]=find(Medial_Axis(:,2)== l(1));
% % [point]=newMedialAxis(pos1,:);
% % pointx=point(2);
% % pointy=point(1);
% % [pointend]=newMedialAxis(end,:);
% % pointx=point(2);
% % pointy=point(1);
pend=Medial_Axis(end,:);
pstart=Medial_Axis(1,:);
p11 = [Medial_Axis(pos1,1); Medial_Axis(pos1,2)];
p22 = [Medial_Axis(end,1); Medial_Axis(end,2)];
ps=[Medial_Axis(pstart,1); Medial_Axis(pstart,2)];
d = norm(p11 - p22);
% daxis=norm(ps-p11);

if(d>350)

% 
Medial_Axis=Medial_Axis(pos1:end,:);
else
  Medial_Axis=Medial_Axis(1:pos1,:);  
end
end

    newMedialAxis=Medial_Axis;
    
    
    
    
    
   