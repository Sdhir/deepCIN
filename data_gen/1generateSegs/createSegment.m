function [final_box,seg_num] =createSegment(medial_coord,seg_width,nn,final_image,shift_val,num_method,factor_distance,display)
%save('medial_coord.mat', 'medial_coord');
edgeIm=edge(final_image);
if display
    plot(medial_coord(:,2),medial_coord(:,1),'Linewidth',2);
end
[Y,X]=find(edgeIm==1);
%plot(medial_coord,'Linewidth',2);
interv_ar = 1;
intersec1 = zeros(size(final_image));
for i=1:size(medial_coord,1)
    intersec1(medial_coord(i,1),medial_coord(i,2))=1;
end
intersec_x = [0;medial_coord(1,2)];intersec_y = [0;medial_coord(1,1)];
seg_num = 0;
x_cir = 1;
while medial_coord(end,2) > max(x_cir)
    [x_cir,y_cir] = circle(intersec_x(end,1),intersec_y(end,1),seg_width,display);
    intersec2 = zeros(size(final_image));
    % Limit circle at the edges
    x_cir(x_cir<=0) = 1;
    y_cir(y_cir<=0) = 1;
    x_cir(x_cir>=size(final_image,2)) = 1;
    y_cir(y_cir>=size(final_image,1)) = 1;
    for i=1:size(x_cir,2)
        intersec2(y_cir(1,i),x_cir(1,i))=1;
    end
    se = strel('line',2,90);
    intersec1_morph = imdilate(intersec1,se);
    intersec = intersec1_morph & intersec2;
    [intersec_y,intersec_x]=find(intersec==1);
    %disp(size(intersec_x,1))
    if seg_num == 0
        intersec_y = [0;intersec_y];
        intersec_x = [0;intersec_x];
    end
    idx=find(medial_coord(:,2)==intersec_x(end,1));
    interv_ar = [interv_ar,idx-1];
    seg_num = seg_num + 1;
end 
disp(['Getting ' num2str(seg_num-1) ' segments of image: ',num2str(nn)])
% num_blk = 44;
% fract_blk=1/num_blk;
% interv_ar=ceil(fract_blk*size(medial_coord,1):fract_blk*size(medial_coord,1):size(medial_coord,1));
% interv_ar=ceil(64:64:size(medial_coord,1));
% disp(interv_ar);
% final_interv_ar=zeros(1,size(interv_ar,2)+1);
% final_interv_ar(1)=1;
% final_interv_ar(2:end)=interv_ar;
% final_box=cell(1,seg_num);
% interv_ar=final_interv_ar;
final_box=cell(1,seg_num);
interv_ar = interv_ar(1,1:end-1);
segment_data=zeros(interv_ar(end),2);

for ar=1:numel(interv_ar)-1
    medial_segment=medial_coord(interv_ar(ar):interv_ar(ar+1),:);
%     disp(size(medial_segment,1))
    if num_method>1
        [coff] = polyfit(medial_segment(:,2),medial_segment(:,1),1);
        line_coord(:,2)=medial_coord(interv_ar(ar):interv_ar(ar+1),2); %x
        line_coord(:,1)=coff(1)*medial_segment(:,2)+coff(2); %y=mx+c
        % vector that has data of all the segments
        segment_data(interv_ar(ar):interv_ar(ar+1),2)=medial_coord(interv_ar(ar):interv_ar(ar+1),2);
        segment_data(interv_ar(ar):interv_ar(ar+1),1)=coff(1)*medial_segment(:,2)+coff(2);
        
    else
        
        line_coord(:,2)=medial_coord(interv_ar(ar):interv_ar(ar+1),2);
        line_coord(:,1)=medial_segment(:,1);
        segment_data(interv_ar(ar):interv_ar(ar+1),2)=medial_coord(interv_ar(ar):interv_ar(ar+1),2);
        segment_data(interv_ar(ar):interv_ar(ar+1),1)=medial_segment(:,1);
    
    end
%     disp(size(line_coord,1))
%%    
    slope=-(line_coord(end,1)-line_coord(1,1))/(line_coord(end,2)-line_coord(1,2));
    rotAngle=atand(slope);
    % display boxes
    if display
        plot(segment_data(interv_ar(ar):interv_ar(ar+1),2),segment_data(interv_ar(ar):interv_ar(ar+1),1),'Linewidth',2);
    end   
    
    seg_coord=[segment_data(interv_ar(ar):interv_ar(ar+1),2) segment_data(interv_ar(ar):interv_ar(ar+1),1)];
    %%%%adding extra
    
   medial_segments{ar} = [seg_coord];
    
    [~,posInfo]=(min(abs(X-floor(line_coord(floor(size(line_coord,1)/2),2)))));
    
    newposInfo=find(X==X(posInfo));
    posInfo=newposInfo;
    if numel(posInfo)<=1
        
        posInfo(2)=posInfo(1)-1;
        posInfo(1)=posInfo(1)+1;
    end
    
    mid_vert_line= [X(posInfo(1)) line_coord(floor(size(line_coord,1)/2),2) X(posInfo(end));Y(posInfo(1))-shift_val line_coord(floor(size(line_coord,1)/2),1) Y(posInfo(end))+shift_val];
    
    
    %     end_vert_line= [1 line_coord(end,2) 1;1 line_coord(end,1) 1]
    theta=-rotAngle;
    %% rotate middle line
    y_center=mid_vert_line(1,2);
    x_center=mid_vert_line(2,2);
    
    center = repmat([y_center; x_center], 1, length(mid_vert_line));
    
    R = [cosd(theta) -sind(theta); sind(theta) cosd(theta)];
    v=mid_vert_line;
    % do the rotation...
    s = v - center; % shift points in the plane so that the center of rotation is at the origin
    so = R*s; % apply the rotation about the origin
    vo = so + center; % shift again so the origin goes back to the desired center of rotation
    % this can be done in one line as:
    % vo = R*(v - center) + center
    
    % pick out the vectors of rotated x- and y-data
    mid_x_rotated = vo(1,:); % rotate all data
    mid_y_rotated = vo(2,:);
    
    % overlapping patches
    % A vector along the ray
    if factor_distance ~= 1
        V = seg_coord(end,:)-seg_coord(1,:); % approx. segment_width
        factor_distance = 0.5; % factor to change the widths
        pext = seg_coord(1,:) + V*factor_distance;
        seg_coord = [seg_coord;pext];
    end
    
    
    %%
    
    if theta>=0
        left_vert_line= [seg_coord(1,1)+abs(mid_x_rotated(1)-mid_x_rotated(2)) seg_coord(1,1) seg_coord(1,1)-abs(mid_x_rotated(2)-mid_x_rotated(3));...
            seg_coord(1,2)-abs(mid_y_rotated(1)-mid_y_rotated(2)) seg_coord(1,2) seg_coord(1,2)+abs(mid_y_rotated(2)-mid_y_rotated(3))];
        
        right_vert_line= [seg_coord(end,1)+abs(mid_x_rotated(1)-mid_x_rotated(2)) seg_coord(end,1) seg_coord(end,1)-abs(mid_x_rotated(2)-mid_x_rotated(3));...
            seg_coord(end,2)-abs(mid_y_rotated(1)-mid_y_rotated(2)) seg_coord(end,2) seg_coord(end,2)+abs(mid_y_rotated(2)-mid_y_rotated(3))];
    elseif theta<0
        left_vert_line= [seg_coord(1,1)-abs(mid_x_rotated(1)-mid_x_rotated(2)) seg_coord(1,1) seg_coord(1,1)+abs(mid_x_rotated(2)-mid_x_rotated(3));...
            seg_coord(1,2)-abs(mid_y_rotated(1)-mid_y_rotated(2)) seg_coord(1,2) seg_coord(1,2)+abs(mid_y_rotated(2)-mid_y_rotated(3))];
        
        right_vert_line= [seg_coord(end,1)-abs(mid_x_rotated(1)-mid_x_rotated(2)) seg_coord(end,1) seg_coord(end,1)+abs(mid_x_rotated(2)-mid_x_rotated(3));...
            seg_coord(end,2)-abs(mid_y_rotated(1)-mid_y_rotated(2)) seg_coord(end,2) seg_coord(end,2)+abs(mid_y_rotated(2)-mid_y_rotated(3))];
    end
    
    
    %% Plot the lines
    
    final_box_x=[left_vert_line(1,3) left_vert_line(1,1) right_vert_line(1,1) right_vert_line(1,3) left_vert_line(1,3)];
    final_box_y=[left_vert_line(2,3) left_vert_line(2,1) right_vert_line(2,1) right_vert_line(2,3) left_vert_line(2,3)];
    if display
        hold on
        plot(line_coord(floor(size(line_coord,1)/2),2),line_coord(floor(size(line_coord,1)/2),1),'y*','LineWidth',2);
        line(mid_x_rotated,mid_y_rotated,'Color','g','Linewidth',2)
        line(final_box_x,final_box_y,'Color','m','Linewidth',2)
    end
    final_box{ar}=[final_box_x;final_box_y];
    
    clear line_coord mid_vert_line left_vert_line right_vert_line
    
end