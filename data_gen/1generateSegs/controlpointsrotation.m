function [newMedialAxis]= controlpointsrotation(binaryMask,colorImg,nn,set_span)
grayImg=colorImg(:,:,2); % convert color image to gray image, take the blue plane as it showed best results

%% find bounding box for the epithleium area
[bbox]=regionprops(binaryMask,'BoundingBox','Area');

maxAr=0;
for kk=1:numel(bbox)
    maxAr_new=bbox(kk).Area;
    if maxAr_new>maxAr
        maxAr=maxAr_new;
        ind_bb=kk;
    end
end
BoundingBox=  cat(1, bbox(ind_bb).BoundingBox);
%rectangle('Position',[BoundingBox],'EdgeColor','r');
%% find the nuclei using the extended minima transform followed by structural element reconstruction of the nuclei
BW = imextendedmin((grayImg),75); % use extended minima transform, span is set to 75.
se90 = strel('disk', 1, 4); % use structural element to reconstruct the nuclei
se0 = strel('disk', 1, 4);
BWsdil = imdilate(BW, [se90 se0]);

%% label the nuclei
[L,num]=bwlabel(BWsdil);
L(L<2)=0;

%% find the bounding box points
p1=[bbox(ind_bb).BoundingBox(1) bbox(ind_bb).BoundingBox(2)];
p2=[bbox(ind_bb).BoundingBox(1) bbox(ind_bb).BoundingBox(2)+bbox(ind_bb).BoundingBox(4)];
p3=[bbox(ind_bb).BoundingBox(1)+bbox(ind_bb).BoundingBox(3) bbox(ind_bb).BoundingBox(2)+bbox(ind_bb).BoundingBox(4)];
p4=[bbox(ind_bb).BoundingBox(1)+bbox(ind_bb).BoundingBox(3) bbox(ind_bb).BoundingBox(2)];
% p5=p1
% p6=
% p7=
% p8=

%  line([p1(1) p3(1)],[p1(2) p3(2)])
% line([p2(1) p4(1)],[p2(2) p4(2)])


%% find the masks created by rotation. sixteen masks are created. 
% use the formula of y=mx+c for a straight line to find the coordinates of
% the separating lines. In addition create one third lines.
d1=(p1(2)-p3(2))/(p1(1)-p3(1));
s1_pts_x=p1(1):p3(1);
c1=p1(2)-d1*p1(1);
s1_pts_y=d1.*(s1_pts_x)+c1;
d2=(p4(2)-p2(2))/(p4(1)-p2(1));
s2_pts_x=p2(1):p4(1);
c2=p2(2)-d2*p2(1);
s2_pts_y=d2.*(s2_pts_x)+c2;
s3_pts_x=p1(1):p3(1);
s3_pts_y=(p2(2)+p1(2))/2*ones(size(s3_pts_x));
s4_pts_y=(p1(2):p2(2));
s4_pts_x=(p1(1)+p3(1))/2*ones(size(s4_pts_y));
maskBoundingBox=zeros(size(grayImg));
maskBoundingBox(p1(2):p2(2),p1(1):p4(1))=1;
maskDiag1=zeros(size(grayImg));

for kk=1:numel(s1_pts_x)
    maskDiag1(round(s1_pts_y(kk)),round(s1_pts_x(1):s1_pts_x(kk)))=1;
end



newImg=maskDiag1.*BWsdil;
mask1=and(newImg,L);
% figure;imshow(mask1)
mask2=xor(L,mask1);
% figure;imshow(mask2)

maskArea1=and(maskDiag1,binaryMask);
area=regionprops(maskArea1,'Area','Eccentricity','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr1=area(ind_ar).Area; ecc1=area(ind_ar).Eccentricity;
else
    
    maskAr1=area(1).Area;ecc1=area(1).Eccentricity;
end

maskDiag2=xor(maskBoundingBox,maskDiag1);
maskArea2=and(maskDiag2,binaryMask);
area=regionprops(maskArea2,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr2=area(ind_ar).Area; ecc2=area(ind_ar).Eccentricity;
else
    
    maskAr2=area(1).Area;ecc2=area(1).Eccentricity;
end


maskDiag3=zeros(size(grayImg));

for kk=1:numel(s2_pts_x)
    maskDiag3(round(s2_pts_y(kk)),round(s2_pts_x(1):s2_pts_x(kk)))=1;
end
newImg=maskDiag3.*BWsdil;
mask3=and(newImg,L);
% figure;imshow(mask3)
mask4=xor(L,mask3);
% figure;imshow(mask4)

maskArea3=and(maskDiag3,binaryMask);
area=regionprops(maskArea3,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr3=area(ind_ar).Area; ecc3=area(ind_ar).Eccentricity;
else
    
    maskAr3=area(1).Area;ecc3=area(1).Eccentricity;
end
maskDiag4=xor(maskBoundingBox,maskDiag3);
maskArea4=and(maskDiag4,binaryMask);
area=regionprops(maskArea4,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr4=area(ind_ar).Area; ecc4=area(ind_ar).Eccentricity;
else
    
    maskAr4=area(1).Area;ecc4=area(1).Eccentricity;
end



% now find the middle points of bounding box

maskDiag5=zeros(size(grayImg));

maskDiag5(p1(2):p2(2),p1(1):p1(1)+(p4(1)-p1(1))/2)=1;
newImg=maskDiag5.*BWsdil;
mask5=and(newImg,L);
% figure;imshow(mask5)
mask6=xor(L,mask5);
% figure;imshow(mask6)


maskArea5=and(maskDiag5,binaryMask);
area=regionprops(maskArea5,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr5=area(ind_ar).Area; ecc5=area(ind_ar).Eccentricity;
else
    
    maskAr5=area(1).Area;ecc5=area(1).Eccentricity;
end
maskDiag6=xor(maskBoundingBox,maskDiag5);
maskArea6=and(maskDiag6,binaryMask);
area=regionprops(maskArea6,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr6=area(ind_ar).Area; ecc6=area(ind_ar).Eccentricity;
else
    
    maskAr6=area(1).Area; ecc6=area(1).Eccentricity;
end




maskDiag7=zeros(size(grayImg));
maskDiag7(p1(2):p1(2)+(p2(2)-p1(2))/2,p1(1):p4(1))=1;
newImg=maskDiag7.*BWsdil;
mask7=and(newImg,L);
% figure;imshow(mask7)
mask8=xor(L,mask7);
% figure;imshow(mask8)


maskArea7=and(maskDiag7,binaryMask);
area=regionprops(maskArea7,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr7=area(ind_ar).Area; ecc7=area(ind_ar).Eccentricity;
else
    
    maskAr7=area(1).Area;ecc7=area(1).Eccentricity;
end
maskDiag8=xor(maskBoundingBox,maskDiag7);
maskArea8=and(maskDiag8,binaryMask);
area=regionprops(maskArea8,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr8=area(ind_ar).Area; ecc8=area(ind_ar).Eccentricity;
else
    
    maskAr8=area(1).Area;ecc8=area(1).Eccentricity;
end

%% Add new areas as per Dr.Stanley
p5=[p1(1)+(p3(1)-p1(1))/4 p1(2)];
p6=[p1(1)+3*(p3(1)-p1(1))/4 p1(2)];
p7=[p6(1) p3(2)];
p8=[p5(1) p2(2)];
p9=[p1(1) p1(2)+(p2(2)-p1(2))/4];
p10=[p1(1) p1(2)+3*(p2(2)-p1(2))/4];
p11=[p3(1) p4(2)+3*(p3(2)-p4(2))/4];
p12=[p3(1) p4(2)+(p3(2)-p4(2))/4];

xx=[p1(1) p5(1) p7(1) p8(1) p2(1) p10(1) p9(1)];
yy=[p1(2) p5(2) p7(2) p8(2) p2(2) p10(2) p9(2)];

maskDiag9=poly2mask(xx,yy,size(BW,1),size(BW,2));


newImg=maskDiag9.*BWsdil;
mask9=and(newImg,L);
mask10=xor(L,mask9);
maskArea9=and(maskDiag9,binaryMask);
area=regionprops(maskArea9,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr9=area(ind_ar).Area; ecc9=area(ind_ar).Eccentricity;
else
    
    maskAr9=area(1).Area;ecc9=area(1).Eccentricity;
end

maskDiag10=xor(maskBoundingBox,maskDiag9);
maskArea10=and(maskDiag10,binaryMask);
area=regionprops(maskArea10,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr10=area(ind_ar).Area; ecc10=area(ind_ar).Eccentricity;
else
    
    maskAr10=area(1).Area;ecc10=area(1).Eccentricity;
end



xx=[p6(1) p4(1) p12(1) p11(1) p3(1) p7(1) p8(1)];
yy=[p6(2) p4(2) p12(2) p11(2) p3(2) p7(2) p8(2)];

maskDiag11=poly2mask(xx,yy,size(BW,1),size(BW,2));


newImg=maskDiag11.*BWsdil;
mask11=and(newImg,L);
mask12=xor(L,mask11);
maskArea11=and(maskDiag11,binaryMask);
area=regionprops(maskArea11,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr11=area(ind_ar).Area; ecc11=area(ind_ar).Eccentricity;
else
    
    maskAr11=area(1).Area;ecc11=area(1).Eccentricity;
end

maskDiag12=xor(maskBoundingBox,maskDiag11);
maskArea12=and(maskDiag12,binaryMask);
area=regionprops(maskArea12,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr12=area(ind_ar).Area; ecc12=area(ind_ar).Eccentricity;
else
    
    maskAr12=area(1).Area;ecc12=area(1).Eccentricity;
end





xx=[p9(1) p11(1) p3(1) p7(1) p8(1) p2(1) p10(1)];
yy=[p9(2) p11(2) p3(2) p7(2) p8(2) p2(2) p10(2)];

maskDiag13=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag13.*BWsdil;
mask13=and(newImg,L);
mask14=xor(L,mask13);
maskArea13=and(maskDiag13,binaryMask);
area=regionprops(maskArea13,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr13=area(ind_ar).Area; ecc13=area(ind_ar).Eccentricity;
else
    
    maskAr13=area(1).Area;ecc13=area(1).Eccentricity;
end

maskDiag14=xor(maskBoundingBox,maskDiag13);
maskArea14=and(maskDiag14,binaryMask);
area=regionprops(maskArea14,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr14=area(ind_ar).Area; ecc14=area(ind_ar).Eccentricity;
else
    
    maskAr14=area(1).Area;ecc14=area(1).Eccentricity;
end






xx=[p10(1) p12(1) p11(1) p3(1) p7(1) p8(1) p2(1)];
yy=[p10(2) p12(2) p11(2) p3(2) p7(2) p8(2) p2(2)];

maskDiag15=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag15.*BWsdil;
mask15=and(newImg,L);
mask16=xor(L,mask15);
maskArea15=and(maskDiag15,binaryMask);
area=regionprops(maskArea15,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr15=area(ind_ar).Area; ecc15=area(ind_ar).Eccentricity;
else
    
    maskAr15=area(1).Area;ecc15=area(1).Eccentricity;
end

maskDiag16=xor(maskBoundingBox,maskDiag15);
maskArea16=and(maskDiag16,binaryMask);
area=regionprops(maskArea16,'Area','Eccentricity');
if numel(area)>1
    maxAr=0;
    for kk=1:numel(area)
        maxAr_new=area(kk).Area;
        if maxAr_new>maxAr
            maxAr=maxAr_new;
            ind_ar=kk;
        end
    end
    maskAr16=area(ind_ar).Area; ecc16=area(ind_ar).Eccentricity;
else
    
    maskAr16=area(1).Area;ecc16=area(1).Eccentricity;
end




%% compute number of nuclei in each segmented mask

[L1,n1]=bwlabel(mask1);
maskAr1= bwarea(mask1);
[L2,n2]=bwlabel(mask2);
maskAr2= bwarea(mask2);
[L3,n3]=bwlabel(mask3);
maskAr3= bwarea(mask3);
[L4,n4]=bwlabel(mask4);
maskAr4= bwarea(mask4);
[L5,n5]=bwlabel(mask5);
maskAr5= bwarea(mask5);
[L6,n6]=bwlabel(mask6);
maskAr6= bwarea(mask6);
[L7,n7]=bwlabel(mask7);
maskAr7= bwarea(mask7);
[L8,n8]=bwlabel(mask8);
maskAr8= bwarea(mask8);
[L9,n9]=bwlabel(mask9);
maskAr9= bwarea(mask9);
[L10,n10]=bwlabel(mask10);
maskAr10= bwarea(mask10);
[L11,n11]=bwlabel(mask11);
maskAr11= bwarea(mask11);

[L12,n12]=bwlabel(mask12);
maskAr12= bwarea(mask12);

[L13,n13]=bwlabel(mask13);
maskAr13= bwarea(mask13);

[L14,n14]=bwlabel(mask14);
maskAr14= bwarea(mask14);

[L15,n15]=bwlabel(mask15);
maskAr15= bwarea(mask15);

[L16,n16]=bwlabel(mask16);
maskAr16= bwarea(mask16);


%% compute ratio of nuclei. different ratios are computes: 1) ratio of number of nuclei, 2) area, 3)eccentricity of mask

f1=n1/n2;
f2=n2/n1;
f3=n3/n4;
f4=n4/n3;
f5=n5/n6;
f6=n6/n5;
f7=n7/n8;
f8=n8/n7;
f9=n9/n10;
f10=n10/n9;
f11=n11/n12;
f12=n12/n11;
f13=n13/n14;
f14=n14/n13;
f15=n15/n16;
f16=n16/n15;

f=[f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16];

maskAr=[maskAr1 maskAr2 maskAr3 maskAr4 maskAr5 maskAr6 maskAr7 maskAr8 maskAr9 maskAr10 maskAr11 maskAr12 maskAr13 maskAr14 maskAr15 maskAr16];

n=[n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15 16];
e=[ecc1 ecc2 ecc3 ecc4 ecc5 ecc6 ecc7 ecc8 ecc9 ecc10 ecc11 ecc12 ecc13 ecc14 ecc15 ecc16];

% the following expression was obtained empirically
 
%  v=n./(maskAr/10000);
 v=((n./f).*e);
[v1,v2]=max(v);



d5=(p5(2)-p7(2))/(p5(1)-p7(1));
s5_pts_x=p5(1):p7(1);
c5=p5(2)-d5*p5(1);
s5_pts_y=d5.*(s5_pts_x)+c5;
d6=(p6(2)-p8(2))/(p6(1)-p8(1));
s6_pts_x=p8(1):p6(1);
c6=p6(2)-d6*p6(1);
s6_pts_y=d6.*(s6_pts_x)+c6;


d7=(p9(2)-p11(2))/(p9(1)-p11(1));
s7_pts_x=p9(1):p11(1);
c7=p9(2)-d7*p9(1);
s7_pts_y=d7.*(s7_pts_x)+c7;
d8=(p12(2)-p10(2))/(p12(1)-p10(1));
s8_pts_x=p10(1):p12(1);
c8=p12(2)-d8*p12(1);
s8_pts_y=d8.*(s8_pts_x)+c8;

%%control points initialization

p14=[p4(1) (p4(2)+p3(2))/2];
p13=[(p2(1)+p3(1))/2 p2(2)];
p16=[p1(1) (p1(2)+p2(2))/2];
p15=[(p1(1)+p4(1))/2 p1(2)];

%%points to see if the symmetry check is fine

p17=[(p8(1)+ p13(1))/2 p13(2)];
p18=[(p7(1)+ p13(1))/2 p13(2)];

p19=[p14(1) (p11(2)+p14(2))/2];
p20=[p14(1) (p12(2)+p14(2))/2];

p21=[(p15(1)+p6(1))/2 p15(2)];
p22=[(p15(1)+p5(1))/2 p15(2)];

p23=[p16(1) (p9(2)+p16(2))/2];
p24=[p16(1) (p10(2)+p16(2))/2];

d9=(p23(2)-p19(2))/(p23(1)-p19(1));
s9_pts_x=p23(1):p19(1);
c9=p23(2)-d9*p23(1);                           %%control lines along the X axis
s9_pts_y=d9.*(s9_pts_x)+c9;
d10=(p24(2)-p20(2))/(p24(1)-p20(1));
s10_pts_x=p20(1):p24(1);
c10=p24(2)-d10*p24(1);
s10_pts_y=d10.*(s10_pts_x)+c10;


d11=(p22(2)-p18(2))/(p22(1)-p18(1));
s11_pts_x=p22(1):p18(1);
c11=p22(2)-d11*p22(1);
s11_pts_y=d11.*(s11_pts_x)+c11;        %% control points along the Yaxis
d12=(p17(2)-p21(2))/(p17(1)-p21(1));
s12_pts_x=p21(1):p17(1);
c12=p17(2)-d12*p17(1);
s12_pts_y=d12.*(s12_pts_x)+c12;


%%%creation of the control masks 8 of them

xx=[p22(1) p5(1) p1(1) p9(1) p10(1) p2(1) p8(1) p18(1)];
yy=[p22(2) p5(2) p1(2) p9(2) p10(2) p2(2) p8(2) p18(2)];

maskDiag17=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag17.*BWsdil;
mask17=and(newImg,L);
mask18=xor(L,mask17);           %%new mask 17 and 18
maskArea17=and(maskDiag17,binaryMask);


maskDiag18=xor(maskBoundingBox,maskDiag17);
maskArea18=and(maskDiag18,binaryMask);

xx=[p21(1) p15(1) p5(1) p1(1) p9(1) p10(1) p2(1) p8(1) p17(1)];
yy=[p21(2) p15(2) p5(2) p1(2) p9(2) p10(2) p2(2) p8(2) p17(2)];

maskDiag19=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag19.*BWsdil;
mask19=and(newImg,L);
mask20=xor(L,mask17);                   %%new mask 19 and 20
maskArea19=and(maskDiag19,binaryMask);


maskDiag20=xor(maskBoundingBox,maskDiag19);
maskArea20=and(maskDiag20,binaryMask);


xx=[p23(1) p9(1) p1(1) p5(1) p6(1) p4(1) p12(1) p19(1)];
yy=[p23(2) p9(2) p1(2) p5(2) p6(2) p4(2) p12(2) p19(2)];

maskDiag23=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag23.*BWsdil;
mask23=and(newImg,L);
mask24=xor(L,mask17);                   %%new mask 23 and 24
maskArea23=and(maskDiag23,binaryMask);


maskDiag24=xor(maskBoundingBox,maskDiag23);
maskArea24=and(maskDiag24,binaryMask);

xx=[p24(1) p9(1) p1(1) p5(1) p6(1) p4(1) p12(1) p20(1)];
yy=[p24(2) p9(2) p1(2) p5(2) p6(2) p4(2) p12(2) p20(2)];

maskDiag22=poly2mask(xx,yy,size(BW,1),size(BW,2));

newImg=maskDiag22.*BWsdil;
mask22=and(newImg,L);
mask21=xor(L,mask17);                   %%new mask 21 and 22
maskArea22=and(maskDiag22,binaryMask);


maskDiag21=xor(maskBoundingBox,maskDiag22);
maskArea21=and(maskDiag21,binaryMask);

[L17,n17]=bwlabel(mask17);
maskAr17= bwarea(mask17);
[L18,n18]=bwlabel(mask18);
maskAr18= bwarea(mask18);
[L19,n19]=bwlabel(mask19);
maskAr19= bwarea(mask19);
[L20,n20]=bwlabel(mask20);
maskAr20= bwarea(mask20);
[L21,n21]=bwlabel(mask21);
maskAr21= bwarea(mask21);
[L22,n22]=bwlabel(mask22);
maskAr22= bwarea(mask22);
[L23,n23]=bwlabel(mask23);
maskAr23= bwarea(mask23);
[L24,n24]=bwlabel(mask24);
maskAr24= bwarea(mask24);

f17=n17/n18;
f18=n18/n17;
f19=n19/n20;
f20=n20/n19;
f21=n21/n22;
f22=n22/n21;
f23=n23/n24;
f24=n24/n23;

% close all

image_edge=edge(binaryMask);
se = strel('disk',2);
ii1 = imclose(image_edge,se);
[XX,YY]=find(ii1==1);
% figure,imshow(ii1)
%figure(nn)
if v2==1 || v2==2
    %     subplot(3,1,2),imshow(mask1),subplot(3,1,3),imshow(mask2) 
    %line([s1_pts_x(1),s1_pts_x(end)],[s1_pts_y(1),s1_pts_y(end)])
    newMedialAxis=[s1_pts_y' s1_pts_x'];
elseif v2==3 || v2==4
    %     subplot(3,1,2),imshow(mask3),subplot(3,1,3),imshow(mask4)
    %line([s2_pts_x(1),s2_pts_x(end)],[s2_pts_y(1),s2_pts_y(end)])
    newMedialAxis=[s2_pts_y' s2_pts_x'];
elseif v2==5 || v2==6
    %     subplot(3,1,2),imshow(mask5),subplot(3,1,3),imshow(mask6)
    %line([s4_pts_x(1),s4_pts_x(end)],[s4_pts_y(1),s4_pts_y(end)])
    newMedialAxis=[s4_pts_y' s4_pts_x'];
elseif v2==7 || v2==8
    %line([s3_pts_x(1),s3_pts_x(end)],[s3_pts_y(1),s3_pts_y(end)])
    %     subplot(3,1,2),imshow(mask7),subplot(3,1,3),imshow(mask8)
    newMedialAxis=[s3_pts_y' s3_pts_x'];
   
    %%%finding the edge of the image
    
    
elseif v2==9 || v2==10
    symask109=imrotate(maskArea10,180);
    symarea109=xor(symask109,maskArea9);
    rot109=regionprops(symarea109,'Area');
     maxAr=0;
    for kk=1:numel(rot109)
        maxAr_new=rot109(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap109=maxAr;
    r109=anotoverlap109./maskAr9;
    comp1=r109.*f9;
    comp2=r109.*f10;
    
    
    symask17=imrotate(maskArea17,180);
    symarea2=xor(symask17,maskArea18);
    rot2=regionprops(symarea2,'Area');
     maxAr=0;
    for kk=1:numel(rot2)
        maxAr_new=rot2(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap2=maxAr;
    r2=anotoverlap2./maskAr18;
    comp3=r2.*f17;
    comp4=r2.*f18;
    comp=[comp1 comp2 comp3 comp4];
    [val po]=max(comp);
    if po==3 || po==4
        %line([s11_pts_x(1),s11_pts_x(end)],[s11_pts_y(1),s11_pts_y(end)])
        newMedialAxis=[s11_pts_y' s11_pts_x'];
     
    else
     %line([s5_pts_x(1),s5_pts_x(end)],[s5_pts_y(1),s5_pts_y(end)])
    newMedialAxis=[s5_pts_y' s5_pts_x'];
    end
    
    
elseif v2==11 || v2==12
    symask1112=imrotate(maskArea11,180);
    symarea1112=xor(symask1112,maskArea12);
    rot1112=regionprops(symarea1112,'Area');
     maxAr=0;
    for kk=1:numel(rot1112)
        maxAr_new=rot1112(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap1112=maxAr;
    r109=anotoverlap1112./maskAr12;
    comp1=r109.*f11;
    comp2=r109.*f12;
    
    symask20=imrotate(maskArea20,180);
    symarea20=xor(symask20,maskArea19);
    rot20=regionprops(symarea20,'Area');
     maxAr=0;
    for kk=1:numel(rot20)
        maxAr_new=rot20(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap20=maxAr;
    r2=anotoverlap20./maskAr19;
    comp3=r2.*f20;
    comp4=r2.*f19;
    comp=[comp1 comp2 comp3 comp4];
    [val pos]=max(comp);
    if pos==3 || pos==4
        %line([s12_pts_x(1),s12_pts_x(end)],[s12_pts_y(1),s12_pts_y(end)])
        newMedialAxis=[s12_pts_y' s12_pts_x']; 
     
    else
    %line([s6_pts_x(1),s6_pts_x(end)],[s6_pts_y(1),s6_pts_y(end)])
    newMedialAxis=[s6_pts_y' s6_pts_x'];
    end
    
    
elseif v2==13 || v2==14
    symask14=imrotate(maskArea14,180);
    symarea1413=xor(symask14,maskArea13);
    rot1413=regionprops(symarea1413,'Area');
     maxAr=0;
    for kk=1:numel(rot1413)
        maxAr_new=rot1413(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap1413=maxAr;
    r1413=anotoverlap1413./maskAr13;
    comp1=r1413.*f13;
    comp2=r1413.*f14;
    
    symask23=imrotate(maskArea24,180);
    symarea2=xor(symask23,maskArea23);
    rot2=regionprops(symarea2,'Area');
     maxAr=0;
    for kk=1:numel(rot2)
        maxAr_new=rot2(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap23=maxAr;
    r2=anotoverlap23./maskAr23;
    comp3=r2.*f23;
    comp4=r2.*f24;
    comp=[comp1 comp2 comp3 comp4];
    [val pos]=max(comp);
    if pos==3 || pos==4
        %line([s9_pts_x(1),s9_pts_x(end)],[s9_pts_y(1),s9_pts_y(end)])
newMedialAxis=[s9_pts_y' s9_pts_x'];
    else
         %line([s7_pts_x(1),s7_pts_x(end)],[s7_pts_y(1),s7_pts_y(end)])
    newMedialAxis=[s7_pts_y' s7_pts_x'];
    end
    
    
    
    elseif v2==15 || v2==16
        symask15=imrotate(maskArea15,180);
    symarea1516=xor(symask15,maskArea16);
    rot1516=regionprops(symarea1516,'Area');
     maxAr=0;
    for kk=1:numel(rot1516)
        maxAr_new=rot1516(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap1516=maxAr;
    r1516=anotoverlap1516./maskAr16;
    comp1=r1516.*f15;
    comp2=r1516.*f16;
    
    symask21=imrotate(maskArea21,180);
    symarea21=xor(symask21,maskArea22);
    rot21=regionprops(symarea21,'Area');
     maxAr=0;
    for kk=1:numel(rot21)
        maxAr_new=rot21(kk).Area;
         maxAr=maxAr+maxAr_new;
    end
    anotoverlap21=maxAr;
    r2=anotoverlap21./maskAr22;
    comp3=r2.*f21;
    comp4=r2.*f22;
    comp=[comp1 comp2 comp3 comp4];
    [val pos]=max(comp);
    if pos==3 || pos==4
        %line([s10_pts_x(1),s10_pts_x(end)],[s10_pts_y(1),s10_pts_y(end)])
newMedialAxis=[s10_pts_y' s10_pts_x'];
    
    else
    %line([s8_pts_x(1),s8_pts_x(end)],[s8_pts_y(1),s8_pts_y(end)])
    %     subplot(3,1,2),imshow(mask7),subplot(3,1,3),imshow(mask8)
    newMedialAxis=[s8_pts_y' s8_pts_x'];
    end
end
newMedialAxis=floor(newMedialAxis);

%  stl=strel('disk',10);
%  newMedialAxis=imdilate(newMedialAxis,stl);
matt=zeros(size(binaryMask));
for it=1:size(newMedialAxis,1)
    p1=round(newMedialAxis(it,1));
    p2=round(newMedialAxis(it,2));
    matt(p1,p2)=1;
end

%     figure,imshow(matt)

axis_pts=matt&ii1;
% figure,imshow(axis_pts)

[k,l]=find(axis_pts==1);

if(numel(k)>1)

p11 = [k(1); l(1)];
p22 = [k(end); l(end)];
d = norm(p11 - p22);

if(d>100)
    
if(k(1)==k(end))
    
pos1=find(newMedialAxis(:,2)== l(1));
pos2=find(newMedialAxis(:,2)== l(end));


elseif(l(1)==l(end))
[pos1]=find(newMedialAxis(:,1)== k(1));
[pos2]=find(newMedialAxis(:,1)== k(end));

else
[pos1]=find(newMedialAxis(:,2)== l(1));
[pos2]=find(newMedialAxis(:,2)== l(end));

end    
% % (1)l2=l(2);
newMedialAxis=newMedialAxis(pos1:pos2,:);
% 
end
% 
% 
else
[pos1]=find(newMedialAxis(:,2)== l(1));
% % [point]=newMedialAxis(pos1,:);
% % pointx=point(2);
% % pointy=point(1);
% % [pointend]=newMedialAxis(end,:);
% % pointx=point(2);
% % pointy=point(1);
% 
newMedialAxis=newMedialAxis(pos1:end,:);
end
% [newMedialAxis]=newMedialAxis;

% saveppt('Histology_New_MedialAXIS.ppt',[imgfn(1:end-4)])




