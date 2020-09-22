function [x_cir,y_cir] =  circle(x,y,r,display)
%x and y are the coordinates of the center of the circle
%r is the radius of the circle
%0.01 is the angle step, bigger values will draw the circle faster but
%you might notice imperfections (not very smooth)
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
x_cir = round(x+xp);
y_cir = round(y+yp);
if display
    plot(x_cir,y_cir,'LineWidth',2);
end
end