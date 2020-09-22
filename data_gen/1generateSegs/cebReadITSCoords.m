
 function [x,y]=cebReadITSCoords(dir,fn)
 %
 % Very simple function to read one set of (x,y) coords output
 % from IT-SNAPS to an .xml file.
 % The whole file is just one line.  All we do below is
 % read the <x>n...n</x> and <y>n..n</y> pairs.
 
    pathfn = [dir filesep fn];
 
    fh = fopen(pathfn);

    line =fgetl(fh); % This is old version not working using fopen instead
    %line =fopen(fh);
     
             
    [matchstart,matchend,tokenindices,xstr]=regexp(line,'<x>.*?</x>');
    [matchstart,matchend,tokenindices,ystr]=regexp(line,'<y>.*?</y>');
                        
             
     x = readStrings(xstr);
     y = readStrings(ystr);
 
    
 end

%=================================================================
function w=readStrings(wstr)
%
   m=size(wstr,2);
            
   for k=1:m
                
      str=wstr{k};
      n=size(str,2);
      
      ibeg=4;
      iend=n-4;
      strdat=str(ibeg:iend);
      w(k)=str2num(strdat);
                
    end
end
 




      
       