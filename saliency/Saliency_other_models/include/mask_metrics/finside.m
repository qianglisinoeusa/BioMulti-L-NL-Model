function [ total_score, score_finside ] = fix_inside( scanpath, mask, ff_flag )
    if nargin<3, ff_flag=1; end

    total_score=0;
    score_finside=0;
    if ~iscell(scanpath)
        if ff_flag==1 && size(scanpath,1)>1
          scanpath(1,:)=[];
        end
        for i=1:size(scanpath,1)
           x=scanpath(i,1); 
           y=scanpath(i,2); 
           if mask(y,x)>0
               score_finside(i)=1;
           end
        end
    else
        for c=1:length(scanpath)
            if ff_flag==1 && size(scanpath{c},1)>1
              scanpath{c}(1,:)=[];
            end
            for i=1:size(scanpath{c},1)
               x=scanpath{c}(i,1); 
               y=scanpath{c}(i,2); 
               if mask(y,x)>0
                   score_pp(c,i)=1;
               end
            end
        end
        score_finside=nanmean(score_pp,1);
    end
    total_score=nanmean(score_finside(:));
    
    
end

