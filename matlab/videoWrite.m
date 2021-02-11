    small_clip=886; 
    num=num2str(small_clip);
    
    vid_o = [num2str(small_clip),'_Origined','.avi'];
    vid_p = [num2str(small_clip),'.avi'];
    vid_d = [num2str(small_clip),'_Corrupted', '.avi'];
   
    v_o = VideoReader(vid_o);
    v_p = VideoReader(vid_p);
    v_d = VideoReader(vid_d);
    
    v_o = double(read(v_o));
    v_p = double(read(v_p));
    v_d = double(read(v_d));
           
    v_c = cat(2, v_o, v_d);
    v_all = cat(2, v_c, v_p);
    
    for i=1:25
        frames{i} = v_all(:,:,:,i)/255;
   
    end
    
    v = VideoWriter('2_sigmoid.avi');
    open(v)
    for i =1:25
        v.writeVideo(frames{i})
    end
    close(v)