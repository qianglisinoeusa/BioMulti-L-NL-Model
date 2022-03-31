x=1:5;
y=1:5;

[X,Y]=meshgrid(x,y);

jobidx = getenv('SLURM_ARRAY_TASK_ID')

XID = X(jobidx)
disp(XID)
YID = Y(jobidx)
disp(YID)     
