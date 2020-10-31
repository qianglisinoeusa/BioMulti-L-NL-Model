

function [metrics] = get_metrics_properties( n_trials1,  indexes_other_trials)

            metrics = cell(1,29);

            metrics{1}.name = 'AUC_Judd';
            metrics{1}.function = 'AUC_Judd';
            metrics{1}.type = 'metric_nonshuffled';
            metrics{1}.baseline_type = '';
            metrics{1}.comparison_type = 'smap-bmap';
            metrics{1}.scoring = 1;
            metrics{1}.indexes_other = [];
            metrics{1}.trials = [];
            
            metrics{2}.name = 'AUC_Borji';
            metrics{2}.function = 'AUC_Borji';
            metrics{2}.type = 'metric_nonshuffled';
            metrics{2}.baseline_type = '';
            metrics{2}.comparison_type = 'smap-bmap';
            metrics{2}.scoring = 1;
            metrics{2}.indexes_other = [];
            metrics{2}.trials = [];
            
            metrics{3}.name = 'CC';
            metrics{3}.function = 'CC';
            metrics{3}.type = 'metric_nonshuffled';
            metrics{3}.baseline_type = '';
            metrics{3}.comparison_type = 'smap-dmap';
            metrics{3}.scoring = 1;
            metrics{3}.indexes_other = [];
            metrics{3}.trials = [];
            
            metrics{4}.name = 'NSS';
            metrics{4}.function = 'NSS';
            metrics{4}.type = 'metric_nonshuffled';
            metrics{4}.baseline_type = '';
            metrics{4}.comparison_type = 'smap-bmap';
            metrics{4}.scoring = 1;
            metrics{4}.indexes_other = [];
            metrics{4}.trials = [];
            
            metrics{5}.name = 'EMD';
            metrics{5}.function = 'EMD';
            metrics{5}.type = 'metric_nonshuffled';
            metrics{5}.baseline_type = '';
            metrics{5}.comparison_type = 'smap-dmap';
            metrics{5}.scoring = -1;
            metrics{5}.indexes_other = [];
            metrics{5}.trials = [];
            
            metrics{6}.name = 'KL';
            metrics{6}.function = 'KLdiv';
            metrics{6}.type = 'metric_nonshuffled';
            metrics{6}.baseline_type = '';
            metrics{6}.comparison_type = 'smap-dmap';
            metrics{6}.scoring = -1;
            metrics{6}.indexes_other = [];
            metrics{6}.trials = [];
            
            metrics{7}.name = 'SIM';
            metrics{7}.function = 'similarity';
            metrics{7}.type = 'metric_nonshuffled';
            metrics{7}.baseline_type = '';
            metrics{7}.comparison_type = 'smap-dmap';
            metrics{7}.scoring = 1;
            metrics{7}.indexes_other = [];
            metrics{7}.trials = [];
            
            metrics{8}.name = 'sAUC';
            metrics{8}.function = 'AUC_shuffled';
            metrics{8}.type = 'metric_shuffled_type1';
            metrics{8}.baseline_type = 'max';
            metrics{8}.comparison_type = 'smap-bmap-sbmap';
            metrics{8}.scoring = 1;
            metrics{8}.indexes_other = indexes_other_trials; %permutations of other trials
            metrics{8}.trials = n_trials1; %baseline compute all trials (limit of 100)
            
            
            metrics{9}.name = 'finside';
            metrics{9}.function = 'finside';
            metrics{9}.type = 'metric_nonshuffled';
            metrics{9}.baseline_type = '';
            metrics{9}.comparison_type = 'scanpath-mmap';
            metrics{9}.scoring = 1;
            metrics{9}.indexes_other = [];
            metrics{9}.trials = [];
            
            metrics{10}.name = 'sAUC_trials';
            metrics{10}.function = 'AUC_shuffled';
            metrics{10}.type = 'metric_shuffled_type6';
            metrics{10}.baseline_type = 'max';
            metrics{10}.comparison_type = 'smap-bmap-sbmap';
            metrics{10}.scoring = 1;
            metrics{10}.indexes_other = indexes_other_trials; %permutations of other trials
            if n_trials1 < 10
                metrics{10}.trials(1) = n_trials1; %baseline compute 10 trials
            else
                 metrics{10}.trials(1) = 10;
            end
            metrics{10}.trials(2)=1; %compute trials (10 other per image)= neval*10
            
            metrics{11}.name = 'IG_baseline';
            metrics{11}.function = 'IG';
            metrics{11}.type = 'metric_shuffled_type5';
            metrics{11}.baseline_type = 'max';
            metrics{11}.comparison_type = 'smap-bmap-baseline';
            metrics{11}.scoring = 1;
            metrics{11}.indexes_other = indexes_other_trials; %other images from presaved baseline
            metrics{11}.trials = n_trials1;  %compute all trials
            
            
             metrics{12}.name = 'SIndex';
             metrics{12}.function = 'SIndex';
             metrics{12}.type = 'metric_nonshuffled';
             metrics{12}.baseline_type = '';
             metrics{12}.comparison_type = 'smap-mmap';
             metrics{12}.scoring = 1;
             metrics{12}.indexes_other = [];
             metrics{12}.trials = [];
            
            metrics{13}.name = 'SaccadeLanding';
            metrics{13}.function = 'pp_slanding';
            metrics{13}.type = 'metric_nonshuffled';
            metrics{13}.baseline_type = '';
            metrics{13}.comparison_type = 'scanpath-pp_scanpath';
            metrics{13}.scoring = -1;
            metrics{13}.indexes_other = [];
            metrics{13}.trials = [];
            
            metrics{14}.name = 'SaccadeAmplitude';
            metrics{14}.function = 'samplitude';
            metrics{14}.type = 'metric_nonshuffled';
            metrics{14}.baseline_type = '';
            metrics{14}.comparison_type = 'scanpath_single';
            metrics{14}.scoring = -1;
            metrics{14}.indexes_other = [];
            metrics{14}.trials = [];

            metrics{15}.name = 'SaccadeLanding_accum';
            metrics{15}.function = 'pp_accumsl';
            metrics{15}.type = 'metric_nonshuffled';
            metrics{15}.baseline_type = '';
            metrics{15}.comparison_type = 'scanpath-pp_scanpath';
            metrics{15}.scoring = -1;
            metrics{15}.indexes_other = [];
            metrics{15}.trials = [];
            
            metrics{16}.name = 'SaccadeAmplitude_accum';
            metrics{16}.function = 'pp_accumsa';
            metrics{16}.type = 'metric_nonshuffled';
            metrics{16}.baseline_type = '';
            metrics{16}.comparison_type = 'scanpath-pp_scanpath';
            metrics{16}.scoring = -1;
            metrics{16}.indexes_other = [];
            metrics{16}.trials = [];
            
            
            metrics{17}.name = 'SaccadeAmplitudeDiff';
            metrics{17}.function = 'pp_samplitude_diff';
            metrics{17}.type = 'metric_nonshuffled';
            metrics{17}.baseline_type = '';
            metrics{17}.comparison_type = 'scanpath-pp_scanpath';
            metrics{17}.scoring = -1;
            metrics{17}.indexes_other = [];
            metrics{17}.trials = [];


            metrics{18}.name = 'SaccadeAmplitudeDiff2';
            metrics{18}.function = 'pp_samplitude_diff2';
            metrics{18}.type = 'metric_nonshuffled';
            metrics{18}.baseline_type = '';
            metrics{18}.comparison_type = 'scanpath-pp_scanpath';
            metrics{18}.scoring = -1;
            metrics{18}.indexes_other = [];
            metrics{18}.trials = [];
            
            
            metrics{19}.name = 'SaccadeLanding_01';
            metrics{19}.function = 'pp_slanding_01';
            metrics{19}.type = 'metric_nonshuffled';
            metrics{19}.baseline_type = '';
            metrics{19}.comparison_type = 'scanpath-pp_scanpath';
            metrics{19}.scoring = -1;
            metrics{19}.indexes_other = [];
            metrics{19}.trials = [];
            
            metrics{20}.name = 'SaccadeLanding_10';
            metrics{20}.function = 'pp_slanding_10';
            metrics{20}.type = 'metric_nonshuffled';
            metrics{20}.baseline_type = '';
            metrics{20}.comparison_type = 'scanpath-pp_scanpath';
            metrics{20}.scoring = -1;
            metrics{20}.indexes_other = [];
            metrics{20}.trials = [];
            
            metrics{21}.name = 'SaccadeLanding_11';
            metrics{21}.function = 'pp_slanding_11';
            metrics{21}.type = 'metric_nonshuffled';
            metrics{21}.baseline_type = '';
            metrics{21}.comparison_type = 'scanpath-pp_scanpath';
            metrics{21}.scoring = -1;
            metrics{21}.indexes_other = [];
            metrics{21}.trials = [];
            
            metrics{22}.name = 'SaccadeAmplitudeDiff_01';
            metrics{22}.function = 'pp_samplitude_diff_01';
            metrics{22}.type = 'metric_nonshuffled';
            metrics{22}.baseline_type = '';
            metrics{22}.comparison_type = 'scanpath-pp_scanpath';
            metrics{22}.scoring = -1;
            metrics{22}.indexes_other = [];
            metrics{22}.trials = [];
            
            metrics{23}.name = 'SaccadeAmplitudeDiff_10';
            metrics{23}.function = 'pp_samplitude_diff_10';
            metrics{23}.type = 'metric_nonshuffled';
            metrics{23}.baseline_type = '';
            metrics{23}.comparison_type = 'scanpath-pp_scanpath';
            metrics{23}.scoring = -1;
            metrics{23}.indexes_other = [];
            metrics{23}.trials = [];
            
            metrics{24}.name = 'SaccadeAmplitudeDiff_11';
            metrics{24}.function = 'pp_samplitude_diff_11';
            metrics{24}.type = 'metric_nonshuffled';
            metrics{24}.baseline_type = '';
            metrics{24}.comparison_type = 'scanpath-pp_scanpath';
            metrics{24}.scoring = -1;
            metrics{24}.indexes_other = [];
            metrics{24}.trials = [];
            
            metrics{25}.name = 'SaccadeAmplitudeDiff2_01';
            metrics{25}.function = 'pp_samplitude_diff2_01';
            metrics{25}.type = 'metric_nonshuffled';
            metrics{25}.baseline_type = '';
            metrics{25}.comparison_type = 'scanpath-pp_scanpath';
            metrics{25}.scoring = -1;
            metrics{25}.indexes_other = [];
            metrics{25}.trials = [];
            
            metrics{26}.name = 'SaccadeAmplitudeDiff2_10';
            metrics{26}.function = 'pp_samplitude_diff2_10';
            metrics{26}.type = 'metric_nonshuffled';
            metrics{26}.baseline_type = '';
            metrics{26}.comparison_type = 'scanpath-pp_scanpath';
            metrics{26}.scoring = -1;
            metrics{26}.indexes_other = [];
            metrics{26}.trials = [];
            
            metrics{27}.name = 'SaccadeAmplitudeDiff2_11';
            metrics{27}.function = 'pp_samplitude_diff2_11';
            metrics{27}.type = 'metric_nonshuffled';
            metrics{27}.baseline_type = '';
            metrics{27}.comparison_type = 'scanpath-pp_scanpath';
            metrics{27}.scoring = -1;
            metrics{27}.indexes_other = [];
            metrics{27}.trials = [];
            
            metrics{28}.name = 'corrSaccadeAmplitude';
            metrics{28}.function = 'pp_corrsa';
            metrics{28}.type = 'metric_nonshuffled';
            metrics{28}.baseline_type = '';
            metrics{28}.comparison_type = 'scanpath-pp_scanpath';
            metrics{28}.scoring = -1;
            metrics{28}.indexes_other = [];
            metrics{28}.trials = [];
            
            metrics{29}.name = 'covSaccadeAmplitude';
            metrics{29}.function = 'pp_covsa';
            metrics{29}.type = 'metric_nonshuffled';
            metrics{29}.baseline_type = '';
            metrics{29}.comparison_type = 'scanpath-pp_scanpath';
            metrics{29}.scoring = -1;
            metrics{29}.indexes_other = [];
            metrics{29}.trials = [];

%LeMeur To do Metrics: DTW (scanpath comparison), HR (hit rate) and R (mean of all metrics)

end



%% old / deprecated


%             metrics{9}.name = 'centdist';
%             metrics{9}.function = 'centdist';
%             metrics{9}.type = 'metric_shuffled_type5';
%             metrics{9}.baseline_type = 'max';
%             metrics{9}.comparison_type = 'smap-fbaseline';
%             metrics{9}.scoring = 1;
%             metrics{9}.indexes_other = indexes_other_trials;  %permutations of other trials
%             metrics{9}.trials = n_trials1;  %baseline compute all trials (limit of 100)

%             metrics{10}.name = 'sAUC_Murray';
%             metrics{10}.function = 'get_roc3';
%             metrics{10}.type = 'metric_shuffled_type3';
%             metrics{10}.baseline_type = 'cell';
%             metrics{10}.comparison_type = 'smap-bmap-sbmap';
%             metrics{10}.scoring = 1;
%             metrics{10}.indexes_other = indexes_other_trials;
%             metrics{10}.trials = n_trials1;
            
%             metrics{11}.name = 'sKL_Murray';
%             metrics{11}.function = 'get_kl3';
%             metrics{11}.type = 'metric_shuffled_type3';
%             metrics{11}.baseline_type = 'cell';
%             metrics{11}.comparison_type = 'smap-bmap-sbmap';
%             metrics{11}.scoring = 1;
%             metrics{11}.indexes_other = indexes_other_trials;
%             metrics{11}.trials = n_trials1;
%             metrics{15}.name = 'SaccadeLanding_saccades';
%             metrics{15}.function = 'pp_slanding';
%             metrics{15}.type = 'metric_nonshuffled';
%             metrics{15}.baseline_type = '';
%             metrics{15}.comparison_type = 'scanpath_saccades-pp_scanpath';
%             metrics{15}.scoring = -1;
%             metrics{15}.indexes_other = [];
%             metrics{15}.trials = [];

%             metrics{16}.name = 'SaccadeAmplitude_saccades';
%             metrics{16}.function = 'samplitude';
%             metrics{16}.type = 'metric_nonshuffled';
%             metrics{16}.baseline_type = '';
%             metrics{16}.comparison_type = 'scanpath_saccades_single';
%             metrics{16}.scoring = -1;
%             metrics{16}.indexes_other = [];
%             metrics{16}.trials = [];

%             metrics{15}.name = 'SaccadeLanding_group';
%             metrics{15}.function = 'slanding_group';
%             metrics{15}.type = 'metric_nonshuffled';
%             metrics{15}.baseline_type = '';
%             metrics{15}.comparison_type = 'scanpath-scanpath-gaze';
%             metrics{15}.scoring = -1;
%             metrics{15}.indexes_other = [];
%             metrics{15}.trials = [];
% 
%             metrics{16}.name = 'SaccadeAmplitude_group';
%             metrics{16}.function = 'samplitude_group';
%             metrics{16}.type = 'metric_nonshuffled';
%             metrics{16}.baseline_type = '';
%             metrics{16}.comparison_type = 'scanpath-scanpath-scanpath_next-gaze';
%             metrics{16}.scoring = -1;
%             metrics{16}.indexes_other = [];
%             metrics{16}.trials = [];



