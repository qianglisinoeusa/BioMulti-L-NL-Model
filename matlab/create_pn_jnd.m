function pn = create_pn_jnd( metric_par )
% Create lookup tables for intensity -> JND mapping

c_l = logspace( -5, 5, 2048 );

s_A = hdrvdp_joint_rod_cone_sens( c_l, metric_par );
s_R = hdrvdp_rod_sens( c_l, metric_par ) * 10.^metric_par.rod_sensitivity;

% s_C = s_L = S_M
s_C = 0.5 * interp1( c_l, max(s_A-s_R, 1e-3), min( c_l*2, c_l(end) ) );

pn = struct();

[pn.Y{1} pn.jnd{1}] = build_jndspace_from_S( log10(c_l), s_C );
[pn.Y{2} pn.jnd{2}] = build_jndspace_from_S( log10(c_l), s_R );

end