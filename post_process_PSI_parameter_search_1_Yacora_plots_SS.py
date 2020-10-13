# 27/02/2020
# This is an extract from post_process_PSI_parameter_search_1_Yacora.py
# only to have them in a separate place.

try:
	print('global_pass_SS_'+str(global_pass))
except Exception as e:
	print(e)
	global_pass = 2
	print('global_pass_SS_'+str(global_pass))

figure_index = 0


# NOTE  the first value of the "_full" arrays is for Lyman alpha. I use it only to estimate n=2 population density
energy_difference_full = np.array([10.1988,1.88867, 2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520, 3.31917])  # eV
# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
statistical_weigth_full = np.array([8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338])  # gi-gk
einstein_coeff_full = np.array([4.6986e+00, 4.4101e-01, 8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4, 1.8927e-04]) * 1e8  # 1/s

excitation_full = []
for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
	if isel==0:
		temp = read_adf15(pecfile_2, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	else:
		temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
	excitation_full.append(temp)
excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

recombination_full = []
for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
	if isel==0:
		temp = read_adf15(pecfile_2, 67, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	else:
		temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
	recombination_full.append(temp)
recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV

T_Hp = np.min([np.max([1000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited),merge_Te_prof_multipulse_interp_crop_limited/eV_to_K],axis=0),12000*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)],axis=0)	# K
# T_Hm = (273+20+temp_increase)*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)	# K
# T_H2p = (273+20+temp_increase)*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)	# K
# T_H2 = (273+20+temp_increase)*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)	# K
T_Hm = 1*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)/eV_to_K	# K
T_H2p = 1*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)/eV_to_K	# K
T_H2 = 1*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)/eV_to_K	# K
T_H = 1*np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)/eV_to_K	# K


population_coefficients = ((excitation_full *  nH_ne_all).T /multiplicative_factor_full).T
population_coefficients += ((recombination_full * nHp_ne_all).T /multiplicative_factor_full).T

temp1 = From_Hn_with_Hp_pop_coeff_full(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_Hp.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nHp_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,excited_states_From_Hn_with_Hp)
temp2 = From_Hn_with_H2p_pop_coeff_full(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_H2p.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,excited_states_From_Hn_with_H2p)
population_coefficients += (nHm_ne_all.flatten()*( temp1 + temp2 ).T).reshape((np.shape(population_coefficients)))

temp = From_H2_pop_coeff_full(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,excited_states_From_H2)
population_coefficients += (nH2_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

temp = From_H2p_pop_coeff_full(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,excited_states_From_H2p)
population_coefficients += (nH2p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

temp = From_H3p_pop_coeff_full(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,excited_states_From_H3p)
population_coefficients += (nH3p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

population_states = population_coefficients * merge_ne_prof_multipulse_interp_crop_limited**2 * 1e40


plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
population_states_H2 = np.zeros((15,*np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
for v_index in range(15):
	if v_index==0:
		population_states_H2[0] = merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		population_states_H2[v_index] = merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr

profile_centres_score[np.abs(profile_centres_score)>np.max(TS_r)] = np.max(TS_r/1000)
plt.figure(figsize=(20, 10))
plt.errorbar(TS_r/1000, merge_ne_prof_multipulse,yerr=merge_dne_multipulse);
plt.errorbar([profile_centres[0]/1000,profile_centres[0]/1000],np.sort(merge_ne_prof_multipulse)[[0,-1]],xerr=[profile_centres_score[0]/1000,profile_centres_score[0]/1000],color='b',label='found centre')
plt.plot(np.ones((2))*(profile_centres[0]+profile_sigma[0]*2.355/2)/1000,np.sort(merge_ne_prof_multipulse)[[0,-1]],'--',color='grey',label='FWHM')
plt.plot(np.ones((2))*(profile_centres[0]-profile_sigma[0]*2.355/2)/1000,np.sort(merge_ne_prof_multipulse)[[0,-1]],'--',color='grey')
plt.plot([centre/1000,centre/1000],np.sort(merge_ne_prof_multipulse)[[0,-1]],'k--',label='centre')
plt.errorbar(r_crop+centre/1000,merge_ne_prof_multipulse_interp_crop_limited,yerr=merge_dne_prof_multipulse_interp_crop_limited,color='r',label='averaged')
plt.plot(np.ones((2))*(averaged_profile_sigma*2.355/2+centre/1000),np.sort(merge_ne_prof_multipulse)[[0,-1]],'r--',label='density FWHM')
plt.legend(loc='best')
plt.xlabel('radial location [m]')
plt.ylabel('[10^20 # m^-3]')
plt.title('untreated electron density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor])+'\centre at %.3gmm' %(centre)+'\nheat input from upstream %.3g[W], heat output to target %.3g[W], Power balance %.3g[W]\n Maximum estimated density for H2 %.3g[10^20 # m^-3],upstream_electron_particle_flux %.3g[#/s],target_electron_particle_flux %.3g[#/s] ' %(heat_inflow_upstream,heat_flux_target,heat_inflow_upstream-heat_flux_target,max_nH2_from_pressure_all/1e20,upstream_electron_particle_flux,target_electron_particle_flux))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.errorbar(TS_r/1000, merge_Te_prof_multipulse,yerr=merge_dTe_multipulse);
plt.errorbar([profile_centres[1]/1000,profile_centres[1]/1000],np.sort(merge_Te_prof_multipulse)[[0,-1]],xerr=[profile_centres_score[1]/1000,profile_centres_score[1]/1000],color='b',label='found centre')
plt.plot(np.ones((2))*(profile_centres[1]+profile_sigma[1]*2.355/2)/1000,np.sort(merge_Te_prof_multipulse)[[0,-1]],'--',color='grey',label='FWHM')
plt.plot(np.ones((2))*(profile_centres[1]-profile_sigma[1]*2.355/2)/1000,np.sort(merge_Te_prof_multipulse)[[0,-1]],'--',color='grey')
plt.plot([centre/1000,centre/1000],np.sort(merge_Te_prof_multipulse)[[0,-1]],'k--',label='centre')
plt.errorbar(r_crop+centre/1000,merge_Te_prof_multipulse_interp_crop_limited,yerr=merge_dTe_prof_multipulse_interp_crop_limited,color='r',label='averaged')
plt.plot(np.ones((2))*(averaged_profile_sigma*2.355/2+centre/1000),np.sort(merge_Te_prof_multipulse)[[0,-1]],'r--',label='density FWHM')
plt.legend(loc='best')
plt.xlabel('radial location [m]')
plt.ylabel('[eV]')
plt.title('untreated electron temperature\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor])+'\centre at %.3gmm' %(centre)+'\nheat input from upstream %.3g[W], heat output to target %.3g[W], Power balance %.3g[W]\n Maximum estimated density for H2 %.3g[10^20 # m^-3],upstream_electron_particle_flux %.3g[#/s],target_electron_particle_flux %.3g[#/s] ' %(heat_inflow_upstream,heat_flux_target,heat_inflow_upstream-heat_flux_target,max_nH2_from_pressure_all/1e20,upstream_electron_particle_flux,target_electron_particle_flux))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


plt.figure(figsize=(20, 10))
plt.plot(r_crop, nHp_ne_all, label='nHp_ne');
plt.plot(r_crop, nH_ne_all, label='nH_ne');
plt.plot(r_crop, nHm_ne_all, label='nHm_ne');
plt.plot(r_crop, nH2_ne_all, label='nH2_ne');
plt.plot(r_crop, nH2p_ne_all, label='nH2p_ne');
plt.semilogy()
plt.legend(loc='best')
plt.ylim(bottom=1e-5)
plt.ylabel('[#/# ne]')
plt.xlabel('radial location [m]')
plt.title(
	'species density / electron density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
		[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited, label='ne');
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited *nHp_ne_all, label='nHp');
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited *nH_ne_all, label='nH');
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited *nHm_ne_all, label='nHm');
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited *nH2_ne_all, label='nH2');
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited *nH2p_ne_all, label='nH2p');
plt.semilogy()
plt.legend(loc='best')
plt.ylim(bottom=np.min(merge_ne_prof_multipulse_interp_crop_limited*1e-5))
plt.ylabel('[10^20 # m^-3]')
plt.xlabel('radial location [m]')
plt.title(
	'species density\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
		[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

charge_unbalance = -np.ones_like(merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all - nHm_ne_all + nH2p_ne_all + nH3p_ne_all
plt.figure(figsize=(20, 10))
plt.plot(r_crop, charge_unbalance);
plt.xlabel('radial location [m]')
plt.ylabel('Charge density/ne [au]')
plt.title(
	'Fractional charge unbalance\n(0=neutral, 1=positive charge, -1=negative charge)\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
		[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(r_crop, residuals_all);
plt.xlabel('radial location [m]')
plt.ylabel('[W m^-3]')
plt.title(
	'relative residual unaccounted line emission\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
		[1, spatial_factor, time_shift_factor]) + '\nsum = ' + str(np.sum(residuals_all)))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
temp[np.isnan(temp)] = 0
effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) * (merge_ne_prof_multipulse_interp_crop_limited * (10 ** 20) * nH_ne_all) ).astype('float')

temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
temp[np.isnan(temp)] = 0
effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
effective_recombination_rates = (effective_recombination_rates *nHp_ne_all* (merge_ne_prof_multipulse_interp_crop * (10 ** 20)) ** 2).astype('float')


plt.figure(figsize=(20, 10))
plt.plot(r_crop, effective_ionisation_rates,label='ionisation');
plt.plot(r_crop, effective_recombination_rates,label='recombination');
plt.legend(loc='best')
plt.xlabel('radial location [m]')
plt.ylabel('effective rates [# m^-3 s-1]')
plt.semilogy()
plt.title('effective rates from ADAS\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
# plt.title('effective_ionisation_rates')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure(figsize=(20, 10))
plt.plot(r_crop, effective_ionisation_rates,label='ionisation');
plt.plot(r_crop, effective_recombination_rates,label='recombination');
plt.legend(loc='best')
plt.xlabel('radial location [m]')
plt.ylabel('effective rates [# m^-3 s-1]')
plt.title('effective rates from ADAS\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
# plt.title('effective_ionisation_rates')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


# fig = plt.figure()
# # fig=plt.figure(merge_ID_target * 100 + 2)
# ax = fig.add_subplot(1, 10, 1)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,merge_Te_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
# # plt.set_sketch_params(scale=2)
# ax.set_aspect(40)
# plt.title('TS\nelectron temperature')
# plt.xlabel('radial location [mm]')
# plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('[eV]')
# # plt.axes().set_aspect(0.1)
#
#
# ax=fig.add_subplot(1, 10, 2)
# im=ax.pcolor(1000*temp_r.T,temp_t.T,merge_ne_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
# ax.set_aspect(40)
# plt.title('TS\nelectron density')
# plt.xlabel('radial location [mm]')
# plt.yticks([])
# # plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('[10^20 # m^-3]')
#
# ax=fig.add_subplot(1, 10, 3)
# im=ax.pcolor(1000*temp_r.T,temp_t.T,inverted_profiles_crop[:, 0].T,cmap='rainbow')
# ax.set_aspect(40)
# plt.title('OES\nn=4>2 line')
# plt.xlabel('radial location [mm]')
# plt.yticks([])
# # plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')
#
# ax=fig.add_subplot(1, 10, 4)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_ionisation_rates.T,cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e6,1e19), norm=LogNorm());
# ax.set_aspect(40)
# plt.title('effective ionisation\nrates')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')
#
# ax=fig.add_subplot(1, 10, 5)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_recombination_rates.T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('effective recombination\nrates')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')
#
# ax=fig.add_subplot(1, 10, 6)
# if np.sum(nH_ne_all>0)>0:
# 	im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH_ne_all).T, cmap='rainbow',vmin=max(np.max(merge_ne_prof_multipulse_interp_crop_limited*nH_ne_all)/1e6,1e-3), norm=LogNorm());
# else:
# 	im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')
#
#
# ax=fig.add_subplot(1, 10, 7)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nHp_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H+ density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')
#
#
# ax=fig.add_subplot(1, 10, 8)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nHm_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H- density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')
#
#
# ax=fig.add_subplot(1, 10, 9)
# if np.sum(nH2_ne_all>0)>0:
# 	im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all).T, cmap='rainbow',vmin=max(np.max(merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all)/1e6,1e-3), norm=LogNorm());
# else:
# 	im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H2 density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')
#
#
# ax=fig.add_subplot(1, 10, 10)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH2p_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H2+ density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')


# ax=fig.add_subplot(1, 6, 6)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,(merge_ne_prof_multipulse_interp_crop_limited*nH3p_ne_all).T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('H3+ density')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('density [# m^-3]')
#
# figure_index += 1
# # plt.pause(0.001)
# plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
# 	figure_index) + '.eps', bbox_inches='tight')
# plt.close()


# fig = plt.figure()
#
# ax = fig.add_subplot(1, 5, 1)
# im = ax.pcolor(1000*temp_r.T,temp_t.T,merge_Te_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
# # plt.set_sketch_params(scale=2)
# ax.set_aspect(40)
# plt.title('TS\nelectron temperature')
# plt.xlabel('radial location [mm]')
# plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('[eV]')
# # plt.axes().set_aspect(0.1)
#
#
# ax=fig.add_subplot(1, 5, 2)
# im=ax.pcolor(1000*temp_r.T,temp_t.T,merge_ne_prof_multipulse_interp_crop_limited.T,cmap='rainbow')
# ax.set_aspect(40)
# plt.title('TS\nelectron density')
# plt.xlabel('radial location [mm]')
# plt.yticks([])
# # plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('[10^20 # m^-3]')
#
#
# ax = fig.add_subplot(1, 5, 3)
# im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_ionisation_rates.T, cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e6,1e19), norm=LogNorm());
# ax.set_aspect(40)
# plt.title('effective ionisation\nrates')
# plt.xlabel('radial location [mm]')
# plt.ylabel('time [ms]')
# # plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')
#
# ax = fig.add_subplot(1, 5, 4)
# im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_recombination_rates.T, cmap='rainbow')
# ax.set_aspect(40)
# plt.title('effective recombination\nrates')
# plt.xlabel('radial location [mm]')
# # plt.ylabel('time [ms]')
# plt.yticks([])
# fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')
#
# ax=fig.add_subplot(1, 5, 5)
# im=ax.pcolor(1000*temp_r.T,temp_t.T,inverted_profiles_crop[:, 0].T,cmap='rainbow')
# ax.set_aspect(40)
# plt.title('OES\nn=4>2 line')
# plt.xlabel('radial location [mm]')
# plt.yticks([])
# # plt.ylabel('time [ms]')
# fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')
# figure_index += 1
# # plt.pause(0.001)
# plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
# 	figure_index) + '.eps', bbox_inches='tight')
# plt.close()


plt.figure(figsize=(20, 10))
# threshold_radious = 0.005
plt.plot(r_crop, effective_ionisation_rates/np.max(effective_ionisation_rates), label='ionisation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(effective_ionisation_rates)));
plt.plot(r_crop, effective_recombination_rates/np.max(effective_recombination_rates), label='recombination rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(effective_recombination_rates)));
plt.plot(r_crop, merge_Te_prof_multipulse_interp_crop_limited/np.max(merge_Te_prof_multipulse_interp_crop_limited),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(merge_Te_prof_multipulse_interp_crop_limited)));
plt.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited/np.max(merge_ne_prof_multipulse_interp_crop_limited),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(merge_ne_prof_multipulse_interp_crop_limited)));
plt.legend(loc='best')
plt.xlabel('radial location [mm]')
plt.ylabel('relative value [au]')
plt.title('Radial comparison '+str(target_OES_distance)+'mm from the target')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# if merge_ID_target in [851,85,90,91,92, 93, 94]:
# 	merge_ID_target_at_the_target = 91
#
# 	all_j=find_index_of_file(merge_ID_target_at_the_target,df_settings,df_log,only_OES=True)
# 	target_chamber_pressure_at_the_target = []
# 	target_OES_distance_at_the_target = []
# 	for j in all_j:
# 		target_chamber_pressure_at_the_target.append(df_log.loc[j,['p_n [Pa]']])
# 		target_OES_distance_at_the_target.append(df_log.loc[j,['T_axial']])
# 	target_chamber_pressure_at_the_target = np.nanmean(target_chamber_pressure_at_the_target)	# Pa
# 	target_OES_distance_at_the_target = np.nanmean(target_OES_distance_at_the_target)	# Pa
# 	# Ideal gas law
# 	max_nH2_from_pressure_at_the_target = target_chamber_pressure_at_the_target/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K
#
# 	path_where_to_save_everything_target = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target_at_the_target)
# 	merge_Te_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_Te_prof_multipulse']
# 	merge_dTe_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dTe_multipulse']
# 	merge_ne_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_ne_prof_multipulse']
# 	merge_dne_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dne_multipulse']
# 	merge_time_original_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_time']
# 	merge_time_target = time_shift_factor + merge_time_original_target
#
# 	profile_centres_target = []
# 	profile_centres_score_target = []
# 	for index in range(np.shape(merge_ne_prof_multipulse_target)[0]):
# 		yy = merge_ne_prof_multipulse_target[index]
# 		p0 = [np.max(yy), 10, 0]
# 		bds = [[0, -40, np.min(TS_r)], [np.inf, 40, np.max(TS_r)]]
# 		fit = curve_fit(gauss, TS_r, yy, p0, maxfev=100000, bounds=bds)
# 		profile_centres_target.append(fit[0][-1])
# 		profile_centres_score_target.append(fit[1][-1, -1])
# 	# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
# 	profile_centres_target = np.array(profile_centres_target)
# 	profile_centres_score_target = np.array(profile_centres_score_target)
# 	# centre = np.nanmean(profile_centres[profile_centres_score < 1])
# 	centre_target = np.nansum(profile_centres_target/(profile_centres_score_target**1))/np.sum(1/profile_centres_score_target**1)
# 	TS_r_new_target = (TS_r + centre_target) / 1000
# 	# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
# 	# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
# 	# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)
#
# 	temp1 = np.zeros_like(inverted_profiles[:, 0])
# 	temp2 = np.zeros_like(inverted_profiles[:, 0])
# 	temp3 = np.zeros_like(inverted_profiles[:, 0])
# 	temp4 = np.zeros_like(inverted_profiles[:, 0])
# 	interp_range_t = max(dt / 2, TS_dt) * 1
# 	interp_range_r = max(dx / 2, TS_dr) * 1
# 	weights_r = (np.zeros_like(merge_Te_prof_multipulse) + TS_r_new_target)/interp_range_r
# 	weights_t = (((np.zeros_like(merge_Te_prof_multipulse)).T + merge_time).T)/interp_range_t
# 	for i_t, value_t in enumerate(new_timesteps):
# 		if np.sum(np.abs(merge_time_target - value_t) < interp_range_t) == 0:
# 			continue
# 		for i_r, value_r in enumerate(np.abs(r)):
# 			if np.sum(np.abs(TS_r_new_target - value_r) < interp_range_r) == 0:
# 				continue
# 			elif np.sum(np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)) == 0:
# 				continue
# 			selected_values_t = np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)
# 			selected_values_r = np.abs(TS_r_new_target - value_r) < interp_range_r
# 			selecte_values = (np.array([selected_values_t])).T * selected_values_r
# 			selecte_values[merge_Te_prof_multipulse_target == 0] = False
# 			weights = 1/(weights_r[selecte_values]-value_r)**2 + 1/(weights_t[selecte_values]-value_t)**2
# 			if np.sum(selecte_values) == 0:
# 				continue
# 			# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
# 			# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
# 			temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse[selecte_values]*weights / merge_dTe_multipulse[selecte_values]) / np.sum(weights / merge_dTe_multipulse[selecte_values])
# 			# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
# 			# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
# 			temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse[selecte_values]*weights / merge_dne_multipulse[selecte_values]) / np.sum(weights / merge_dne_multipulse[selecte_values])
# 			if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
# 				# temp2[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dTe_multipulse[selecte_values]) ** 2)) ** 0.5
# 				# temp4[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dne_multipulse[selecte_values]) ** 2)) ** 0.5
# 				temp2[i_t, i_r] = 1/(np.sum(1 / merge_dTe_multipulse[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selecte_values])/merge_dTe_multipulse[selecte_values])**2 )**0.5)
# 				temp4[i_t, i_r] = 1/(np.sum(1 / merge_dne_multipulse[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selecte_values])/merge_dne_multipulse[selecte_values])**2 )**0.5)
# 			else:
# 				temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selecte_values])/merge_dTe_multipulse[selecte_values])**2 )**0.5)
# 				temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selecte_values])/merge_dne_multipulse[selecte_values])**2 )**0.5)
# 				temp2[i_t, i_r] = max(temp2_temp,(np.max(merge_Te_prof_multipulse[selecte_values])-np.min(merge_Te_prof_multipulse[selecte_values]))/2 )
# 				temp4[i_t, i_r] = max(temp4_temp,(np.max(merge_ne_prof_multipulse[selecte_values])-np.min(merge_ne_prof_multipulse[selecte_values]))/2 )
#
# 	merge_Te_prof_multipulse_interp_target = np.array(temp1)
# 	merge_dTe_prof_multipulse_interp_target = np.array(temp2)
# 	merge_ne_prof_multipulse_interp_target = np.array(temp3)
# 	merge_dne_prof_multipulse_interp_target = np.array(temp4)
# 	temp_r, temp_t = np.meshgrid(r, new_timesteps)
#
# 	# I crop to the usefull stuff
# 	start_time = np.abs(new_timesteps - 0).argmin()
# 	end_time = np.abs(new_timesteps - 1.5).argmin() + 1
# 	time_crop = new_timesteps[start_time:end_time]
# 	start_r = np.abs(r - 0).argmin()
# 	end_r = np.abs(r - 5).argmin() + 1
# 	r_crop = r[start_r:end_r]
# 	temp_r, temp_t = np.meshgrid(r_crop, time_crop)
# 	merge_Te_prof_multipulse_interp_crop_target = merge_Te_prof_multipulse_interp_target[start_time:end_time,
# 										   start_r:end_r]
# 	merge_dTe_prof_multipulse_interp_crop_target = merge_dTe_prof_multipulse_interp_target[
# 											start_time:end_time, start_r:end_r]
# 	merge_ne_prof_multipulse_interp_crop_target = merge_ne_prof_multipulse_interp_target[start_time:end_time,
# 										   start_r:end_r]
# 	merge_dne_prof_multipulse_interp_crop_target = merge_dne_prof_multipulse_interp_target[
# 											start_time:end_time, start_r:end_r]
#
#
#
# 	# x_local = xx - spatial_factor * 17.4 / 1000
# 	# dr_crop = np.median(np.diff(r_crop))
#
# 	merge_dTe_prof_multipulse_interp_crop_limited_target = cp.deepcopy(
# 		merge_dTe_prof_multipulse_interp_crop_target)
# 	merge_dTe_prof_multipulse_interp_crop_limited_target[
# 		merge_Te_prof_multipulse_interp_crop_target < 0.1] = 0
# 	merge_Te_prof_multipulse_interp_crop_limited_target = cp.deepcopy(
# 		merge_Te_prof_multipulse_interp_crop_target)
# 	merge_Te_prof_multipulse_interp_crop_limited_target[merge_Te_prof_multipulse_interp_crop_target < 0.1] = 0
# 	merge_dne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(
# 		merge_dne_prof_multipulse_interp_crop_target)
# 	merge_dne_prof_multipulse_interp_crop_limited_target[
# 		merge_ne_prof_multipulse_interp_crop_target < 5e-07] = 0
# 	merge_ne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(
# 		merge_ne_prof_multipulse_interp_crop_target)
# 	merge_ne_prof_multipulse_interp_crop_limited_target[
# 		merge_ne_prof_multipulse_interp_crop_target < 5e-07] = 0
#
# 	plt.figure();
# 	plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited_target, cmap='rainbow');
# 	plt.colorbar().set_label('[10^20 # m^-3]')  # ;plt.pause(0.01)
# 	plt.axes().set_aspect(20)
# 	plt.xlabel('time [ms]')
# 	plt.ylabel('radial location [m]')
# 	plt.title(
# 		'electron density at the target (%.3g mm from it)\nOES_multiplier,spatial_factor,time_shift_factor \n' %target_OES_distance_at_the_target + str(
# 			[1, spatial_factor, time_shift_factor]))
# 	figure_index += 1
# 	plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
# 		figure_index) + '.eps', bbox_inches='tight')
# 	plt.close()
#
# 	plt.figure();
# 	plt.pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited_target, cmap='rainbow');
# 	plt.colorbar().set_label('[eV]')  # ;plt.pause(0.01)
# 	plt.axes().set_aspect(20)
# 	plt.xlabel('time [ms]')
# 	plt.ylabel('radial location [m]')
# 	plt.title(
# 		'electron temperature at the target (%.3g mm from it)\nOES_multiplier,spatial_factor,time_shift_factor \n' %target_OES_distance_at_the_target + str(
# 			[1, spatial_factor, time_shift_factor]))
# 	figure_index += 1
# 	plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
# 		figure_index) + '.eps', bbox_inches='tight')
# 	plt.close()
#
# 	volume = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop]) * (target_OES_distance-target_OES_distance_at_the_target)/1000
# 	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
# 	sound_speed = ((merge_Te_prof_multipulse_interp_crop_limited_target/eV_to_K)*boltzmann_constant_J/(hydrogen_mass))**0.5
# 	ion_flux_target = np.sum(area *merge_ne_prof_multipulse_interp_crop_limited_target*sound_speed * 1e20 * np.median(np.diff(time_crop))/1000,axis=1)
# 	label_ion_sink_at_target = 'ion sink at the target using TS Te and ne at %.3g mm from it' %target_OES_distance_at_the_target
# else:
# 	merge_Te_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_Te_prof_multipulse_interp_crop_limited)
# 	merge_dTe_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dTe_prof_multipulse_interp_crop_limited)
# 	merge_ne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_ne_prof_multipulse_interp_crop_limited)
# 	merge_dne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dne_prof_multipulse_interp_crop_limited)
# 	volume = 2*np.pi*(r_crop + np.diff([0,*r_crop])/2) * np.diff([0,*r_crop]) * target_OES_distance/1000
# 	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
# 	sound_speed = ((merge_Te_prof_multipulse_interp_crop_limited_target/eV_to_K)*boltzmann_constant_J/(hydrogen_mass))**0.5
# 	ion_flux_target = np.sum(area *merge_ne_prof_multipulse_interp_crop_limited_target*sound_speed * 1e20 * np.median(np.diff(time_crop))/1000,axis=1)
# 	label_ion_sink_at_target = 'ion sink at the target using TS Te and ne at %.3g mm from it' %target_OES_distance


upstream_electron_particle_flux = merge_ne_prof_multipulse_interp_crop_limited * 10000* 1e20	# #/s
target_electron_particle_flux = 0.5*merge_ne_prof_multipulse_interp_crop_limited*1e20*(boltzmann_constant_J*(1+5/3)*merge_Te_prof_multipulse_interp_crop_limited/(eV_to_K*hydrogen_mass))**0.5	# #/s
area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))	# m^2
target_chamber_length = 0.351	# m
volume = area * (target_chamber_length+target_OES_distance/1000)

ionisation_source = volume * effective_ionisation_rates	# #/s
recombination_source = volume * effective_recombination_rates	# #/s
inflow_min = area * merge_ne_prof_multipulse_interp_crop_limited * 1000* 1e20	#steady state flow from upstream ~1km/s ballpark from CTS (Jonathan)
inflow_max = area * merge_ne_prof_multipulse_interp_crop_limited * 10000* 1e20	#peak flow from upstream ~10km/s ballpark from CTS (Jonathan)
total_e = volume * merge_ne_prof_multipulse_interp_crop_limited * 1e20
total_Hp = volume * merge_ne_prof_multipulse_interp_crop_limited * nHp_ne_all * 1e20
total_H = volume * merge_ne_prof_multipulse_interp_crop_limited * nH_ne_all * 1e20
total_Hm = volume * merge_ne_prof_multipulse_interp_crop_limited * nHm_ne_all * 1e20
total_H2 = volume * merge_ne_prof_multipulse_interp_crop_limited * nH2_ne_all * 1e20
total_H2p = volume * merge_ne_prof_multipulse_interp_crop_limited * nH2p_ne_all * 1e20
total_H3p = volume * merge_ne_prof_multipulse_interp_crop_limited * nH3p_ne_all * 1e20
# plt.figure();
fig, ax = plt.subplots(figsize=(20, 10))
ax.fill_between(r_crop, inflow_min,inflow_max, inflow_max>=inflow_min,color='green',alpha=0.1,label='electron inflow from upstream\nflow vel 1-10km/s (CTS ballpark)')#, label='inflow, 1-10km/s (CTS)');
ax.plot(r_crop, target_electron_particle_flux, label='target_electron_particle_flux tot=%.3g#/s' %(np.sum(target_electron_particle_flux)));
ax.plot(r_crop, ionisation_source, label='total ionisation source tot=%.3g#/s' %(np.sum(ionisation_source)));
ax.plot(r_crop, recombination_source, label='total recombination source tot=%.3g#/s' %(np.sum(recombination_source)));
ax.plot(r_crop, ionisation_source - recombination_source,'kv', label='ionisation - recombination tot=%.3g#/s' %(np.sum((ionisation_source - recombination_source)[ionisation_source - recombination_source>0])));
ax.plot(r_crop, -(ionisation_source - recombination_source),'k^', label='recombination - ionisation tot=%.3g#/s' %(np.sum((-(ionisation_source - recombination_source)[-(ionisation_source - recombination_source)>0]))));
ax.plot(r_crop, total_e, label='total munber of electrons tot=%.3g#' %(np.sum(total_e)));
ax.plot(r_crop, total_Hp, label='total munber of H+ tot=%.3g#' %(np.sum(total_Hp)));
ax.plot(r_crop, total_H, label='total munber of H tot=%.3g#' %(np.sum(total_H)));
ax.plot(r_crop, total_Hm, label='total munber of H- tot=%.3g#' %(np.sum(total_Hm)));
ax.plot(r_crop, total_H2, label='total munber of H2 tot=%.3g#' %(np.sum(total_H2)));
ax.plot(r_crop, total_H2p, label='total munber of H2+ tot=%.3g#' %(np.sum(total_H2p)));
ax.plot(r_crop, total_H3p, label='total munber of H3+ tot=%.3g#' %(np.sum(total_H3p)));
ax.legend(loc=1,fontsize='small')
ax.set_xlabel('radial location [mm]')
ax.set_ylabel('Particle balance [#]')
ax.set_title('Particles destroyed and generated via recombination and ionisation, from target to '+str(target_OES_distance)+'mm from it, chamber pressure %.3gPa' %(target_chamber_pressure))
ax.set_yscale('log')
ax.set_ylim(bottom=1e-8*np.max([inflow_max,target_electron_particle_flux,ionisation_source,recombination_source, total_e,total_Hp,total_H,total_Hm,total_H2,total_H2p,total_H3p]))

ax2 = ax.twinx()
ax2.plot(r_crop, merge_Te_prof_multipulse_interp_crop_limited/np.max(merge_Te_prof_multipulse_interp_crop_limited),'r--', label='Te TS\n(max='+'%.3g eV)' % np.max(merge_Te_prof_multipulse_interp_crop_limited));
ax2.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited/np.max(merge_ne_prof_multipulse_interp_crop_limited),'b--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(merge_ne_prof_multipulse_interp_crop_limited));
ax2.set_ylabel('Te and ne divided by max value [au]')
ax2.legend(loc=4,fontsize='small')

figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

RR_rate_creation_Hm_source = volume*RR_rate_creation_Hm(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_Hm_source = volume*RR_rate_destruction_Hm(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_H2_source = volume*RR_rate_creation_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_H2_source = volume*RR_rate_destruction_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_Hp_source = volume*RR_rate_creation_Hp(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_Hp_source = volume*RR_rate_destruction_Hp(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_e_source = volume*RR_rate_creation_e(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_e_source = volume*RR_rate_destruction_e(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_H_source = volume*RR_rate_creation_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_H_source = volume*RR_rate_destruction_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_H2p_source = volume*RR_rate_creation_H2p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_destruction_H2p_source = volume*RR_rate_destruction_H2p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
RR_rate_creation_H3p_source = volume*RR_rate_creation_H3p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,population_states,population_states_H2)
# plt.figure();
fig, ax = plt.subplots(figsize=(20, 10))
ax.fill_between(r_crop, inflow_min,inflow_max, inflow_max>=inflow_min,color='green',alpha=0.1,label='electron inflow from upstream\nflow vel 1-10km/s (CTS ballpark)')#, label='inflow, 1-10km/s (CTS)');
ax.plot(r_crop, target_electron_particle_flux, label='target_electron_particle_flux tot=%.3g#/s' %(np.sum(target_electron_particle_flux)));
ax.plot(r_crop, ionisation_source, label='total ionisation source tot=%.3g#/s' %(np.sum(ionisation_source)));
ax.plot(r_crop, recombination_source, label='total recombination source tot=%.3g#/s' %(np.sum(recombination_source)));
ax.plot(r_crop, ionisation_source - recombination_source,'kv', label='ionisation - recombination tot=%.3g#/s' %(np.sum((ionisation_source - recombination_source)[ionisation_source - recombination_source>0])));
ax.plot(r_crop, -(ionisation_source - recombination_source),'k^', label='recombination - ionisation tot=%.3g#/s' %(np.sum((-(ionisation_source - recombination_source)[-(ionisation_source - recombination_source)>0]))));
ax.plot(r_crop, RR_rate_creation_Hm_source, label='total H- creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_Hm_source)));
ax.plot(r_crop, RR_rate_destruction_Hm_source, label='total H- destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_Hm_source)));
ax.plot(r_crop, RR_rate_creation_H2_source, label='total H2 creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_H2_source)));
ax.plot(r_crop, RR_rate_destruction_H2_source, label='total H2 destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_H2_source)));
ax.plot(r_crop, RR_rate_creation_Hp_source, label='total H+ creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_Hp_source)));
ax.plot(r_crop, RR_rate_destruction_Hp_source, label='total H+ destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_Hp_source)));
ax.plot(r_crop, RR_rate_creation_e_source, label='total e creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_e_source)));
ax.plot(r_crop, RR_rate_destruction_e_source, label='total e destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_e_source)));
ax.plot(r_crop, RR_rate_creation_H_source, label='total H creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_H_source)));
ax.plot(r_crop, RR_rate_destruction_H_source, label='total H destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_H_source)));
ax.plot(r_crop, RR_rate_creation_H2p_source, label='total H2+ creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_H2p_source)));
ax.plot(r_crop, RR_rate_destruction_H2p_source, label='total H2+ destruction source tot=%.3g#/s' %(np.sum(RR_rate_destruction_H2p_source)));
ax.plot(r_crop, RR_rate_creation_H3p_source, label='total H3+ creation source tot=%.3g#/s' %(np.sum(RR_rate_creation_H3p_source)));
ax.plot(r_crop, total_e, label='total munber of electrons tot=%.3g#' %(np.sum(total_e)));
ax.plot(r_crop, total_Hp, label='total munber of H+ tot=%.3g#' %(np.sum(total_Hp)));
ax.plot(r_crop, total_H, label='total munber of H tot=%.3g#' %(np.sum(total_H)));
ax.plot(r_crop, total_Hm, label='total munber of H- tot=%.3g#' %(np.sum(total_Hm)));
ax.plot(r_crop, total_H2, label='total munber of H2 tot=%.3g#' %(np.sum(total_H2)));
ax.plot(r_crop, total_H2p, label='total munber of H2+ tot=%.3g#' %(np.sum(total_H2p)));
ax.plot(r_crop, total_H3p, label='total munber of H3+ tot=%.3g#' %(np.sum(total_H3p)));
ax.legend(loc=1,fontsize='small')
ax.set_xlabel('radial location [mm]')
ax.set_ylabel('Particle balance [#]')
ax.set_title('Particles destroyed and generated via recombination and excitation, from target to '+str(target_OES_distance)+'mm from it, chamber pressure %.3gPa' %(target_chamber_pressure))
ax.set_yscale('log')
ax.set_ylim(bottom=1e-8*np.max([inflow_max,target_electron_particle_flux,ionisation_source,recombination_source, total_e,total_Hp,total_H,total_Hm,total_H2,total_H2p,total_H3p]))

ax2 = ax.twinx()
ax2.plot(r_crop, merge_Te_prof_multipulse_interp_crop_limited/np.max(merge_Te_prof_multipulse_interp_crop_limited),'r--', label='Te TS\n(max='+'%.3g eV)' % np.max(merge_Te_prof_multipulse_interp_crop_limited));
ax2.plot(r_crop, merge_ne_prof_multipulse_interp_crop_limited/np.max(merge_ne_prof_multipulse_interp_crop_limited),'b--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(merge_ne_prof_multipulse_interp_crop_limited));
ax2.set_ylabel('Te and ne divided by max value [au]')
ax2.legend(loc=4,fontsize='small')

figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


# Calculation of the ionisation mean free path
# arbitrary_H_temp = 5000	# K, It is the same used for the fittings
thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
temp[np.isnan(temp)] = 0
temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
ionization_length_H = thermal_velocity_H/(temp * merge_ne_prof_multipulse_interp_crop_limited * 1e20 )
ionization_length_H = np.where(np.isnan(ionization_length_H), 0, ionization_length_H)
ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
plt.figure(figsize=(20, 10))
plt.plot(r_crop, ionization_length_H);
plt.xlabel('radial location [m]')
plt.ylabel('ionization_length [m], limited to 1m')
plt.ylim(bottom=np.min(ionization_length_H),top=min(np.max(ionization_length_H),1))
plt.semilogy()
plt.title('ionization length of neutral H from ADAS\nH temperature '+str(np.mean(T_H))+'K\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.close('all')


'''
# reaction rate		e +H(nl) → H- + hν, n ≥ 1
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 2.1.3
# info only on e +H(1s) → H- + hν, n ≥ 1
Eb = 0.754	# eV
beta = Eb/merge_Te_prof_multipulse_interp_crop_limited
reaction_rate = 1.17*( 2* np.pi**(1/2) * beta**(3/2) * np.exp(beta)+1 -2*beta* hyp1f1(1,0.5,beta) )*1e-10 * 1e-6		# m^3/s
reaction_rate[np.isnan(reaction_rate)] = 0
e_H__Hm = merge_ne_prof_multipulse_interp_crop_limited*1e20 * (merge_ne_prof_multipulse_interp_crop_limited * nH_ne_all* 1e20 - np.sum(population_states,axis=0) ) * reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, e_H__Hm ,vmin=max(np.max(e_H__Hm)*1e-12,np.min(e_H__Hm[e_H__Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H(1s) → H- + hν reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		e +H- → e +H + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.1.1
# as mentioned I neglect:
# e +H- → e +H(n ≥ 2) + e
# e +H- → e +H+ + 2e
# because negligible at low temp
# I'm not sure if only electron energy or also H-
# I'll assume it is electron energy because usually it is the highest
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
def internal(Te):
	Te = np.array([Te])
	T_e_temp = Te/eV_to_K	# K
	T_e_temp[T_e_temp==0]=300	# ambient temperature
	e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
	e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
	e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
	cross_section = 2.06/e_energy * np.log( np.exp(1) + 2.335 * 1e-4 * e_energy ) * np.exp(-13.75/(e_energy**0.868)) * (1-(0.754/e_energy)**2) * 1e-13 * 1e-4
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	# cross_section_min = cross_section.flatten()[(np.abs(e_energy-1)).argmin()]
	# cross_section[e_energy<1]=cross_section_min
	cross_section[cross_section<0]=0
	reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
	return reaction_rate
reaction_rate = list(map(internal,(merge_Te_prof_multipulse_interp_crop_limited.flatten())))
reaction_rate = np.reshape(reaction_rate,np.shape(merge_Te_prof_multipulse_interp_crop_limited))
e_Hm__e_H_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, e_Hm__e_H_e ,vmin=max(np.max(e_Hm__e_H_e)*1e-12,np.min(e_Hm__e_H_e[e_Hm__e_H_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H- → e +H + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H+ +H- →H(2) +H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.2.1
# I'm not sure if only H+ energy or also H-
# I'll assume it is H+ energy because usually it is the highest
# Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
# Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
# Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
# Hp_energy = np.array(([Hp_energy.tolist()])*9)
# coefficients = np.array([-3.49880888*1e1, 2.15245051*1e-1, -2.35628664*1e-2, 5.49471553*1e-2, 5.37932888*1e-3, -6.05507021*1e-3, 9.99168329*1e-4, -6.63625564*1e-5,1.61228385*1e-6])
# cross_section = (np.exp(np.sum(((np.log(Hp_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
# cross_section[np.logical_not(np.isfinite(cross_section))]=0
# cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
# cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
# cross_section[Hp_energy[0]<0.1]=cross_section_min
# cross_section[Hp_energy[0]>20000]=cross_section_max
# cross_section[cross_section<0] = 0
# reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
baricenter_impact_energy[baricenter_impact_energy<0.1]=0.1
baricenter_impact_energy[baricenter_impact_energy>20000]=20000
# baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
coefficients = np.array([-3.49880888*1e1, 2.15245051*1e-1, -2.35628664*1e-2, 5.49471553*1e-2, 5.37932888*1e-3, -6.05507021*1e-3, 9.99168329*1e-4, -6.63625564*1e-5,1.61228385*1e-6])
# cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* 1e-4
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
cross_section[baricenter_impact_energy<0.1]=cross_section_min
cross_section[baricenter_impact_energy>20000]=cross_section_max
cross_section[cross_section<0] = 0
reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
Hp_Hm__H_2_H = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*nHp_ne_all*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_Hm__H_2_H ,vmin=max(np.max(Hp_Hm__H_2_H)*1e-12,np.min(Hp_Hm__H_2_H[Hp_Hm__H_2_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H- →H(2) +H(1s) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H+ +H- →H(3) +H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.2.1
# I'm not sure if only H+ energy or also H-
# I'll assume it is H+ energy because usually it is the highest
# Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
# Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
# Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
# Hp_energy = np.array(([Hp_energy.tolist()])*9)
# coefficients = np.array([-3.11479336*1e+1, -7.73020527*1e-1, 5.49204378*1e-2, -2.73324984*1e-3, -1.22831288*1e-3, 4.35049828*1e-4, -6.21659501*1e-5, 4.12046807*1e-6, -1.039784996*1e-7])
# cross_section = (np.exp(np.sum(((np.log(Hp_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
# cross_section[np.logical_not(np.isfinite(cross_section))]=0
# cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
# cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
# cross_section[Hp_energy[0]<0.1]=cross_section_min
# cross_section[Hp_energy[0]>20000]=cross_section_max
# cross_section[cross_section<0] = 0
# reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
baricenter_impact_energy[baricenter_impact_energy<0.1]=0.1
baricenter_impact_energy[baricenter_impact_energy>20000]=20000
# baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
coefficients = np.array([-3.11479336*1e+1, -7.73020527*1e-1, 5.49204378*1e-2, -2.73324984*1e-3, -1.22831288*1e-3, 4.35049828*1e-4, -6.21659501*1e-5, 4.12046807*1e-6, -1.039784996*1e-7])
# cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* 1e-4
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
cross_section[baricenter_impact_energy<0.1]=cross_section_min
cross_section[baricenter_impact_energy>20000]=cross_section_max
cross_section[cross_section<0] = 0
reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
Hp_Hm__H_3_H = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*nHp_ne_all*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_Hm__H_3_H ,vmin=max(np.max(Hp_Hm__H_3_H)*1e-12,np.min(Hp_Hm__H_3_H[Hp_Hm__H_3_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H- →H(3) +H(1s) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H+ +H- → H2+(v) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.2.2
# valid from at least 1e-3eV
Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
cross_section = 1.38 / (baricenter_impact_energy**0.85 * (1+0.065 * baricenter_impact_energy**2.7)) * 1e-16 *1e-4
cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy-0.001)).argmin()]
cross_section[baricenter_impact_energy<0.001]=cross_section_min
cross_section[cross_section<0] = 0
reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
Hp_Hm__H2p_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*nHp_ne_all*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_Hm__H2p_e ,vmin=max(np.max(Hp_Hm__H2p_e)*1e-12,np.min(Hp_Hm__H2p_e[Hp_Hm__H2p_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H- → H2+(v) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H- +H(1s) →H2-(B2Σ+g ) →H(1s) +H(1s) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.3.2
H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/hydrogen_mass)**0.5).T
H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hm),200))+H_energy).T + Hm_energy).T
baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
coefficients = np.array([-3.61799082*1e+1, 1.16615172, -1.41928602*1e-1, -1.11195959*1e-2, -1.72505995*1e-3, 1.59040356*1e-3, -2.53196144*1e-4, 1.66978235*1e-5, -4.09725797*1e-7])
cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-0.1)).argmin()]
cross_section_max = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-20000)).argmin()]
cross_section[baricenter_impact_energy[0]<0.1]=cross_section_min
cross_section[baricenter_impact_energy[0]>20000]=cross_section_max
cross_section[cross_section<0] = 0
reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
Hm_H1s__H1s_H1s_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*(merge_ne_prof_multipulse_interp_crop_limited*1e20*nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hm_H1s__H1s_H1s_e ,vmin=max(np.max(Hm_H1s__H1s_H1s_e)*1e-12,np.min(Hm_H1s__H1s_H1s_e[Hm_H1s__H1s_H1s_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H- +H(1s) →H2-(B2Σ+g ) →H(1s) +H(1s) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H- +H(1s) →H2-(X2Σ+u ; B2Σ+g ) →H2(X1Σ+ g ; v) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 3.3.2
H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/hydrogen_mass)**0.5).T
H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hm),200))+H_energy).T + Hm_energy).T
baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
coefficients = np.array([-3.44152907*1e+1, -3.39348209*1e-1, 5.66591705*1e-2, -9.05150459*1e-3, 7.66060418*1e-4, -4.27126462*1e-5, -1.57273749*1e-7, 2.57607677*1e-7, -1.20071919*1e-8])
cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-0.1)).argmin()]
cross_section_max = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-20000)).argmin()]
cross_section[baricenter_impact_energy[0]<0.1]=cross_section_min
cross_section[baricenter_impact_energy[0]>20000]=cross_section_max
cross_section[cross_section<0] = 0
reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
Hm_H1s__H2_v_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nHm_ne_all*(merge_ne_prof_multipulse_interp_crop_limited*1e20*nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hm_H1s__H2_v_e ,vmin=max(np.max(Hm_H1s__H2_v_e)*1e-12,np.min(Hm_H1s__H2_v_e[Hm_H1s__H2_v_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H- +H(1s) →H2-(X2Σ+u ; B2Σ+g ) →H2(X1Σ+ g ; v) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H+ +H(1s) +H(1s) →H+ +H2(v)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 2.2.4, equation 46a
# valid for temperature up to 30000K (~2.587eV)
# I'm not sure which temperature to use here. I take a mean of T_H+, T_H, T_H
temperature = np.mean(np.array([T_Hp,T_H,T_H]),axis=0)
reaction_rate = 1.145/(( temperature )**1.12) *1e-29 * 1e-6 * 1e-6		# m^6/s
reaction_rate[np.logical_not(np.isfinite(reaction_rate))] = 0
reaction_rate_max = reaction_rate.flatten()[(np.abs(temperature-30000)).argmin()]
reaction_rate[temperature>30000]=reaction_rate_max
Hp_H_H__H_H2 = merge_ne_prof_multipulse_interp_crop_limited*1e20 * nHp_ne_all*((merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))**2)*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_H_H__H_H2 ,vmin=max(np.max(Hp_H_H__H_H2)*1e-12,np.min(Hp_H_H__H_H2[Hp_H_H__H_H2>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H(1s) +H(1s) →H+ +H2(v) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H +H +H →H +H2(v)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 2.3.4
# valid for temperature up to 20000-30000K (~2eV)
reaction_rate = 1.65/T_H *1e-30 * 1e-6 * 1e-6		# m^6/s
reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H-20000)).argmin()]
reaction_rate[T_H>20000]=reaction_rate_max
H_H_H__H_H2 = ((merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all)**3)*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H_H_H__H_H2 ,vmin=max(np.max(H_H_H__H_H2)*1e-12,np.min(H_H_H__H_H2[H_H_H__H_H2>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H(1s) +H(1s) →H+ +H2(v) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		e+H2(X1Σg;v) → e+H2∗(N1,3Λσ;eps) → e+H(1s)+H(nl)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# valid for energy up to ~1000eV
# Vibrational partition function
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
# electronic partition function approximated by g0=1
# I think I need to consider only the electron energy and not H2
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
# DISSOCIATION VIA SINGLETS

# VIBRATIONAL STATE = 0
# chapter 4.2.1a + c
# I need to loop on the output excited molecular state
# dipole allowed X 1 Σ +g (v = 0) → Σ u , Π u excitations with N = 2-4
# coefficients: ∆E(eV ), α, A 1, A 2, A 3, A 4, % dissociation vs excitation
coefficients_all = np.array([[12.754,0.550,3.651e-2,-0.8405,1.2365,2.5236,0.44],[14.85,0.550,6.827e-3,-0.1572,0.23122,0.47191,38.5],[15.47,0.550,2.446e-3,-5.631e-2,6.2846e-2,0.16908,4.41],[13.29,0.552,3.653e-2,-0.8398,1.2368,2.8740,1.1],[14.996,0.552,8.913e-3,-0.2049,0.30178,0.70126,0.92],[15.555,0.552,3.872e-3,-8.902e-2,0.13110,0.30464,1.2]])
reaction_rate = 0
for coefficients in coefficients_all:
	x = e_energy/coefficients[0]
	cross_section = 5.984/e_energy*((1-1/x)**coefficients[1]) * ( coefficients[2] + coefficients[3]/x + coefficients[4]/(x**2) + coefficients[5]*np.log(x)) * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section = cross_section * coefficients[6] /100
	cross_section[cross_section<0] = 0
	reaction_rate += np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
# symmetry-forbidden transitions X1Σ+ g (v = 0) →N1Λg
# coefficients: ∆E(ev),α,A 1, % dissociation vs excitation
coefficients_all = np.array([[13.13,2.71,0.8322,4.1],[14.98,2.71,2.913e-2,4.1],[14.816,2.75,1.43e-2,4.1],[14.824,2.80,5.40e-2,4.1]])
for coefficients in coefficients_all:
	x = e_energy/coefficients[0]
	cross_section = 5.984/e_energy*((1-1/x)**coefficients[1]) * coefficients[2] * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section = cross_section * coefficients[2] /100
	cross_section[cross_section<0] = 0
	reaction_rate += np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
e_H2_X1Σg_x__e_H_H = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
e_H2_X1Σg_x__e_H_H[0] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
# VIBRATIONAL STATE > 0
# chapter 4.2.1b + c
# I loop also on the vibrational state
# I need to loop on the output excited molecular state
# dipole-allowed transitions to N1Σ+ and N1Πu states with N = 2-4
# X1Σ+(v) → B1Σu, C1Πu
# coefficients: a 1,a 2,a 3,a 4
coefficients_all1 = np.array([[0.9754,0.3708,-0.2800,0.5479],[1.1106,0.8921,-0.2019,0.6545]])
# coefficients: v = 1, ... , 14
coefficients_all2 = np.array([[10.75,10.13,9.62,9.19,8.81,8.48,8.21,7.97,7.80,7.68,7.64,7.70,7.94,9.11],[11.61,11.12,10.81,10.53,10.30,10.12,9.97,9.86,9.80,9.78,9.79,9.88,10.01,10.21]])
# coefficients: % dissociation vs excitation
coefficients_all3 = np.array([[2.18,5.58,8.78,7.98,5.67,5.55,6.39,5.80,4.64,4.24,4.51,4.23,3.44,2.56],[5.71,13.0,16.2,13.1,12.9,15.4,13.8,13.1,13.4,12.7,10.5,8.92,7.22,6.93]])
for index,coefficients1 in enumerate(coefficients_all1):
	for v_index,coefficients2,coefficients3 in (np.array([np.arange(14),coefficients_all2[index],coefficients_all3[index]]).T):
		v_index = int(v_index)
		x = e_energy/coefficients2
		shape_function = (coefficients1[0]/x) * ( (1-1/x)**coefficients1[1] ) * ( coefficients1[2] + coefficients1[3]/x + np.log(x) ) * 1e-16 * 1e4	# m^2
		cross_section = shape_function * ((13.6/coefficients2)**3)	# m^2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section = cross_section * coefficients3 /100
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^6 / s
		e_H2_X1Σg_x__e_H_H[v_index+1] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index+1)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
# X1Σ+(v) → B', B''1Σu,D,D'1Πu
# coefficients: ∆E XΛ (R v t )(eV ),b 1,b 2,b 3,b 4
# v=0,...,13
coefficients_all1_1 = np.array([[14.85,0.1688,1.2433,0.8581,1.0069],[14.99,0.3226,0.4512,-0.277,0.5916],[15.67,0.0651,1.1633,0.8448,1.114],[15.66,0.1256,0.7526,-0.111,0.5811]])
coefficients_all1_2 = np.array([[13.09,0.1937,1.3554,0.93,0.9659],[13.35,0.3696,0.6749,-0.3478,0.7568],[13.86,0.0746,1.2327,0.8157,1.0515],[14.04,0.1471,0.7778,-0.2842,0.6728]])
# coefficients: % dissociation vs excitation
coefficients_all2 = np.array([[40.2,37.7,43.5,43.3,48.4,55.1,61.3,72.0,79.7,90.8,90.5,84.0,45.8,27.7],[18.5,29.4,27.0,31.6,39.2,40.2,50.9,60.6,64.7,75.5,59.9,29.2,11.4,4.10],[5.27,13.2,17.4,15.1,13.85,16.3,16.0,13.1,13.8,13.9,11.4,8.71,6.06,3.87],[6.44,15.2,19.0,15.5,14.4,18.8,18.5,16.6,17.0,16.2,13.2,10.1,7.21,4.10]])
# coefficients: C1, ... , 12
coefficients_all3 = np.array([[0,0,2,0,1.1,-1.30e-1,1.94e-2,7.0e-1,2.42e-2,-3.21e-3,-1.08e-4,1.36e-4],[0,0,-2.01e-1,0,1.1,-1.30e-1,1.26e-2,5.0e-1,3.44e-2,-5.40e-3,-9.20e-5,2.55e-4],[2.93e-3,2.25,1,2.00e-1,1,0,0,0,0,0,0,0],[1.10e-4,3.6,5.50e-1,2.6e-1,1,0,0,0,0,0,0,0]])
# prolem: I need the ∆EXΛ(Rtv) is the vertical transition energy for B', B''1Σu,D,D'1Πu and v>1
# I approximate neglecting it and using the value at v=1
coefficients_all4 = np.array([[13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09,13.09],[13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35,13.35],[13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86,13.86],[14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04,14.04]])
for index_reaction in range(4):
	for v_index in range(14):
		if v_index==0:
			coefficients1 = coefficients_all1_1[index_reaction,1:]
			coefficients2 = coefficients_all1_1[index_reaction,0]
		else:
			coefficients1 = coefficients_all1_2[index_reaction,1:]
			coefficients2 = coefficients_all1_2[index_reaction,0]
		coefficients3 = coefficients_all2[index_reaction,v_index]
		x = e_energy/coefficients2
		shape_function = (coefficients1[0]/x) * ( (1-1/x)**coefficients1[1] ) * ( coefficients1[2] + coefficients1[3]/x + np.log(x) ) * 1e-16 * 1e4	# m^2
		if v_index in [0]:
			cross_section = shape_function * 1	# m^2
		else:
			coefficients4 = coefficients_all4[index_reaction,v_index-1]
			coefficients5 = coefficients_all3[index_reaction]
			cross_section = shape_function * ((coefficients2/coefficients4)**((1+coefficients5[0]*(v_index+1)**coefficients5[1])*(coefficients5[2] + (2/x)**coefficients5[3])))*(coefficients5[4] + coefficients5[5]*(v_index+1) + (coefficients5[6]/(x**coefficients5[7])+coefficients5[8])*((v_index+1)**2) + coefficients5[9]*((v_index+1)**3) + (coefficients5[10]/(x**coefficients5[7])+coefficients5[11])*((v_index+1)**4))	# m^2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section = cross_section * coefficients3 /100
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		if v_index==0:
			e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr

# DISSOCIATION VIA TRIPLETS

# VIBRATIONAL STATE = 0
# chapter 4.2.2a
# through b3Σ+ and e3Σ+
# coefficients: A,β,γ,∆E(eV ), % dissociation vs excitation
coefficients_all = np.array([[11.16,2.33,3.78,7.93,100],[0.190,4.5,1.60,13.0,20]])
for coefficients in coefficients_all:
	x = e_energy/coefficients[3]
	cross_section = coefficients[0]/(x**3) * ((1-1/(x**coefficients[1]))**coefficients[2]) * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section = cross_section * coefficients[4] /100
	cross_section[cross_section<0] = 0
	reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
	e_H2_X1Σg_x__e_H_H[0] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
# VIBRATIONAL STATE 0 - 10
# chapter 4.2.2b
# through b3Σu, a3Σg, c3Πu only to dissociation
# coefficients: b1,...,b6
# valid 1000 to 200000K
coefficients_all = np.array([[-11.565,-7.6012e-2,-78.433,0.7496,-2.2126,0.22006],[-12.035,-6.6082e-2,-67.806,0.72403,-1.5419,1.5195],[-13.566,-4.3737e-2,-55.933,0.72286,-2.3103,1.5844],[-46.664,0.74122,-15.297,-2.2384e-2,-1.3674,1.3621],[-37.463,0.81763,-0.40373,-0.45851,-18.093,1.1460e-2],[-28.283,0.99053,-10.377,-8.5590e-2,-11.053,6.7271e-2],[-23.724,1.0112,-2.9905,-0.24701,-17.931,3.4376e-2],[-19.547,1.0224,-1.7489,-0.31413,-19.408,2.8643e-2],[-15.937,1.0213,-10175,-0.3804,-20.24,2.4170e-2],[-12.712,1.0212,-0.604,-0.44572,-20.766,2.1159e-2],[-0.40557,-0.49721,-9.9025,1.0212,-21.031,1.9383e-2]])
for v_index,coefficients in enumerate(coefficients_all):
	T_scaled = T_e_temp/1000
	reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**coefficients[5]) ) 		# m^3 / s
	reaction_rate_min = reaction_rate.flatten()[(np.abs(T_e_temp-1000)).argmin()]
	reaction_rate[T_e_temp<1000]=reaction_rate_min
	if v_index==0:
		e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
e_H2_X1Σg_x__e_H_H = np.sum(e_H2_X1Σg_x__e_H_H,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2_X1Σg_x__e_H_H ,vmin=max(np.max(e_H2_X1Σg_x__e_H_H)*1e-12,np.min(e_H2_X1Σg_x__e_H_H[e_H2_X1Σg_x__e_H_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e+H2(X1Σg;v) → e+H2∗(N1,3Λσ;eps) → e+H(1s)+H(nl), v=0-14, singlets and triplets reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 4.3.1
# I neglect what happens for electronic excited states because they will anyway be less populated
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
# electronic partition function approximated by g0=1
# I think I need to consider only the electron energy and not H2
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
e_H2N13Λσ__e_H2pX2Σg_e = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# coefficients: ∆Ev(eV )  NOTE except for v=0 the rest are all guesses
coefficients_all = np.array([15.42,13.8074509803922,12.6988235294118,11.7917647058824,11.0358823529412,10.3807843137255,9.87686274509804,9.32254901960784,8.86901960784314,8.41549019607843,8.06274509803922,7.76039215686274,7.45803921568627,7.1556862745098])
for v_index,coefficients in enumerate(coefficients_all):
	x = e_energy/coefficients
	cross_section = ((coefficients/coefficients_all[0])**1.15) * 1.828/x * ((1-1/(x**0.92))**2.19) * np.log(2.05*e_energy) * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section[cross_section<0] = 0
	reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
	if v_index==0:
		e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
e_H2N13Λσ__e_H2pX2Σg_e = np.sum(e_H2N13Λσ__e_H2pX2Σg_e,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2N13Λσ__e_H2pX2Σg_e ,vmin=max(np.max(e_H2N13Λσ__e_H2pX2Σg_e)*1e-12,np.min(e_H2N13Λσ__e_H2pX2Σg_e[e_H2N13Λσ__e_H2pX2Σg_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e, v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate
# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; eps’) +e → H+ +H(1s) + 2e
# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(B2Σu+; eps’) +e → H+ +H(1s) + 2e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 4.3.1
# I neglect what happens for electronic excited states because they will anyway be less populated
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
# electronic partition function approximated by g0=1
# I think I need to consider only the electron energy and not H2
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_H2N13Λσ__Hp_H_2e = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# coefficients: C1,..,C5
# valid 3000 to 200000K
coefficients_all = np.array([[-2.1196e2,1.0022,-20.35,-4.5201,1.0773e-2],[-2.0518e2,0.99226,-19.905,-3.3364,1.1725e-2],[-1.9936e2,0.98837,-19.6,-3.0891,1.2838e-2],[-1.9398e2,0.98421,-19.457,-3.1386,1.3756e-2],[-1.8893e2,0.97647,-19.397,-3.2807,1.4833e-2],[-1.8422e2,0.96189,-19.31,-3.2609,1.6030e-2],[-1.7903e2,0.94593,-19.17,-3.0592,1.7254e-2],[-1.7364e2,0.93986,-19.052,-2.988,1.8505e-2],[-1.6960e2,0.93507,-18.908,-2.7334,1.8810e-2],[-1.6664e2,0.92602,-18.723,-2.2024,1.8055e-2],[-1.6521e2,0.92124,-18.549,-1.6895,1.6245e-2],[-1.6569e2,0.93366,-18.479,-1.6311,1.5194e-2],[-1.6464e2,0.94682,-18.44,-1.7259,1.5304e-2],[-1.6071e2,0.95533,-18.405,-1.8938,1.6254e-2]])
for v_index,coefficients in enumerate(coefficients_all):
	T_scaled = T_e_temp/1000
	reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2] + coefficients[3]*np.exp(-coefficients[4]*T_scaled)) 		# m^3 / s
	reaction_rate_min = reaction_rate.flatten()[(np.abs(T_e_temp-3000)).argmin()]
	reaction_rate_max = reaction_rate.flatten()[(np.abs(T_e_temp-200000)).argmin()]
	reaction_rate[T_e_temp<3000]=reaction_rate_min
	reaction_rate[T_e_temp>200000]=reaction_rate_max
	if v_index==0:
		e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
e_H2N13Λσ__Hp_H_2e = np.sum(e_H2N13Λσ__Hp_H_2e,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2N13Λσ__Hp_H_2e ,vmin=max(np.max(e_H2N13Λσ__Hp_H_2e)*1e-12,np.min(e_H2N13Λσ__Hp_H_2e[e_H2N13Λσ__Hp_H_2e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e, v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate:
# e +H2(X1Σ+g ; v) →H-2 (X2Σ+u ) → H- +H(1s)
# e +H2(X1Σ+g ; v) →H−2 (B2Σ+g ) → H− +H(1s)
if True:
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 4.4.1
	# I neglect what happens for different intermediate excited states
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	# electronic partition function approximated by g0=1
	# I think I need to consider only the electron energy and not H2
	T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
	T_e_temp[T_e_temp==0]=300	# ambient temperature
	e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
	e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
	e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
	e_H2X1Σg__Hm_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	# coefficients: E th,v (eV ), σ v (10 -16 cm 2 )
	coefficients_all = np.array([[3.72,3.22e-5],[3.21,5.18e-4],[2.72,4.16e-3],[2.26,2.20e-2],[1.83,1.22e-1],[1.43,4.53e-1],[1.36,1.51],[0.713,4.48],[0.397,10.1],[0.113,13.9],[-0.139,11.8],[-0.354,8.87],[-0.529,7.11],[-0.659,5],[-0.736,3.35]])
	for v_index,coefficients in enumerate(coefficients_all):
		cross_section =  coefficients[1]*np.exp(-(e_energy-np.abs(coefficients[0]))/0.45) * 1e-16 * 1e-4	# m^2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		if v_index==0:
			e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
else:
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 4.4.1
	# I neglect what happens for different intermediate excited states
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	# electronic partition function approximated by g0=1
	# I think I need to consider only the electron energy and not H2
	T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
	T_e_temp[T_e_temp==0]=300	# ambient temperature
	e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
	e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
	e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
	e_H2X1Σg__Hm_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	# coefficients: E th,v (eV ), σ v (10 -16 cm 2 )
	coefficients_all = np.array([[3.72,3.22e-5],[3.21,5.18e-4],[2.72,4.16e-3],[2.26,2.20e-2],[1.83,1.22e-1],[1.43,4.53e-1],[1.36,1.51],[0.713,4.48],[0.397,10.1],[0.113,13.9],[-0.139,11.8],[-0.354,8.87],[-0.529,7.11],[-0.659,5],[-0.736,3.35]])
	# data from https://www-amdis.iaea.org/ALADDIN/
	# this is for v>3
	interpolation_ev = np.array([1.0000E-01,1.1775E-01,1.3865E-01,1.6326E-01,1.9224E-01,2.2636E-01,2.6655E-01,3.1386E-01,3.6957E-01,4.3517E-01,5.1241E-01,6.0336E-01,7.1046E-01,8.3657E-01,9.8506E-01,1.1599E+00,1.3658E+00,1.6082E+00,1.8937E+00,2.2298E+00,2.6256E+00,3.0917E+00,3.6405E+00,4.2866E+00,5.0475E+00,5.9435E+00,6.9985E+00,8.2407E+00,9.7035E+00,1.1426E+01,1.3454E+01,1.5842E+01,1.8654E+01,2.1965E+01,2.5864E+01,3.0455E+01,3.5861E+01,4.2226E+01,4.9721E+01,5.8547E+01,6.8939E+01,8.1176E+01,9.5585E+01,1.1255E+02,1.3253E+02,1.5605E+02,1.8375E+02,2.1637E+02,2.5478E+02,3.0000E+02])
	interpolation_reaction_rate = np.array([1.5800E-08,1.8056E-08,2.0202E-08,2.2169E-08,2.3891E-08,2.5299E-08,2.6336E-08,2.6954E-08,2.7125E-08,2.6839E-08,2.6113E-08,2.4986E-08,2.3516E-08,2.1779E-08,1.9857E-08,1.7834E-08,1.5789E-08,1.3790E-08,1.1893E-08,1.0137E-08,8.5487E-09,7.1395E-09,5.9110E-09,4.8562E-09,3.9627E-09,3.2146E-09,2.5945E-09,2.0848E-09,1.6691E-09,1.3321E-09,1.0602E-09,8.4191E-10,6.6722E-10,5.2786E-10,4.1696E-10,3.2890E-10,2.5912E-10,2.0391E-10,1.6031E-10,1.2592E-10,9.8839E-11,7.7541E-11,6.0811E-11,4.7682E-11,3.7388E-11,2.9318E-11,2.2991E-11,1.8026E-11,1.4124E-11,1.1050E-11]) * 1e-6		# m^3/s
	interpolator_reaction_rate = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_reaction_rate),fill_value='extrapolate')
	for v_index in range(5):
		if v_index<=3:
			cross_section =  coefficients_all[v_index][1]*np.exp(-(e_energy-np.abs(coefficients_all[v_index][0]))/0.45) * 1e-16 * 1e-4	# m^2
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		else:
			reaction_rate = np.zeros_like(merge_Te_prof_multipulse_interp_crop_limited)
			reaction_rate[merge_Te_prof_multipulse_interp_crop_limited>0] = np.exp(interpolator_reaction_rate(np.log(merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited>0])))
			reaction_rate[reaction_rate<0] = 0
		if v_index==0:
			e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
e_H2X1Σg__Hm_H1s = np.sum(e_H2X1Σg__Hm_H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2X1Σg__Hm_H1s , cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2(X1Σ+g ; v) → H- +H(1s), v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate:
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 4.5.1
# e +H2(X1Σ+g;v) →H2-(X2Σ+u) → e +H2(X1Σ+;eps)→ e +H(1s) +H(1s)
# e +H2(X1Σ+g;v) →H2-(B2Σ+g) → e +H2(b3Σ+;eps)→ e +H(1s) +H(1s)
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
# electronic partition function approximated by g0=1
# I think I need to consider only the electron energy and not H2
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_H2X1Σg__e_H1s_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# via H2- ground state
# coefficients: a1,..,a6
coefficients_all = np.array([[-50.862,0.92494,-28.102,-4.5231e-2,0.46439,0.8795],[-48.125,0.91626,-24.873,-4.9898e-2,0.45288,0.87604],[-41.218,0.96738,-23.167,-4.8546e-2,-1.7222,0.19858],[-37.185,0.96391,-21.264,-5.1701e-2,-1.8121,0.19281],[-35.397,0.85294,-18.452,-6.522e-2,-0.56595,8.8997e-2],[-33.861,0.9301,-20.852,-3.016e-2,5.561,0.45548],[-23.751,0.9402,-19.626,-3.276e-2,-0.3982,1.58655],[-19.988,0.83369,-18.7,-3.552e-2,-0.38065,1.74205],[-18.278,0.8204,-17.754,-4.453e-2,-0.10045,2.5025],[-13.589,0.7021,-16.85,-5.012e-2,-0.77502,0.3423],[-11.504,0.84513,-14.603,-6.775e2,-3.2615,0.13666]])
for v_index,coefficients in enumerate(coefficients_all):
	T_scaled = T_e_temp/1000
	reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) 		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	if v_index==0:
		e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
# via H2- excited state
# coefficients: a1,..,a6
coefficients_all = np.array([[-15.76,-0.052659,-84.679,1.0414,-8.2933,0.18756],[-16.162,-0.0498200333333333,-74.3906666666667,1.01586,-6.15236666666667,0.227996666666667],[-16.564,-0.0469810666666667,-64.1023333333333,0.99032,-4.01143333333333,0.268433333333333],[-16.966,-0.0441421,-53.814,0.96478,-1.8705,0.30887],[-16.1206666666667,-0.0490894,-47.1276666666667,0.94422,-1.72766666666667,0.208215033333333],[-15.2753333333333,-0.0540367,-40.4413333333333,0.92366,-1.58483333333333,0.107560066666667],[-14.43,-0.058984,-33.755,0.9031,-1.442,0.0069051],[-14.4276666666667,-0.0575976666666667,-28.0646666666667,0.897233333333333,-1.5259,0.00691206666666667],[-14.4253333333333,-0.0562113333333333,-22.3743333333333,0.891366666666667,-1.6098,0.00691903333333333],[-14.423,-0.054825,-16.684,0.8855,-1.6937,0.006926],[-16.2556666666667,-0.0396174,-26.4876666666667,0.799833333333333,13.6192,0.0993073333333334],[-18.0883333333333,-0.0244098,-36.2913333333333,0.714166666666667,28.9321,0.191688666666667],[-19.921,-0.0092022,-46.095,0.6285,44.245,0.28407]])
for v_index,coefficients in enumerate(coefficients_all):
	T_scaled = T_e_temp/1000
	reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) 		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	if v_index==0:
		e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
e_H2X1Σg__e_H1s_H1s = np.sum(e_H2X1Σg__e_H1s_H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2X1Σg__e_H1s_H1s  ,vmin=max(np.max(e_H2X1Σg__e_H1s_H1s)*1e-12,np.min(e_H2X1Σg__e_H1s_H1s[e_H2X1Σg__e_H1s_H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2(X1Σ+g;v) →H2-(X2Σ+u)→e +H2(X1Σ+;eps)→ e +H(1s) +H(1s), v=0-12\ne +H2(X1Σ+g;v) →H2-(B2Σ+g) → e +H2(b3Σ+;eps)→ e +H(1s) +H(1s) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H+ +H2(X1Σ+g ; v) →H(1s) +H2+(X2Σ+ g ; v)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 5.2.1a
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
Hp_H2X1Σg__H1s_H2pX2Σg = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# v=0-8
# coefficients: E 0v(eV),a 1,...,a 6,b 1,...,b 11
coefficients_all = np.array([[2.67,18.6,-1.66,2.53,1.93,0,0,17.3,105,2,1.0e4,-1.12,3.64e-4,0.9,5.03e-19,4,5.87e-28,5.5],[1.74,2.51,-0.56,4.21,4.07,1.0e-5,1,58,11.28,0.246,1,0,3.92e-5,1.11,4.95e-17,3.65,3.88e-26,5.2],[1.17,3.01,-0.63,7.04,10.74,1.0e-5,1,26.53,25.2,0.65,1,0,1.56e-6,1.45,5.50e-19,4,8.50e-27,5.3],[0.48,4.5,-0.57,5,14.62,1.0e-5,1,39.5,9.35,0.51,1,0,5.32e-7,1.6,3.52e-20,4.25,3.50e-27,5.4],[0,24,0.32,0,0,0.145,1.84,10.8,0,0,1,-0.297,2.92e-4,0.76,4.93e-11,2.35,2.62e-27,5.5],[0,11.75,0.092,0,0,3.86e-3,2.86,20,0,0,1,-0.193,1.36e-5,1.15,4.46e-12,2.61,4.31e-27,5.5],[0,11.58,0.091,0,0,3.84e-3,2.87,20.04,0,0,1,-0.192,1.34e-5,1.15,4.46e-12,2.61,4.31e-27,5.5],[0,0,0,0,0,0,0,33,0,0,1,-0.022,1.22e-2,0.36,6.51e-8,1.78,3.25e-23,4.86],[0,0,0,0,0,0,0,30,0,0,1,-0.017,1.87e-2,0.375,9.0e-10,2.18,1.85e-25,5.25]])
for v_index,coefficients in enumerate(coefficients_all):
	cross_section1 =  coefficients[1]*(Hp_energy*coefficients[2])*((1-((coefficients[0]/Hp_energy)**coefficients[3]))**coefficients[4])*np.exp(-coefficients[5]*(Hp_energy**coefficients[6])) * 1e-16 * 1e-4	# m^2
	cross_section1[cross_section1<0] = 0
	cross_section2 =  coefficients[7]*np.exp(-coefficients[8]/(Hp_energy**coefficients[9])) / (coefficients[10]*(Hp_energy**coefficients[11]) + coefficients[12]*(Hp_energy**coefficients[13]) + coefficients[14]*(Hp_energy**coefficients[15]) + coefficients[16]*(Hp_energy**coefficients[17]) ) * 1e-16 * 1e-4	# m^2
	cross_section2[cross_section2<0] = 0
	cross_section=cross_section1+cross_section2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
	if v_index==0:
		Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
# v=9-14
for v_index in range(9,15):
	if v_index==9:
		f_v =1
	else:
		f_v = 1.97/(v_index - 8)**1.23
	cross_section =  27*f_v/((Hp_energy**0.033) + 9.85e-10 * (Hp_energy**2.16) + 1.66*f_v*1e-25 * (Hp_energy**5.25)) * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section[cross_section<0] = 0
	reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
	if v_index==0:
		Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
Hp_H2X1Σg__H1s_H2pX2Σg = np.sum(Hp_H2X1Σg__H1s_H2pX2Σg,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_H2X1Σg__H1s_H2pX2Σg ,vmin=max(np.max(Hp_H2X1Σg__H1s_H2pX2Σg)*1e-12,np.min(Hp_H2X1Σg__H1s_H2pX2Σg[Hp_H2X1Σg__H1s_H2pX2Σg>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H2(X1Σ+g ; v) →H(1s) +H2+(X2Σ+ g ; v), v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H+ +H2(X1Σ+ g ; v) → H+ +H(1s) +H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 5.3
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
Hp_H2X1Σg__Hp_H1s_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# v=0-8
# coefficients: E 0v (eV ),a 1,...,a 4
# valid up to 20eV
coefficients_all = np.array([[6.717,7.52e3,4.64,5.37,2.18],[5.943,1.56e3,3.91,3.42,1.55],[5.313,3.83e2,3.22,2.71,1.5],[4.526,72.5,2.4,2.32,1.34],[3.881,23.8,1.64,1.86,1.5],[3.278,8.68,0.94,1.39,1.04],[2.716,6.85,0.58,1.2,1.14],[2.2215,8.695,0.47,1.115,1.195],[1.727,10.54,0.36,1.03,1.25],[1.3225,16.7,0.32,0.88,1.515],[0.918,22.86,0.28,0.73,1.78],[0.627,26.485,0.24,0.69,1.71],[0.336,30.11,0.2,0.65,1.64],[0.18085,32.015,0.175,0.615,1.86],[0.0257,33.92,0.15,0.58,2.08]])
for v_index,coefficients in enumerate(coefficients_all):
	cross_section =  (coefficients[1]/(Hp_energy**coefficients[2]))*((1-(1.5*coefficients[0]/Hp_energy)**coefficients[3])**coefficients[4]) * 1e-16 * 1e-4	# m^2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section[cross_section<0] = 0
	reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
	reaction_rate_max = reaction_rate.flatten()[(np.abs(T_Hp*eV_to_K-20)).argmin()]
	reaction_rate[T_Hp*eV_to_K>20]=reaction_rate_max
	if v_index==0:
		Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
Hp_H2X1Σg__Hp_H1s_H1s = np.sum(Hp_H2X1Σg__Hp_H1s_H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_H2X1Σg__Hp_H1s_H1s ,vmin=max(np.max(Hp_H2X1Σg__Hp_H1s_H1s)*1e-12,np.min(Hp_H2X1Σg__Hp_H1s_H1s[Hp_H2X1Σg__Hp_H1s_H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H2(X1Σ+ g ; v) → H+ +H(1s) +H(1s), v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H- +H2(v) →H +H2(v") + e
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm),200))).T*(2*boltzmann_constant_J*T_Hm.T/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm.T))**(3/2) , (T_Hm.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
if False:	# NOTE: I'm not using this because the this formulas return something (cros section >0) only for high energies, something I don't have in my cases
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 6.1.1
	# v=0, only available data
	Hm_energy_scaled = Hm_energy/1000
	cross_section1 =  81.1/((Hm_energy_scaled**0.63)*(1+2.03e-6 * Hm_energy_scaled**1.95)) * ( 1-np.exp(-0.057*(np.nanmax([(Hm_energy_scaled/2.18 -1)**0.94,np.zeros_like(Hm_energy_scaled)],axis=0)) )) * 1e-16 * 1e-4	# m^2
	cross_section1[cross_section1<0] = 0
	cross_section2 =  1.22e3 /((Hm_energy_scaled**0.5)*(1+6.91e-4 * Hm_energy_scaled**0.4)) * np.exp(-125/(Hm_energy_scaled**0.663)) * 1e-16 * 1e-4	# m^2
	cross_section2[cross_section2<0] = 0
	cross_section=cross_section1+cross_section2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	reaction_rate = np.sum(cross_section*Hm_velocity*Hm_velocity_PDF,axis=-1)* np.mean(np.diff(Hm_velocity))		# m^3 / s
	v_index=0
	Hm_H2v__H_H2v_e = reaction_rate *nHm_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
else:
	# data from https://www-amdis.iaea.org/ALADDIN/
	# no info on the vibrational state, I assume it is valid for all
	Hm_H2v__H_H2v_e = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	interpolation_ev = np.array([2.3183E+00,3.2014E+00,4.4210E+00,6.1052E+00,8.4309E+00,1.1643E+01,1.6078E+01,2.2203E+01,3.0661E+01,4.2341E+01,5.8471E+01,8.0746E+01,1.1151E+02,1.5398E+02,2.1264E+02,2.9365E+02,4.0552E+02,5.6000E+02,7.7333E+02,1.0679E+03,1.4748E+03,2.0366E+03,2.8124E+03,3.8838E+03,5.3633E+03,7.4065E+03,1.0228E+04,1.4124E+04,1.9505E+04,2.6935E+04,3.7197E+04,5.1367E+04,7.0935E+04,9.7957E+04,1.3527E+05,1.8681E+05,2.5797E+05,3.5624E+05,4.9196E+05,6.7937E+05,9.3817E+05,1.2956E+06,1.7891E+06,2.4707E+06,3.4119E+06,4.7116E+06,6.5065E+06,8.9852E+06,1.2408E+07,1.7135E+07])	# eV
	interpolation_cross_section = np.array([9.5063E-17,1.5011E-16,2.1010E-16,2.6792E-16,3.1859E-16,3.6021E-16,3.9345E-16,4.2053E-16,4.4425E-16,4.6737E-16,4.9230E-16,5.2099E-16,5.5498E-16,5.9540E-16,6.4289E-16,6.9763E-16,7.5912E-16,8.2608E-16,8.9629E-16,9.6648E-16,1.0324E-15,1.0892E-15,1.1314E-15,1.1542E-15,1.1537E-15,1.1280E-15,1.0773E-15,1.0041E-15,9.1306E-16,8.1021E-16,7.0205E-16,5.9470E-16,4.9323E-16,4.0127E-16,3.2091E-16,2.5285E-16,1.9672E-16,1.5146E-16,1.1562E-16,8.7634E-17,6.6012E-17,4.9419E-17,3.6733E-17,2.7048E-17,1.9657E-17,1.4021E-17,9.7429E-18,6.5296E-18,4.1680E-18,2.4949E-18]) * 1e-4		# m^2
	interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
	for v_index in range(15):
		cross_section=np.exp(interpolator_cross_section(np.log(Hm_energy)))
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*Hm_velocity*Hm_velocity_PDF,axis=-1)* np.mean(np.diff(Hm_velocity))		# m^3 / s
		if v_index==0:
			Hm_H2v__H_H2v_e[v_index] += reaction_rate *nHm_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			Hm_H2v__H_H2v_e[v_index] += reaction_rate *nHm_ne_all* merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	Hm_H2v__H_H2v_e = np.sum(Hm_H2v__H_H2v_e,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, Hm_H2v__H_H2v_e ,vmin=max(np.max(Hm_H2v__H_H2v_e)*1e-12,np.min(Hm_H2v__H_H2v_e[Hm_H2v__H_H2v_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H- +H2(v) →H +H2(v") + e, v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H(1s) +H2(v) → H(1s) + 2H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 6.2.2
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
H1s_H2v__H1s_2H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# coefficients: a1,..,a5
coefficients_all = np.array([[2.06964e1,7.32149e7,1.7466,4.75874e3,-9.42775e-1],[2.05788e1,4.32679e7,1.6852,1.91812e3,-8.16838e-1],[2.05183e1,5.15169e7,1.73345,3.09006e3,-8.88414e-1],[2.04460e1,1.87116e8,1.87951,9.04442e3,-9.78327e-1],[2.03608e1,4.93688e8,1.99499,2.32656e4,-1.06294],[2.02426e1,1.80194e8,1.92249,1.28777e4,-1.02713],[2.00161e1,2.96945e5,1.31044,9.55214e2,-1.07546],[1.98954e1,4.53104e5,1.37055,3.88065e2,-8.71521e-1],[1.97543e1,5.13174e5,1.39819,3.54272e2,-8.07563e-1],[1.97464e1,9.47230e4,1.24048,2.28283e2,-8.51591e-1],[1.95900e1,6.43990e4,1.22211,1.16196e2,-7.35645e-1],[1.94937e1,3.49017e4,1.20883,1.26329e2,-8.15130e-1],[1.90708e1,1.05971e5,9.91646e-1,1.05518e2,-1.93837e-1],[1.89718e1,7.76046e5,7.84577e-1,1.31409e3,-1.00479e-2],[1.87530e1,5.81508e5,7.35904e-1,1.69328e3,4.47757e-3]])
for v_index,coefficients in enumerate(coefficients_all):
	reaction_rate = np.exp(-coefficients[0] -coefficients[1]/(T_H**coefficients[2]*(1+coefficients[3]*(T_H**coefficients[4])) )) *1e-6 		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	if v_index==0:
		H1s_H2v__H1s_2H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0)) * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		H1s_H2v__H1s_2H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0)) * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
H1s_H2v__H1s_2H1s = np.sum(H1s_H2v__H1s_2H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, H1s_H2v__H1s_2H1s,vmin=max(np.max(H1s_H2v__H1s_2H1s)*1e-12,np.min(H1s_H2v__H1s_2H1s[H1s_H2v__H1s_2H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H(1s) +H2(v) → H(1s) + 2H(1s), v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H2(v1 = 0) +H2(v) →H2(v1 = 0) + 2H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 6.3.2
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
# valid up to 20000K
# I'm not sure which temperature to use here. I will use T_H2
H2v0_H2v__H2v0_2H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
for v_index,coefficients in enumerate(coefficients_all):
	reaction_rate_0 = 1.3*(1 +0.023*v_index + 1.93e-5 * (v_index**4.75) + 2.85e-24 *(v_index**21.6)  )
	T_0 = (7.47 - 0.322 *v_index) * 1e3
	reaction_rate = reaction_rate_0 * np.exp(-T_0/T_H2) * 1e-10 *1e-6 		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H2-20000)).argmin()]
	reaction_rate[T_H2>20000]=reaction_rate_max
	if v_index==0:
		H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
H2v0_H2v__H2v0_2H1s = np.sum(H2v0_H2v__H2v0_2H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, H2v0_H2v__H2v0_2H1s,vmin=max(np.max(H2v0_H2v__H2v0_2H1s)*1e-12,np.min(H2v0_H2v__H2v0_2H1s[H2v0_H2v__H2v0_2H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H2(v1 = 0) +H2(v) →H2(v1 = 0) + 2H(1s), v=0-14 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction reates:
# e +H2+(v) → e +H2+(2pσu) → e +H+ +H(1s)
# e +H2+(v) → e +H2+(2pπu) → e +H+ +H(n = 2)
# e +H2+(v) →H2∗∗[(2pσu)2] → e +H+ +H(1s)
# e +H2+(v) →H2∗Ryd(N1,3Λσ; eps) → e +H+ +H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.1.2a
# valid from 0.01 to 2000 eV
# I'm not sure if only electron energy or also H2+
# I'll assume it is electron energy because usually it is the highest
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
cross_section = 13.2*np.log(np.exp(1) + 2.55e-4 * e_energy)/( (e_energy**0.31)*(1+0.017*(e_energy**0.76)))* 1e-16 * 1e-4
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-0.01)).argmin()]
cross_section[e_energy<0.01]=cross_section_min
cross_section[cross_section<0] = 0
reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
e_H2p__XXX__e_Hp_H1s = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2p__XXX__e_Hp_H1s,vmin=max(np.max(e_H2p__XXX__e_Hp_H1s)*1e-12,np.min(e_H2p__XXX__e_Hp_H1s[e_H2p__XXX__e_Hp_H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2+(v) → e +H2+(2pσu) → e +H+ +H(1s) \n e +H2+(v) → e +H2+(2pπu) → e +H+ +H(n = 2) \n e +H2+(v) →H2∗∗[(2pσu)2] → e +H+ +H(1s) \n e +H2+(v) →H2∗Ryd(N1,3Λσ; eps) → e +H+ +H(1s) \nreaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate e +H2+(v) → {H2∗∗ (eps);H2∗Ryd(N1,3Λσ; eps)} →H(1s) +H(n ≥ 2)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.1.3a
# valid from 0.01 to 10 eV
# I'm not sure if only electron energy or also H2+
# I'll assume it is electron energy because usually it is the highest
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
cross_section = 17.3* ( 1/((e_energy**0.665)*(1+1.1*(e_energy**0.512) + 0.011*(e_energy**3.10)) ) + 0.133* np.exp(-0.35*((e_energy - 6.05)**2)) )* 1e-16 * 1e-4		# m^2
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_min = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-0.01)).argmin()]
cross_section_max = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-10)).argmin()]
cross_section[e_energy<0.01]=cross_section_min
cross_section[e_energy>10]=cross_section_max
cross_section[cross_section<0] = 0
reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
e_H2p__H1s_Hn2 = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2p__H1s_Hn2,vmin=max(np.max(e_H2p__H1s_Hn2)*1e-12,np.min(e_H2p__H1s_Hn2[e_H2p__H1s_Hn2>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2+(v) → {H2∗∗ (eps);H2∗Ryd(N1,3Λσ; eps)} →H(1s) +H(n ≥ 2) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate e +H2+(v) → e +H+ +H+ + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.1.4
# I'm not sure if only electron energy or also H2+
# I'll assume it is electron energy because usually it is the highest
T_e_temp = merge_Te_prof_multipulse_interp_crop_limited/eV_to_K	# K
T_e_temp[T_e_temp==0]=300	# ambient temperature
e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
cross_section = 7.39/e_energy * np.log(0.18*e_energy) * (1-np.exp(-0.105*((e_energy/15.2 -1)**1.55))) * 1e-16 * 1e-4		# m^2
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section[cross_section<0] = 0
reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
e_H2p__e_Hp_Hp_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, e_H2p__e_Hp_Hp_e,vmin=max(np.max(e_H2p__e_Hp_Hp_e)*1e-12,np.min(e_H2p__e_Hp_Hp_e[e_H2p__e_Hp_Hp_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('e +H2+(v) → e +H+ +H+ + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rates H(1s) +H+2 (v) → H+ +H2(v?)
# PROBLEM I can't find the energy levels of H2+, together with multeplicity.
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.2.2a
# Therefore I cannot get the population density, and as a consequence the reaction rate.

# I can get the vibrationally resolved cross section from
# https://www-amdis.iaea.org/ALADDIN/
# but only for very high energyes, (>200eV), so I cannot use them.


# reaction rate
# H(1s) +H2+(v) → (H3+)∗ →H +H+ +H
# H(1s) +H2+(v) → [H2+(2pσu/2pπu···)] → H +H+ +H
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.2.3
# NO: I tae it from https://www-amdis.iaea.org/ALADDIN/
# H2+ [1sσg;v=0-9], H [G] → H+, H [G], H [G]
interpolation_ev1 = np.array([1.0000E-01,1.3216E-01,1.7467E-01,2.3085E-01,3.0509E-01,4.0321E-01,5.3289E-01,7.0428E-01,9.3080E-01,1.2302E+00,1.6258E+00,2.1487E+00,2.8398E+00,3.7531E+00,4.9602E+00,6.5555E+00,8.6638E+00,1.1450E+01,1.5133E+01,2.0000E+01])
interpolation_ev2 = np.array([0.1,0.4,1,1.5,5,20])
interpolation_reaction_rate = np.array([[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,7.8250E+00,6.3511E+00,3.6876E+00,1.4091E+00,1.6587E+00,1.5533E-78,1.7749E-61,2.4111E-48,2.5606E-38,9.9662E-31],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,2.8825E+00,1.0492E+00,1.2838E+00,3.2525E+00,5.6641E+00,1.3117E-93,4.8863E-75,2.3583E-60,8.3672E-49,9.1706E-40,1.0101E-32,2.9288E-27,4.8779E-23],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,1.1780E+00,2.5938E+00,2.3298E+00,3.5205E+00,2.0763E+00,6.7252E+00,1.0927E-84,5.6088E-70,4.1677E-58,1.5679E-48,8.3129E-41,1.4278E-34,1.5571E-29,1.8523E-25,3.7074E-22],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,3.2127E+00,9.4328E+00,2.8409E+00,3.1842E+00,2.7473E+00,2.3172E+00,1.5918E-88,5.1672E-74,3.3761E-62,1.4551E-52,1.0872E-44,3.0771E-38,6.1862E-33,1.4606E-28,6.0452E-25,6.0274E-22],[1.0630E+00,5.8307E+00,2.4311E+00,3.0605E+00,2.6228E+00,2.1022E+00,1.4070E+00,4.8355E-86,3.8061E-74,2.3267E-64,2.9679E-56,1.7448E-49,8.8682E-44,6.3858E-39,9.5609E-35,3.9963E-31,5.8340E-28,3.5211E-25,9.9705E-23,1.4563E-20],[1.9214E+00,1.0314E-86,3.5242E-74,3.3516E-64,3.0146E-56,7.0056E-50,9.5280E-45,1.4641E-40,4.2859E-37,3.5980E-34,1.1866E-31,1.9489E-29,1.8981E-27,1.2415E-25,5.9392E-24,2.1959E-22,6.4862E-21,1.5570E-19,3.0578E-18,4.9127E-17]]) * 1e-6		# m^3/s
interpolator_reaction_rate = interpolate.interp2d(np.log(interpolation_ev1),np.log(interpolation_ev2), np.log(interpolation_reaction_rate),fill_value='extrapolate',copy=False)
selected = np.logical_and(T_H>0,T_H2p>0)
reaction_rate = np.zeros_like(merge_Te_prof_multipulse_interp_crop_limited)
reaction_rate[selected] = np.exp(np.diag(interpolator_reaction_rate(np.log(T_H2p[selected]*eV_to_K),np.log(T_H[selected]*eV_to_K))))
H1s_H2pv__Hp_H_H = (merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0)) *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H1s_H2pv__Hp_H_H,vmin=max(np.max(H1s_H2pv__Hp_H_H)*1e-12,np.min(H1s_H2pv__Hp_H_H[H1s_H2pv__Hp_H_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H2+ [1sσg;v=0-9], H [G] → H+, H [G], H [G] reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate
# H2+(vi) +H2(v0) → (H3+∗ +H) →H+ +H +H2(v01)
# H2+(vi) +H2(v0) → [H2+(2pσu/2pπu···) +H2] →H+ +H +H2(v01)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.3.2
# valid up to 30keV
H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
cross_section1 = 4.05/(H2p_energy**0.653) * np.exp(-3.15/((H2p_energy-2)**1.65)) * 1e-16 * 1e-4		# m^2
cross_section1[cross_section1<0] = 0
cross_section2 = 0.139*(H2p_energy**0.318) * np.exp(-680/(H2p_energy**2.1))/(1+2.75e-12*(H2p_energy**2.65)+9.04e-23*(H2p_energy*4.65)) * 1e-16 * 1e-4		# m^2
cross_section2[cross_section2<0] = 0
cross_section = cross_section1 + cross_section2
cross_section[np.logical_not(np.isfinite(cross_section))]=0
cross_section_max = cross_section.flatten()[(np.abs(H2p_energy-30000)).argmin()]
cross_section[H2p_energy>30000]=cross_section_max
reaction_rate = np.sum(cross_section*H2p_velocity*H2p_velocity_PDF,axis=-1)* np.mean(np.diff(H2p_velocity))		# m^3 / s
H2pvi_H2v0__Hp_H_H2v1 = nH2_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H2pvi_H2v0__Hp_H_H2v1,vmin=max(np.max(H2pvi_H2v0__Hp_H_H2v1)*1e-12,np.min(H2pvi_H2v0__Hp_H_H2v1[H2pvi_H2v0__Hp_H_H2v1>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H2+(vi) +H2(v0) → (H3+∗ +H) →H+ +H +H2(v01) \n H2+(vi) +H2(v0) → [H2+(2pσu/2pπu···) +H2] →H+ +H +H2(v01) \n reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H2+(vi) +H2(v0) →H3+(v3) +H(1s)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.3.3
plank_constant_eV = 4.135667696e-15	# eV s
plank_constant_J = 6.62607015e-34	# J s
light_speed = 299792458	# m/s
oscillator_frequency = 4161.166 * 1e2	# 1/m
q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
H2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2.T),200))).T*(2*boltzmann_constant_J*T_H2/(2*hydrogen_mass))**0.5).T
H2_velocity_PDF = (4*np.pi*(H2_velocity.T)**2 * gauss( H2_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2))**(3/2) , (T_H2*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H2_energy = 0.5 * H2_velocity**2 * (2*hydrogen_mass) * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + H2_energy).T
# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
form_factor = 17.76/( (baricenter_impact_energy**0.477)*(1+0.0291*(baricenter_impact_energy**3.61)+1.53e-5 * (baricenter_impact_energy**6.55) )  ) * 1e-16 *1e-4	# m^2
interpolation_v = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
interpolation_CMenergy = np.array([0.0000000000001,0.04,0.25,0.5,0.75,1,2,3,4,5,8,10,15,40,60,100,200])	# eV
interpolation_f = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],[0.932380952380952,0.94,0.98,1,1.01,1.075,1.27,1.46,1.23,1.17,1.13,1.01,0.97,0.77,0.61,0.289999999999999,0],[0.875714285714286,0.89,0.965,0.995,1.02,1.1,1.4,1.59,1.245,1.14,1.06,0.93,0.89,0.69,0.53,0.209999999999999,0],[0.848809523809524,0.865,0.95,0.99,1.01,1.11,1.49,1.57,1.175,1.035,0.93,0.82,0.78,0.58,0.42,0.0999999999999994,0],[0.821904761904762,0.84,0.935,0.985,0.99,1.115,1.47,1.46,1.075,0.925,0.81,0.72,0.68,0.48,0.320000000000001,0,0],[0.805952380952381,0.825,0.925,0.98,0.97,1.11,1.44,1.33,0.99,0.835,0.72,0.62,0.59,0.44,0.32,0.0799999999999995,0],[0.784047619047619,0.805,0.915,0.975,0.94,1.1,1.39,1.21,0.9,0.74,0.63,0.54,0.5,0.3,0.14,0,0],[0.768095238095238,0.79,0.905,0.97,0.91,1.09,1.34,1.09,0.82,0.67,0.55,0.47,0.43,0.23,0.0699999999999997,0,0],[0.746190476190476,0.77,0.895,0.965,0.88,1.08,1.29,0.99,0.74,0.6,0.48,0.41,0.37,0.17,0.00999999999999968,0,0],[0.736190476190476,0.76,0.885,0.96,0.85,1.07,1.24,0.89,0.67,0.53,0.415,0.36,0.31,0.0600000000000001,0,0,0],[0.726190476190476,0.75,0.875,0.955,0.82,1.06,1.19,0.79,0.61,0.46,0.35,0.31,0.26,0.0100000000000001,0,0,0],[0.686920535714286,0.714666642,0.8603337,0.95,0.810866639,1.05,1.145673,0.6643,0.5157149,0.373573,0.2592865,0.2219,0.17003,0,0,0,0],[0.665533804761905,0.694878764,0.8489398,0.945,0.787551488,1.04,1.097816,0.55323,0.4373221,0.296966,0.182858,0.1548,0.10039,0,0,0,0],[0.644147073809524,0.675090886,0.8375459,0.94,0.764236337,1.03,1.049959,0.44216,0.3589293,0.220359,0.1064295,0.0876999999999999,0.0307499999999999,0,0,0,0],[0.622760342857143,0.655303008,0.826152,0.935,0.740921186,1.02,1.002102,0.33109,0.2805365,0.143752,0.0300010000000002,0.0205999999999998,0,0,0,0,0],[0.601373611904762,0.63551513,0.8147581,0.93,0.717606035,1.01,0.954245,0.22002,0.2021437,0.0671450000000002,0,0,0,0,0,0,0],[0.579986880952381,0.615727252,0.8033642,0.925,0.694290884,1,0.906388,0.10895,0.1237509,0,0,0,0,0,0,0,0],[0.55860015,0.595939374,0.7919703,0.92,0.670975733,0.99,0.858531,0,0.0453580999999998,0,0,0,0,0,0,0,0],[0.537213419047619,0.576151496,0.7805764,0.915,0.647660582,0.98,0.810674,0,0,0,0,0,0,0,0,0,0]])	# au
interpolator_f = interpolate.interp2d(np.log(interpolation_CMenergy),interpolation_v, interpolation_f,fill_value=0)
H2pvi_H2v0__H3p_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
# Population factors F 0v 0 of H 2 + (v 0 ) levels by electron-impact transitions from H 2 ( 1 Î£ + g ; v = 0) calculated in Franck-Condon approximation
coefficients = np.array([0.092,0.162,0.176,0.155,0.121,0.089,0.063,0.044,0.03,0.021,0.0147,0.0103,0.0072,0.0051,0.0036,0.0024,0.0016,0.0008,0.0002])
for vi_index in range(19):
	cross_section = coefficients[vi_index]*form_factor*(interpolator_f(np.log(baricenter_impact_energy.flatten()),vi_index).reshape(np.shape(baricenter_impact_energy)))
	cross_section[cross_section<0] = 0
	reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * H2_velocity*H2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(H2_velocity))		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	for v0_index in range(15):
		if v_index==0:
			H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20 * merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all*1e20 * 1 * np.exp(-(v0_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
H2pvi_H2v0__H3p_H1s = np.sum(H2pvi_H2v0__H3p_H1s,axis=0)
plt.figure();
plt.pcolor(temp_t, temp_r, H2pvi_H2v0__H3p_H1s ,vmin=max(np.max(H2pvi_H2v0__H3p_H1s)*1e-12,np.min(H2pvi_H2v0__H3p_H1s[H2pvi_H2v0__H3p_H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H2+(vi) +H2(v0) →H3+(v3) +H(1s), singlets and triplets reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate :
# H2+(vi) +H- → (H3∗) →H2(N1,3Λσ;v0) +H(1s),N ≤ 4
# I only know about N=2,3
# I should, but NOT include
# H2+(vi) +H- →H2(X1Σ+g;v0) +H(n≥2)
# because I don't have any info on that one reaction
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.4.1
# I don't have any info on the functions I should be using to calculate the energy resolved cross section, so I use the flat estimate they give
H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + Hm_energy).T
# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
cross_section1 = 7.5e-15 * 1e-4		# m^2	# channel N=2
cross_section2 = 5e-14 * 1e-4		# m^2	# channel N=2
# cross_section = cross_section1 + cross_section2
cross_section = np.ones_like(baricenter_impact_energy) * (cross_section1 + cross_section2)
reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
H2p_Hm__H2N13Λσ_H1s = merge_ne_prof_multipulse_interp_crop_limited*1e20 * nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20 * nHm_ne_all*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H2p_Hm__H2N13Λσ_H1s,vmin=max(np.max(H2p_Hm__H2N13Λσ_H1s)*1e-12,np.min(H2p_Hm__H2N13Λσ_H1s[H2p_Hm__H2N13Λσ_H1s>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H2+(vi) +H- → (H3∗) →H2(N1,3Λσ;v0) +H(1s),N=2,3 reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate H+2 (vi) +H- → (H∗ 3 ) →H+ 3 (v3) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.4.2
H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/(2*hydrogen_mass))**0.5).T
Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
Hm_energy = 0.5 * Hm_velocity**2 * (2*hydrogen_mass) * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + Hm_energy).T
# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
cross_section = 0.38/((baricenter_impact_energy**0.782)*(1+0.039*(baricenter_impact_energy**2.62))) * 1e-16 *1e-4	# m^2
cross_section[cross_section<0] = 0
reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
H2p_Hm__H3p_e = merge_ne_prof_multipulse_interp_crop_limited*1e20 * nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20 * nHm_ne_all*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H2p_Hm__H3p_e,vmin=max(np.max(H2p_Hm__H3p_e)*1e-12,np.min(H2p_Hm__H3p_e[H2p_Hm__H3p_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+2 (vi) +H- → (H∗ 3 ) →H+ 3 (v3) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


# reaction rate		H+ +H(1s) +H(1s) →H(1s) +H2+ (ν)
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 2.2.4, equation 46b
# valid for temperature up to 30000K (~2.587eV)
temperature = np.mean(np.array([T_Hp,T_H,T_H]),axis=0)
reaction_rate = 1.238/(( temperature )**1.046) *1e-29 * 1e-6 * 1e-6		# m^6/s
reaction_rate[np.logical_not(np.isfinite(reaction_rate))] = 0
reaction_rate_max = reaction_rate.flatten()[(np.abs(temperature-30000)).argmin()]
reaction_rate[temperature>30000]=reaction_rate_max
Hp_H_H__H_H2p = merge_ne_prof_multipulse_interp_crop_limited*1e20 * nHp_ne_all*((merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))**2)*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, Hp_H_H__H_H2p,vmin=max(np.max(Hp_H_H__H_H2p)*1e-12,np.min(Hp_H_H__H_H2p[Hp_H_H__H_H2p>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H+ +H(1s) +H(1s) →H(1s) +H2+ (ν) reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H(1s) +H(2) → H2+(v) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# and from Associative Ionisation in Low Energy Collisions, Brouillard, F., Urbain, X., 2002
# based on baricenter impact energy: I assume 1eV, maxwellian distribution
# T_H_single = 1/eV_to_K	# K
# H_velocity = np.logspace(np.log10(0.001),np.log10(10),200)*(2*boltzmann_constant_J*T_H_single/hydrogen_mass)**0.5
# H_velocity_PDF = 4*np.pi*H_velocity**2 * gauss( H_velocity, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H_single))**(3/2) , (T_H_single*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)
# H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
# baricenter_impact_energy = (np.zeros((len(H_energy),len(H_energy)))+H_energy).T + H_energy
H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(2*hydrogen_mass))**0.5).T
H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H_1_energy = 0.5 * H_1_velocity**2 * (2*hydrogen_mass) * J_to_eV
H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(2*hydrogen_mass))**0.5).T
H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H_2_energy = 0.5 * H_2_velocity**2 * (2*hydrogen_mass) * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
cross_section = gauss(baricenter_impact_energy,2.5 * 1e-17 * 1e-4,0.9,3.25)
reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s
H1s_H_2__H2p_e = population_states[0]*(merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H1s_H_2__H2p_e,vmin=max(np.max(H1s_H_2__H2p_e)*1e-12,np.min(H1s_H_2__H2p_e[H1s_H_2__H2p_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H(1s) +H(2) → H2+(v) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H(1s) +H(3) → H2+(v) + e
if False:	# Probrem: the approximation of the true cross section with this function is way too inaccurate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.3.3
	# based on baricenter impact energy: I assume 1eV, maxwellian distribution
	T_H_single = 1/eV_to_K	# K
	H_velocity = np.logspace(np.log10(0.001),np.log10(10),200)*(2*boltzmann_constant_J*T_H_single/hydrogen_mass)**0.5
	H_velocity_PDF = 4*np.pi*H_velocity**2 * gauss( H_velocity, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H_single))**(3/2) , (T_H_single*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)
	H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
	baricenter_impact_energy = (np.zeros((len(H_energy),len(H_energy)))+H_energy).T + H_energy
	cross_section = np.zeros_like(baricenter_impact_energy)
	cross_section[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] += 2.96*(3**4)*1e-19*1e-4/baricenter_impact_energy[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)]
	cross_section[np.logical_and(baricenter_impact_energy<=1,baricenter_impact_energy>0.1)] += 2.96*(3**4)*1e-18*1e-4
	cross_section[baricenter_impact_energy>1] += 2.96*(3**4)*1e-18*1e-4/(baricenter_impact_energy[baricenter_impact_energy>1]**(0.4*3))
	cross_section[cross_section<0] = 0
	reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))**2		# m^3 / s
	H1s_H_3__H2p_e = population_states[1]*(merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
else:
	# data from https://www-amdis.iaea.org/ALADDIN/
	H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(2*hydrogen_mass))**0.5).T
	H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
	H_1_energy = 0.5 * H_1_velocity**2 * (2*hydrogen_mass) * J_to_eV
	H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(2*hydrogen_mass))**0.5).T
	H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
	H_2_energy = 0.5 * H_2_velocity**2 * (2*hydrogen_mass) * J_to_eV
	baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
	interpolation_ev = np.array([5.5235E-02,7.1967E-02,9.0916E-02,1.1491E-01,1.4010E-01,1.7236E-01,2.0159E-01,2.3787E-01,2.7819E-01,3.2052E-01,3.6487E-01,4.1326E-01,4.5760E-01,5.2211E-01,5.7453E-01,6.2694E-01,6.9145E-01,7.5797E-01,8.3659E-01,9.0110E-01,9.7972E-01,1.0583E+00,1.1491E+00,1.2297E+00,1.3103E+00,1.4010E+00,1.5018E+00,1.5925E+00,1.6933E+00,1.7941E+00,1.8949E+00,1.9957E+00,2.1167E+00,2.2376E+00,2.3586E+00,2.4594E+00,2.5803E+00,2.7214E+00,2.8424E+00,2.9835E+00,3.1045E+00,3.2456E+00,3.3867E+00,3.5278E+00,3.6689E+00,3.8302E+00,3.9713E+00,4.1326E+00,4.2938E+00,4.4551E+00,4.6164E+00,4.7776E+00,4.9591E+00,5.1203E+00,5.3018E+00,5.4630E+00,5.6445E+00,5.8259E+00,5.9872E+00])
	interpolation_cross_section = np.array([7.3500E-16,5.9000E-16,4.2400E-16,3.1400E-16,2.3800E-16,2.0200E-16,3.1400E-16,2.6600E-16,2.1400E-16,2.1900E-16,1.9100E-16,2.0200E-16,2.6300E-16,2.6200E-16,2.3800E-16,2.6200E-16,2.1600E-16,2.3400E-16,2.5400E-16,2.8400E-16,2.7500E-16,2.8900E-16,3.0800E-16,3.0500E-16,3.3000E-16,2.8000E-16,3.1200E-16,2.3200E-16,3.0500E-16,2.7100E-16,2.5000E-16,2.2800E-16,2.4900E-16,2.3100E-16,2.0500E-16,1.9200E-16,2.1400E-16,2.0200E-16,2.1000E-16,2.0100E-16,1.8700E-16,1.8600E-16,2.0800E-16,1.7100E-16,1.6100E-16,1.5200E-16,1.6100E-16,1.2900E-16,1.2500E-16,1.2500E-16,1.2600E-16,1.1000E-16,1.0300E-16,1.1200E-16,1.0600E-16,9.3700E-17,7.7400E-17,8.0400E-17,6.9900E-17]) * 1e-4		# m^2
	interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
	cross_section = np.zeros_like(baricenter_impact_energy)
	cross_section[baricenter_impact_energy>0] = np.exp(interpolator_cross_section(np.log(baricenter_impact_energy[baricenter_impact_energy>0])))
	cross_section[cross_section<0] = 0
	reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
	H1s_H_3__H2p_e = population_states[1]*(merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H1s_H_3__H2p_e,vmin=max(np.max(H1s_H_3__H2p_e)*1e-12,np.min(H1s_H_3__H2p_e[H1s_H_3__H2p_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H(1s) +H(3) → H2+(v) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()

# reaction rate		H(1s) +H(4) → H2+(v) + e
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# based on baricenter impact energy: I assume 1eV, maxwellian distribution
# T_H_single = 1/eV_to_K	# K
# H_velocity = np.logspace(np.log10(0.001),np.log10(10),200)*(2*boltzmann_constant_J*T_H_single/hydrogen_mass)**0.5
# H_velocity_PDF = 4*np.pi*H_velocity**2 * gauss( H_velocity, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H_single))**(3/2) , (T_H_single*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)
# H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
# baricenter_impact_energy = (np.zeros((len(H_energy),len(H_energy)))+H_energy).T + H_energy
H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(2*hydrogen_mass))**0.5).T
H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H_1_energy = 0.5 * H_1_velocity**2 * (2*hydrogen_mass) * J_to_eV
H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(2*hydrogen_mass))**0.5).T
H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
H_2_energy = 0.5 * H_2_velocity**2 * (2*hydrogen_mass) * J_to_eV
baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
cross_section = np.zeros_like(baricenter_impact_energy)
cross_section[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] += 2.96*(4**4)*1e-19*1e-4/baricenter_impact_energy[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)]
cross_section[np.logical_and(baricenter_impact_energy<=1,baricenter_impact_energy>0.1)] += 2.96*(4**4)*1e-18*1e-4
cross_section[baricenter_impact_energy>1] += 2.96*(4**4)*1e-18*1e-4/(baricenter_impact_energy[baricenter_impact_energy>1]**(0.4*4))
cross_section[cross_section<0] = 0
reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s
H1s_H_4__H2p_e = population_states[2]*(merge_ne_prof_multipulse_interp_crop_limited*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
plt.figure();
plt.pcolor(temp_t, temp_r, H1s_H_4__H2p_e,vmin=max(np.max(H1s_H_4__H2p_e)*1e-12,np.min(H1s_H_4__H2p_e[H1s_H_4__H2p_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('H(1s) +H(4) → H2+(v) + e reaction rate\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
plt.close()


rate_creation_Hm = e_H__Hm + e_H2X1Σg__Hm_H1s
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_Hm,vmin=max(np.max(rate_creation_Hm)*1e-3,np.min(rate_creation_Hm[rate_creation_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H-\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_Hm = e_Hm__e_H_e + Hp_Hm__H_2_H + Hp_Hm__H_3_H + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + Hm_H2v__H_H2v_e
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of H-\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_Hm-rate_destruction_Hm, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of H-\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2p = Hp_Hm__H2p_e + e_H2N13Λσ__e_H2pX2Σg_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p,vmin=max(np.max(rate_creation_H2p)*1e-3,np.min(rate_creation_H2p[rate_creation_H2p>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2p_from_Hm = Hp_Hm__H2p_e/rate_creation_H2p
rate_creation_H2p_from_Hm[np.logical_not(np.isfinite(rate_creation_H2p_from_Hm))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_Hm,vmin=max(np.max(rate_creation_H2p_from_Hm)*1e-3,np.min(rate_creation_H2p_from_Hm[rate_creation_H2p_from_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2+ from H-\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2p_from_Hp = (Hp_Hm__H2p_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p)/rate_creation_H2p
rate_creation_H2p_from_Hp[np.logical_not(np.isfinite(rate_creation_H2p_from_Hp))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_Hp,vmin=max(np.max(rate_creation_H2p_from_Hp)*1e-3,np.min(rate_creation_H2p_from_Hp[rate_creation_H2p_from_Hp>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2+ from H+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2p_from_H = (Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e)/rate_creation_H2p
rate_creation_H2p_from_H[np.logical_not(np.isfinite(rate_creation_H2p_from_H))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_H,vmin=max(np.max(rate_creation_H2p_from_H)*1e-3,np.min(rate_creation_H2p_from_H[rate_creation_H2p_from_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2+ from H\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2p_from_H2 = (e_H2N13Λσ__e_H2pX2Σg_e + Hp_H2X1Σg__H1s_H2pX2Σg)/rate_creation_H2p
rate_creation_H2p_from_H2[np.logical_not(np.isfinite(rate_creation_H2p_from_H2))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_H2,vmin=max(np.max(rate_creation_H2p_from_H2)*1e-3,np.min(rate_creation_H2p_from_H2[rate_creation_H2p_from_H2>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2+ from H2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_H2p = e_H2p__XXX__e_Hp_H1s + e_H2p__H1s_Hn2 + e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s + H2p_Hm__H3p_e
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of H2+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p-rate_destruction_H2p, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of H2+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2 = Hm_H1s__H2_v_e + Hp_H_H__H_H2 + H_H_H__H_H2 + H2p_Hm__H2N13Λσ_H1s
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2,vmin=max(np.max(rate_creation_H2)*1e-3,np.min(rate_creation_H2[rate_creation_H2>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2_from_Hm = (Hm_H1s__H2_v_e + H2p_Hm__H2N13Λσ_H1s)/rate_creation_H2
rate_creation_H2_from_Hm[np.logical_not(np.isfinite(rate_creation_H2_from_Hm))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2_from_Hm,vmin=max(np.max(rate_creation_H2_from_Hm)*1e-3,np.min(rate_creation_H2_from_Hm[rate_creation_H2_from_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2 from H-\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2_from_Hp = Hp_H_H__H_H2/rate_creation_H2
rate_creation_H2_from_Hp[np.logical_not(np.isfinite(rate_creation_H2_from_Hp))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2_from_Hp,vmin=max(np.max(rate_creation_H2_from_Hp)*1e-3,np.min(rate_creation_H2_from_Hp[rate_creation_H2_from_Hp>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2 from H+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2_from_H = (Hp_H_H__H_H2 + H_H_H__H_H2)/rate_creation_H2
rate_creation_H2_from_H[np.logical_not(np.isfinite(rate_creation_H2_from_H))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2,vmin=max(np.max(rate_creation_H2_from_H)*1e-3,np.min(rate_creation_H2_from_H[rate_creation_H2_from_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2 from H\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H2_from_H2p = H2p_Hm__H2N13Λσ_H1s/rate_creation_H2
rate_creation_H2_from_H2p[np.logical_not(np.isfinite(rate_creation_H2_from_H2p))]=0
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2_from_H2p,vmin=max(np.max(rate_creation_H2_from_H2p)*1e-3,np.min(rate_creation_H2_from_H2p[rate_creation_H2_from_H2p>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H2 from H2+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_H2 = e_H2_X1Σg_x__e_H_H + e_H2N13Λσ__e_H2pX2Σg_e + e_H2N13Λσ__Hp_H_2e + e_H2X1Σg__Hm_H1s + e_H2X1Σg__e_H1s_H1s + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H2X1Σg__Hp_H1s_H1s + H1s_H2v__H1s_2H1s + H2v0_H2v__H2v0_2H1s + H2pvi_H2v0__H3p_H1s
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of H2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2-rate_destruction_H2, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of H2\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H3p = H2pvi_H2v0__H3p_H1s + H2p_Hm__H3p_e
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H2p,vmin=max(np.max(rate_creation_H2p)*1e-3,np.min(rate_creation_H2p[rate_creation_H2p>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H3+\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_H = e_Hm__e_H_e + 2*Hp_Hm__H_2_H + 2*Hp_Hm__H_3_H + Hm_H1s__H1s_H1s_e + 2*e_H2_X1Σg_x__e_H_H + e_H2N13Λσ__Hp_H_2e + e_H2X1Σg__Hm_H1s + 2*e_H2X1Σg__e_H1s_H1s + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H2X1Σg__Hp_H1s_H1s + Hm_H2v__H_H2v_e + 2*H1s_H2v__H1s_2H1s + 2*H2v0_H2v__H2v0_2H1s + e_H2p__XXX__e_Hp_H1s + 2*e_H2p__H1s_Hn2 + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H,vmin=max(np.max(rate_creation_H)*1e-3,np.min(rate_creation_H[rate_creation_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_H = Hm_H1s__H2_v_e + 2*Hp_H_H__H_H2 + 2*H_H_H__H_H2 + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e + e_H__Hm
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_H,vmin=max(np.max(rate_destruction_H)*1e-3,np.min(rate_destruction_H[rate_destruction_H>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of H from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_H-rate_destruction_H, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of H from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_creation_Hp = e_H2N13Λσ__Hp_H_2e + e_H2p__XXX__e_Hp_H1s + 2*e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_Hp,vmin=max(np.max(rate_creation_Hp)*1e-3,np.min(rate_creation_Hp[rate_creation_Hp>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of H+ from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_Hp = Hp_Hm__H_2_H + Hp_Hm__H_3_H + Hp_Hm__H2p_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_Hp,vmin=max(np.max(rate_destruction_Hp)*1e-3,np.min(rate_destruction_Hp[rate_destruction_Hp>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of H+ from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_Hp-rate_destruction_Hp, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of H+ from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


rate_creation_e = e_Hm__e_H_e + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + e_H2N13Λσ__e_H2pX2Σg_e + e_H2N13Λσ__Hp_H_2e + Hm_H2v__H_H2v_e + e_H2p__e_Hp_Hp_e + H2p_Hm__H3p_e + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e
plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_e,vmin=max(np.max(rate_creation_e)*1e-3,np.min(rate_creation_e[rate_creation_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of creation of e- from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

rate_destruction_e = e_H__Hm + e_H2X1Σg__Hm_H1s + e_H2p__H1s_Hn2
plt.figure();
plt.pcolor(temp_t, temp_r, rate_destruction_e,vmin=max(np.max(rate_destruction_e)*1e-3,np.min(rate_destruction_e[rate_destruction_e>0])), cmap='rainbow',norm=LogNorm());
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Rate of destruction of e- from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

plt.figure();
plt.pcolor(temp_t, temp_r, rate_creation_e-rate_destruction_e, cmap='rainbow');
plt.colorbar().set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('Net rate of creation of e- from molecules only\nOES_multiplier,spatial_factor,time_shift_factor \n' + str([1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
plt.figure();
plt.plot(time_crop, np.sum(area * rate_creation_Hm,axis=1)/np.max(np.sum(area * rate_creation_Hm,axis=1)), label='H- creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_Hm,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_Hm,axis=1)/np.max(np.sum(area * rate_destruction_Hm,axis=1)), label='H- destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_Hm,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_creation_H2p,axis=1)/np.max(np.sum(area * rate_creation_H2p,axis=1)), label='H2+ creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H2p,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_H2p,axis=1)/np.max(np.sum(area * rate_destruction_H2p,axis=1)), label='H2+ destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H2p,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_creation_H2,axis=1)/np.max(np.sum(area * rate_creation_H2,axis=1)), label='H2 creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H2,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_H2,axis=1)/np.max(np.sum(area * rate_destruction_H2,axis=1)), label='H2 destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H2,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)/np.max(np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)/np.max(np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)/np.sum(area))));
plt.legend(loc='best', fontsize='x-small')
plt.xlabel('time from beginning of pulse [ms]')
plt.ylabel('relative radial average [au]')
plt.title('Time evolution of the radially averaged molecular reaction rates '+str(target_OES_distance)+'mm from the target')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
plt.figure();
threshold_radious = 0.025
plt.plot(time_crop, np.sum(area * effective_ionisation_rates,axis=1)/np.max(np.sum(area * effective_ionisation_rates,axis=1)), label='ionisation rate from ADAS\n(max='+'%.3g # m^-3 s-1)' %(np.max(np.sum(area * effective_ionisation_rates,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * effective_recombination_rates,axis=1)/np.max(np.sum(area * effective_recombination_rates,axis=1)), label='recombination rate from ADAS\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * effective_recombination_rates,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_creation_H,axis=1)/np.max(np.sum(area * rate_creation_H,axis=1)), label='H creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_H,axis=1)/np.max(np.sum(area * rate_destruction_H,axis=1)), label='H destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_creation_Hp,axis=1)/np.max(np.sum(area * rate_creation_Hp,axis=1)), label='H+ creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_Hp,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_Hp,axis=1)/np.max(np.sum(area * rate_destruction_Hp,axis=1)), label='H+ destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_Hp,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_creation_e,axis=1)/np.max(np.sum(area * rate_creation_e,axis=1)), label='e- creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_e,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * rate_destruction_e,axis=1)/np.max(np.sum(area * rate_destruction_e,axis=1)), label='e- destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_e,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)/np.max(np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * merge_Te_prof_multipulse_interp_crop_limited,axis=1)/np.sum(area))));
plt.plot(time_crop, np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)/np.max(np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * merge_ne_prof_multipulse_interp_crop_limited,axis=1)/np.sum(area))));
plt.legend(loc='best', fontsize='xx-small')
plt.xlabel('time from beginning of pulse [ms]')
plt.ylabel('relative radial average [au]')
plt.title('Time evolution of the radially averaged reaction rates\ncomparixon of atomic vs molecular processes\n'+str(target_OES_distance)+'mm from the target')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# Calculation of the H- destruction mean free path
thermal_velocity_Hm = ( (T_Hm*boltzmann_constant_J)/ hydrogen_mass)**0.5
ionization_length_Hm = thermal_velocity_Hm/( rate_destruction_Hm/(nHm_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20) )
ionization_length_Hm = np.where(np.isnan(ionization_length_Hm), 0, ionization_length_Hm)
ionization_length_Hm = np.where(np.isinf(ionization_length_Hm), np.nan, ionization_length_Hm)
ionization_length_Hm = np.where(np.isnan(ionization_length_Hm), np.nanmax(ionization_length_Hm[np.isfinite(ionization_length_Hm)]), ionization_length_Hm)
plt.figure();
plt.pcolor(temp_t, temp_r, ionization_length_Hm,vmax=min(np.max(ionization_length_Hm),1), cmap='rainbow', norm=LogNorm());
plt.colorbar().set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('"destruction" length of H- from Yacora\nmean H- temperature '+str(np.mean(T_Hm))+'K\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# Calculation of the H2+ destruction mean free path
thermal_velocity_H2p = ( (T_H2p*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5
ionization_length_H2p = thermal_velocity_H2p/( rate_destruction_H2p/(nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20) )
ionization_length_H2p = np.where(np.isnan(ionization_length_H2p), 0, ionization_length_H2p)
ionization_length_H2p = np.where(np.isinf(ionization_length_H2p), np.nan, ionization_length_H2p)
ionization_length_H2p = np.where(np.isnan(ionization_length_H2p), np.nanmax(ionization_length_H2p[np.isfinite(ionization_length_H2p)]), ionization_length_H2p)
plt.figure();
plt.pcolor(temp_t, temp_r, ionization_length_H2p,vmax=min(np.max(ionization_length_H2p),1), cmap='rainbow', norm=LogNorm());
plt.colorbar().set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('"destruction" length of H2+ from Yacora\nmean H2+ temperature '+str(np.mean(T_H2p))+'K\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

# Calculation of the H2 destruction mean free path
thermal_velocity_H2 = ( (T_H2*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5
ionization_length_H2 = thermal_velocity_H2/( rate_destruction_H2/(nH2_ne_all*merge_ne_prof_multipulse_interp_crop_limited*1e20) )
ionization_length_H2 = np.where(np.isnan(ionization_length_H2), 0, ionization_length_H2)
ionization_length_H2 = np.where(np.isinf(ionization_length_H2), np.nan, ionization_length_H2)
ionization_length_H2 = np.where(np.isnan(ionization_length_H2), np.nanmax(ionization_length_H2[np.isfinite(ionization_length_H2)]), ionization_length_H2)
plt.figure();
plt.pcolor(temp_t, temp_r, ionization_length_H2,vmax=min(np.max(ionization_length_H2),1), cmap='rainbow', norm=LogNorm());
plt.colorbar().set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
plt.axes().set_aspect(20)
plt.xlabel('time [ms]')
plt.ylabel('radial location [m]')
plt.title('"destruction" length of H2 from Yacora\nmean H2 temperature '+str(np.mean(T_H2))+'K\nOES_multiplier,spatial_factor,time_shift_factor \n' + str(
	[1, spatial_factor, time_shift_factor]))
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()

length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
i_plasma_radius=np.array([])
for FWHM in (averaged_profile_sigma*2.355):
	i_plasma_radius=np.append(i_plasma_radius, np.abs(r_crop + np.median(np.diff(r_crop))/2-FWHM).argmin())
i_plasma_radius=i_plasma_radius.astype(int)
time_rate_creation_e = np.array([])
time_rate_destruction_e = np.array([])
time_rate_flow_target_e = np.array([])
time_effective_ionisation_rates = np.array([])
time_effective_recombination_rates = np.array([])
for index,ii_plasma_radius in enumerate(i_plasma_radius):
	time_rate_flow_target_e=np.append(time_rate_flow_target_e,np.median(np.diff(r_crop))*np.sum(2*np.pi*r_crop[:ii_plasma_radius]*0.5*merge_ne_prof_multipulse_interp_crop_limited[index,:ii_plasma_radius]*1e20*(boltzmann_constant_J*(1+5/3)*merge_Te_prof_multipulse_interp_crop_limited[index,:ii_plasma_radius]/(eV_to_K*hydrogen_mass))**0.5))
	time_rate_creation_e=np.append(time_rate_creation_e,length*np.median(np.diff(r_crop))*np.sum(rate_creation_e[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
	time_rate_destruction_e=np.append(time_rate_destruction_e,length*np.median(np.diff(r_crop))*np.sum(rate_destruction_e[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
	time_effective_ionisation_rates=np.append(time_effective_ionisation_rates,length*np.median(np.diff(r_crop))*np.sum(effective_ionisation_rates[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
	time_effective_recombination_rates=np.append(time_effective_recombination_rates,length*np.median(np.diff(r_crop))*np.sum(effective_recombination_rates[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
plt.figure();
plt.plot(time_crop, time_rate_flow_target_e, label='sheath absorption at target (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_flow_target_e)));
plt.plot(time_crop, time_rate_creation_e, label='creation via molecules (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_creation_e)));
plt.plot(time_crop, time_rate_destruction_e, label='destruction via molecules (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_destruction_e)));
plt.plot(time_crop, time_effective_ionisation_rates, label='ionisation from H (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_effective_ionisation_rates)));
plt.plot(time_crop, time_effective_recombination_rates, label='recombination fromH+ (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_effective_recombination_rates)));
plt.legend(loc='best', fontsize='x-small')
plt.xlabel('time from beginning of pulse [ms]')
plt.ylabel('creation/destruction rate [#/s]')
plt.title('Total electron creation/destruction rates skimmer to target vs target sheath condition\n'+'total sheath %.3g#, total volume %.3g' %(np.median(np.diff(time_crop))*np.sum(time_rate_flow_target_e),np.median(np.diff(time_crop))*np.sum( time_rate_creation_e-time_rate_destruction_e+time_effective_ionisation_rates-time_effective_recombination_rates ))+'\n'+str(target_OES_distance)+'mm from the target, '+str(target_chamber_pressure)+'Pa target chamber pressure')
figure_index += 1
plt.savefig(path_where_to_save_everything + mod4 + '/pass_SS_'+str(global_pass)+'_post_process_mega_global_fit' + str(
	figure_index) + '.eps', bbox_inches='tight')
plt.close()


np.savez_compressed(path_where_to_save_everything + mod4 +'/SS_molecular_results'+str(global_pass),ionization_length_H=ionization_length_H,rate_creation_Hm=rate_creation_Hm,rate_destruction_Hm=rate_destruction_Hm,ionization_length_Hm=ionization_length_Hm,rate_creation_H2p=rate_creation_H2p,rate_destruction_H2p=rate_destruction_H2p,ionization_length_H2p=ionization_length_H2p,rate_creation_H2=rate_creation_H2,rate_destruction_H2=rate_destruction_H2,ionization_length_H2=ionization_length_H2)
'''
