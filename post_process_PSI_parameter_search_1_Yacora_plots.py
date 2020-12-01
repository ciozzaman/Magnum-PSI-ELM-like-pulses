# 27/02/2020
# This is an extract from post_process_PSI_parameter_search_1_Yacora.py
# only to have them in a separate place.
mkl.set_num_threads(number_cpu_available)
for i in range(2):
	print(' ')
try:
	print('global_pass_'+str(global_pass))
except Exception as e:
	print(e)
	global_pass = 2
	print('global_pass_'+str(global_pass))
for i in range(2):
	print(' ')

figure_index = 0
pre_title = 'merge %.3g, B=%.3gT, pos=%.3gmm, P=%.3gPa, ELMen=%.3gJ \n' %(merge_ID_target,magnetic_field,target_OES_distance,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6)
color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise', 'teal', 'olive']

if False:
	T_Hp = np.min([np.max([1000*np.ones_like(Te_all),Te_all/eV_to_K],axis=0),12000*np.ones_like(Te_all)],axis=0)	# K
	T_Hm = 5000*np.ones_like(Te_all)	# K
	T_H2p = 5000*np.ones_like(Te_all)	# K
	T_H2 = 0.1/eV_to_K*np.ones_like(Te_all)	# K
	T_H = 1*np.ones_like(Te_all)/eV_to_K	# K
elif True:
	# Te_all[Te_all<300*eV_to_K]=300*eV_to_K
	Te_all[nH_ne_all==0]=merge_Te_prof_multipulse_interp_crop_limited[nH_ne_all==0]
	T_Hp = Te_all/eV_to_K	# K
	T_Hp[T_Hp<300]=300
	T_Hm = np.exp(TH2_fit_from_simulations(np.log(Te_all)))/eV_to_K	# K
	T_Hm[T_Hm<300]=300
	T_H2p = np.exp(TH2_fit_from_simulations(np.log(Te_all)))/eV_to_K	# K
	T_H2p[T_H2p<300]=300
	T_H2 = np.exp(TH2_fit_from_simulations(np.log(Te_all)))/eV_to_K	# K
	T_H2[T_H2<300]=300
	T_H = np.exp(TH_fit_from_simulations(np.log(Te_all)))/eV_to_K	# K
	T_H[T_H<300]=300
	nHp_ne_all[nHp_ne_all==0]=1
	ionisation_potential = 13.6	# eV
	dissociation_potential = 2.2	# eV



if initial_conditions:

	profile_centres_score[np.abs(profile_centres_score)>np.max(TS_r/1000)] = np.max(TS_r/1000)
	plt.figure(figsize=(7, 10));
	plt.pcolor(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])], TS_r/1000, merge_ne_prof_multipulse[np.logical_and(merge_time_original>=time_crop[0]-1e-5,merge_time_original<=time_crop[-1]+1e-5)].T, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.errorbar(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],profile_centres[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,yerr=profile_centres_score[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,color='b',label='local centre')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],(profile_centres+profile_sigma*2.355/2)[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],(profile_centres-profile_sigma*2.355/2)[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,'--',color='grey')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])][[0,-1]],[centre/1000,centre/1000],'k--',label='averaged centre')
	plt.legend(loc='best', fontsize='xx-small')
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [mm]')
	plt.title(pre_title+'untreated ne, centre at %.3gmm' %(centre))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(7, 10));
	plt.pcolor(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])], TS_r/1000, merge_Te_prof_multipulse[np.logical_and(merge_time_original>=time_crop[0]-1e-5,merge_time_original<=time_crop[-1]+1e-5)].T, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
	plt.errorbar(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],profile_centres[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,yerr=profile_centres_score[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,color='b',label='local centre')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],(profile_centres+profile_sigma*2.355/2)[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])],(profile_centres-profile_sigma*2.355/2)[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])]/1000,'--',color='grey')
	plt.plot(merge_time_original[np.logical_and(merge_time_original>=time_crop[0],merge_time_original<=time_crop[-1])][[0,-1]],[centre/1000,centre/1000],'k--',label='averaged centre')
	plt.legend(loc='best', fontsize='xx-small')
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [mm]')
	plt.title(pre_title+'untreated Te, centre at %.3gmm' %(centre))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.plot(time_crop, max_nH2_from_pressure_all/1e20);
	plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('density [10^20 # m^-3]')
	plt.title(pre_title+'Maximum estimated density for H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.plot(time_crop, heat_inflow_upstream,label='heat input from upstream');
	plt.plot(time_crop, heat_flux_target,label='heat output to target');
	plt.plot(time_crop, net_power_in,label='net heat to surrounding gas');
	plt.legend(loc='best', fontsize='xx-small')
	plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('energy [J]')
	plt.title(pre_title+'Power balance in the gas surrounding the plasma')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t,temp_r, merge_ne_prof_multipulse_interp_crop_limited, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, merge_dne_prof_multipulse_interp_crop_limited, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'electron density uncertainty')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'electron temperature')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, merge_dTe_prof_multipulse_interp_crop_limited, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'electron temperature uncertainty')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,0], cmap='rainbow',vmax=np.median(np.sort(inverted_profiles_crop[:,0].flatten())[-5:]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Emissivity Balmer line n=4>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,3], cmap='rainbow',vmax=np.median(np.sort(inverted_profiles_crop[:,3].flatten())[-5:]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Emissivity Balmer line n=7>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Here I check the minimum amount of extra radiation I see using data from simulations
	max_nH2_ne_from_simulations = np.zeros_like(nH2_ne_all)
	max_nH_ne_from_simulations = np.zeros_like(nH2_ne_all)
	for my_time_pos in range(len(time_crop)):
		for my_r_pos in range(len(r_crop)):
			if not (ne_all[my_time_pos, my_r_pos] == 0 or Te_all[my_time_pos, my_r_pos] == 0):
				max_nH2_ne_from_simulations[my_time_pos, my_r_pos] = nH2_ne_fit_from_simulations(max(Te_all[my_time_pos, my_r_pos],0.1))
				max_nH_ne_from_simulations[my_time_pos, my_r_pos] = nH_ne_fit_from_simulations(max(Te_all[my_time_pos, my_r_pos],0.1))
	multiplicative_factor = energy_difference * einstein_coeff / J_to_eV
	pop_coeff_max = ((excitation[n_list_all-4] *  max_nH_ne_from_simulations).T /multiplicative_factor[n_list_all-4]).T
	pop_coeff_max += ((recombination[n_list_all-4] * np.ones_like(nH_ne_all)).T /multiplicative_factor[n_list_all-4]).T
	temp = From_H2_pop_coeff_full_extra(np.array([Te_all.flatten(),ne_all.flatten()*1e20]).T,n_list_all)
	pop_coeff_max += (max_nH2_ne_from_simulations.flatten()*temp.T).reshape((np.shape(pop_coeff_max)))
	pop_states_max = pop_coeff_max * ne_all**2 * 1e40
	max_emission = (pop_states_max.T*multiplicative_factor[n_list_all-4]).T
	extra_emission_4 = inverted_profiles_crop[:,0]-max_emission[0]
	extra_emission_4[extra_emission_4<0]=0
	extra_emission_4[Te_all==0]=0
	extra_emission_4[ne_all==0]=0
	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, extra_emission_4/inverted_profiles_crop[:,0], cmap='rainbow',vmax=1,vmin=0);
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of extra emissivity Balmer line n=4>2\nnot excitation, recombination, H2 dissociation')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	extra_emission_7 = inverted_profiles_crop[:,3]-max_emission[3]
	extra_emission_7[extra_emission_7<0]=0
	extra_emission_7[Te_all==0]=0
	extra_emission_7[ne_all==0]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, extra_emission_7/inverted_profiles_crop[:,2], cmap='rainbow',vmax=1,vmin=0);
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of extra emissivity Balmer line n=7>2\nnot excitation, recombination, H2 dissociation')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the ionisation mean free path
	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionization_length_H = thermal_velocity_H/(temp * ne_all * 1e20 )
	ionization_length_H = np.where(np.isnan(ionization_length_H), 0, ionization_length_H)
	ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
	ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ionization_length_H,vmax=min(np.max(ionization_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'ionization length of neutral H from ADAS\ncold H (from simul)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# H_temp = Te, in case the most of atomic hydrogen comes from recombination
	thermal_velocity_H = ( (Te_all*boltzmann_constant_J/eV_to_K)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionization_length_H = thermal_velocity_H/(temp * ne_all * 1e20 )
	ionization_length_H = np.where(np.isnan(ionization_length_H), 0, ionization_length_H)
	ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
	ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ionization_length_H,vmax=min(np.max(ionization_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'ionization length of neutral H from ADAS\n hot H (TH = Te, H from recombination)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	if merge_ID_target in [851,85,90,92,94,74,79,80,81,83,84,66,71,73]:
		if merge_ID_target in [851,85,90,92,94]:
			merge_ID_target_at_the_target = 91
		elif merge_ID_target in [74,79,80,81,83,84]:
			merge_ID_target_at_the_target = 82
		elif merge_ID_target in [66,71,73]:
			merge_ID_target_at_the_target = 72
		else:
			print('something wrong in definint the reference case for TS at the target')
			exit()

		all_j=find_index_of_file(merge_ID_target_at_the_target,df_settings,df_log,only_OES=True)
		target_chamber_pressure_at_the_target = []
		target_OES_distance_at_the_target = []
		for j in all_j:
			target_chamber_pressure_at_the_target.append(df_log.loc[j,['p_n [Pa]']])
			target_OES_distance_at_the_target.append(df_log.loc[j,['T_axial']])
		target_chamber_pressure_at_the_target = np.nanmean(target_chamber_pressure_at_the_target)	# Pa
		target_OES_distance_at_the_target = np.nanmean(target_OES_distance_at_the_target)	# Pa
		# Ideal gas law
		max_nH2_from_pressure_at_the_target = target_chamber_pressure_at_the_target/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K

		path_where_to_save_everything_target = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target_at_the_target)
		merge_Te_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_dne_multipulse']
		merge_time_original_target = np.load(path_where_to_save_everything_target + '/TS_data_merge_' + str(merge_ID_target_at_the_target) + '.npz')['merge_time']
		merge_time_target = time_shift_factor + merge_time_original_target

		TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
		TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
		TS_dr = np.median(np.diff(TS_r)) / 1000
		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		profile_target_centres = []
		profile_target_sigma = []
		profile_target_centres_score = []
		for index in range(np.shape(merge_ne_prof_multipulse_target)[0]):
			yy = merge_ne_prof_multipulse_target[index]
			yy_sigma = merge_dne_multipulse_target[index]
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			if np.sum(yy>0)<5:
				profile_target_centres.append(0)
				profile_target_sigma.append(10)
				profile_target_centres_score.append(np.max(TS_r))
				continue
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), 10, 0]
			bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
			profile_target_centres.append(fit[0][-1])
			profile_target_sigma.append(fit[0][-2])
			profile_target_centres_score.append(fit[1][-1, -1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		profile_target_centres = np.array(profile_target_centres)
		profile_target_sigma = np.array(profile_target_sigma)
		profile_target_centres_score = np.array(profile_target_centres_score)
		# centre = np.nanmean(profile_target_centres[profile_target_centres_score < 1])
		centre_target = np.nansum(profile_target_centres/(profile_target_centres_score**1))/np.sum(1/profile_target_centres_score**1)
		TS_r_new_target = (TS_r - centre_target) / 1000
		print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
		# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

		# This is the mean of Te and ne weighted in their own uncertainties.
		temp1 = np.zeros_like(inverted_profiles[:, 0])
		temp2 = np.zeros_like(inverted_profiles[:, 0])
		temp3 = np.zeros_like(inverted_profiles[:, 0])
		temp4 = np.zeros_like(inverted_profiles[:, 0])
		interp_range_t = max(dt, TS_dt) * 1.5
		interp_range_r = max(dx, TS_dr) * 1.5
		weights_r = (np.zeros_like(merge_Te_prof_multipulse_target) + TS_r_new_target)
		weights_t = (((np.zeros_like(merge_Te_prof_multipulse_target)).T + merge_time_target).T)
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time_target - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(np.abs(TS_r_new_target) - value_r) < interp_range_r) == 0:
					continue
				elif np.sum(np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)) == 0:
					continue
				selected_values_t = np.logical_and(np.abs(merge_time_target - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_target, axis=1) > 0)
				selected_values_r = np.abs(np.abs(TS_r_new_target) - value_r) < interp_range_r
				selecte_values = (np.array([selected_values_t])).T * selected_values_r
				selecte_values[merge_Te_prof_multipulse_target == 0] = False
				# weights = 1/(weights_r[selecte_values]-value_r)**2 + 1/(weights_t[selecte_values]-value_t)**2
				weights = 1/((weights_t[selecte_values]-value_t)/interp_range_t)**2 + 1/((weights_r[selecte_values]-value_r)/interp_range_r)**2
				if np.sum(selecte_values) == 0:
					continue
				# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse_target[selecte_values]*weights / merge_dTe_multipulse_target[selecte_values]) / np.sum(weights / merge_dTe_multipulse_target[selecte_values])
				# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse_target[selecte_values]*weights / merge_dne_multipulse_target[selecte_values]) / np.sum(weights / merge_dne_multipulse_target[selecte_values])
				if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					# temp2[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dTe_multipulse[selecte_values]) ** 2)) ** 0.5
					# temp4[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dne_multipulse[selecte_values]) ** 2)) ** 0.5
					temp2[i_t, i_r] = 1/(np.sum(1 / merge_dTe_multipulse_target[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse_target[selecte_values])/merge_dTe_multipulse_target[selecte_values])**2 )**0.5)
					temp4[i_t, i_r] = 1/(np.sum(1 / merge_dne_multipulse_target[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse_target[selecte_values])/merge_dne_multipulse_target[selecte_values])**2 )**0.5)
				else:
					# temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selecte_values])/merge_dTe_multipulse[selecte_values])**2 )**0.5)
					# temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selecte_values])/merge_dne_multipulse[selecte_values])**2 )**0.5)
					temp2[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dTe_multipulse_target[selecte_values]),(np.max(merge_Te_prof_multipulse_target[selecte_values])-np.min(merge_Te_prof_multipulse_target[selecte_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					temp4[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dne_multipulse_target[selecte_values]),(np.max(merge_ne_prof_multipulse_target[selecte_values])-np.min(merge_ne_prof_multipulse_target[selecte_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties

		merge_Te_prof_multipulse_interp_target = np.array(temp1)
		merge_dTe_prof_multipulse_interp_target = np.array(temp2)
		merge_ne_prof_multipulse_interp_target = np.array(temp3)
		merge_dne_prof_multipulse_interp_target = np.array(temp4)
		temp_r, temp_t = np.meshgrid(r, new_timesteps)

		# I crop to the usefull stuff
		start_time = np.abs(new_timesteps - 0).argmin()
		end_time = np.abs(new_timesteps - 1.5).argmin() + 1
		time_crop = new_timesteps[start_time:end_time]
		start_r = np.abs(r - 0).argmin()
		end_r = np.abs(r - 5).argmin() + 1
		r_crop = r[start_r:end_r]
		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		merge_Te_prof_multipulse_interp_target_crop = merge_Te_prof_multipulse_interp_target[start_time:end_time,start_r:end_r]
		merge_Te_prof_multipulse_interp_target_crop[merge_Te_prof_multipulse_interp_target_crop<0]=0
		merge_dTe_prof_multipulse_interp_target_crop = merge_dTe_prof_multipulse_interp_target[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_target_crop = merge_ne_prof_multipulse_interp_target[start_time:end_time,start_r:end_r]
		merge_ne_prof_multipulse_interp_target_crop[merge_ne_prof_multipulse_interp_target_crop<0]=0
		merge_dne_prof_multipulse_interp_target_crop = merge_dne_prof_multipulse_interp_target[start_time:end_time, start_r:end_r]
		inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
		inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
		inverted_profiles_sigma_crop = inverted_profiles_sigma[start_time:end_time, :, start_r:end_r]
		inverted_profiles_sigma_crop[np.isnan(inverted_profiles_sigma_crop)] = 0
		all_fits_crop = all_fits[start_time:end_time]
		# inverted_profiles_crop[inverted_profiles_crop<0] = 0

		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		averaged_profile_sigma_target = []
		for index in range(np.shape(merge_ne_prof_multipulse_interp_target_crop)[0]):
			yy = merge_ne_prof_multipulse_interp_target_crop[index]
			yy_sigma = merge_dne_prof_multipulse_interp_target_crop[index]
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			if (np.sum(yy>0)==0 or np.sum(yy_sigma>0)==0):
				averaged_profile_sigma_target.append(0)
				continue
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), np.max(r_crop)/2, 0]
			bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
			fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma)
			averaged_profile_sigma_target.append(fit[0][-2])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		averaged_profile_sigma_target = np.array(averaged_profile_sigma_target)

		# x_local = xx - spatial_factor * 17.4 / 1000
		dr_crop = np.median(np.diff(r_crop))

		# merge_dTe_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dTe_prof_multipulse_interp_target_crop)
		# merge_dTe_prof_multipulse_interp_crop_limited_target[merge_Te_prof_multipulse_interp_target_crop < 0.1] = 0
		# merge_Te_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_Te_prof_multipulse_interp_target_crop)
		# merge_Te_prof_multipulse_interp_crop_limited_target[merge_Te_prof_multipulse_interp_target_crop < 0.1] = 0
		# merge_dne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_dne_prof_multipulse_interp_crop)
		# merge_dne_prof_multipulse_interp_crop_limited_target[merge_ne_prof_multipulse_interp_target_crop < 5e-07] = 0
		# merge_ne_prof_multipulse_interp_crop_limited_target = cp.deepcopy(merge_ne_prof_multipulse_interp_target_crop)
		# merge_ne_prof_multipulse_interp_crop_limited_target[merge_ne_prof_multipulse_interp_target_crop < 5e-07] = 0
		Te_all_target = cp.deepcopy(merge_Te_prof_multipulse_interp_target_crop)
		ne_all_target = cp.deepcopy(merge_ne_prof_multipulse_interp_target_crop)


		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t,temp_r, merge_ne_prof_multipulse_interp_target_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_target*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'ne at the target (%.3g mm from it)' %target_OES_distance_at_the_target)
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, merge_dne_prof_multipulse_interp_target_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_target*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'ne uncertainty at the target (%.3g mm from it)' %target_OES_distance_at_the_target)
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_target_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_target*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Te at the target (%.3g mm from it)' %target_OES_distance_at_the_target)
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, merge_dTe_prof_multipulse_interp_target_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_target*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Te uncertainty at the target (%.3g mm from it)' %target_OES_distance_at_the_target)
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		ionisation_potential = 13.6	# eV
		dissociation_potential = 2.2	# eV
		target_adiabatic_collisional_velocity = ((Te_all_target + 5/3 *Te_all_target)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
		target_Bohm_adiabatic_flow = 0.5*ne_all_target*1e20*target_adiabatic_collisional_velocity	# 0.5 time reduction on ne in the presheath
		electron_charge = 1.60217662e-19	# C
		electron_mass = 9.10938356e-31	# kg
		electron_sound_speed = (Te_all_target/eV_to_K*boltzmann_constant_J/(electron_mass))**0.5	# m/s
		ion_sound_speed = (Te_all_target/eV_to_K*boltzmann_constant_J/(hydrogen_mass))**0.5	# m/s
		sheath_potential_drop = Te_all_target/eV_to_K*boltzmann_constant_J/electron_charge * np.log(4*ion_sound_speed/electron_sound_speed)
		sheath_potential_drop[np.isnan(sheath_potential_drop)]=0
		presheath_potential_drop = Te_all_target/eV_to_K*boltzmann_constant_J/electron_charge * np.log(0.5)
		ions_pre_sheath_acceleration = 0.4
		electrons_pre_sheath_acceleration = 0.2
		neutrals_natural_reflection = 0.6
		target_heat_flow = target_Bohm_adiabatic_flow*((2.5*boltzmann_constant_J*Te_all_target/eV_to_K-electron_charge*sheath_potential_drop-electron_charge*presheath_potential_drop)*(1-ions_pre_sheath_acceleration) + 2*Te_all_target/eV_to_K*boltzmann_constant_J*(1-electrons_pre_sheath_acceleration) + ionisation_potential + dissociation_potential*(1-neutrals_natural_reflection))
		ion_flux_target = np.sum(target_Bohm_adiabatic_flow * np.median(np.diff(time_crop))/1000 * area,axis=-1)
		label_ion_sink_at_target = 'ion sink at the target using TS Te and ne at %.3g mm from it' %target_OES_distance_at_the_target
	else:
		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		ionisation_potential = 13.6	# eV
		dissociation_potential = 2.2	# eV
		target_adiabatic_collisional_velocity = ((Te_all + 5/3 *T_Hp*eV_to_K)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
		target_Bohm_adiabatic_flow = 0.5*ne_all*1e20*target_adiabatic_collisional_velocity	# 0.5 time reduction on ne in the presheath
		electron_charge = 1.60217662e-19	# C
		electron_mass = 9.10938356e-31	# kg
		electron_sound_speed = (Te_all/eV_to_K*boltzmann_constant_J/(electron_mass))**0.5	# m/s
		ion_sound_speed = (T_Hp*boltzmann_constant_J/(hydrogen_mass))**0.5	# m/s
		sheath_potential_drop = Te_all/eV_to_K*boltzmann_constant_J/electron_charge * np.log(4*ion_sound_speed/electron_sound_speed)
		sheath_potential_drop[np.isnan(sheath_potential_drop)]=0
		presheath_potential_drop = Te_all/eV_to_K*boltzmann_constant_J/electron_charge * np.log(0.5)
		ions_pre_sheath_acceleration = 0.4
		electrons_pre_sheath_acceleration = 0.2
		neutrals_natural_reflection = 0.6
		target_heat_flow = target_Bohm_adiabatic_flow*((2.5*boltzmann_constant_J*T_Hp-electron_charge*sheath_potential_drop-electron_charge*presheath_potential_drop)*(1-ions_pre_sheath_acceleration) + 2*Te_all/eV_to_K*boltzmann_constant_J*(1-electrons_pre_sheath_acceleration) + ionisation_potential + dissociation_potential*(1-neutrals_natural_reflection))
		ion_flux_target = np.sum(target_Bohm_adiabatic_flow * np.median(np.diff(time_crop))/1000 * area,axis=-1)
		label_ion_sink_at_target = "ion sink at the target using it's own TS Te and ne at %.3g mm from it" %target_OES_distance

	if merge_ID_target in [851,85,86,87,88,89,90,91,92,93,94,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,96,97,98,74,75,76,77,78,79,80,81,82,83,84,66,67,68,69,71,72,73]:
		if merge_ID_target in [851,85,86,87,88,89,90,91,92,93,94]:
			merge_ID_target_at_the_upstream = 95
		elif merge_ID_target in [37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,96,97,98]:
			merge_ID_target_at_the_upstream = 99
		elif merge_ID_target in [74,75,76,77,78,79,80,81,82,83,84]:
			merge_ID_target_at_the_upstream = 101
		elif merge_ID_target in [66,67,68,69,71,72,73]:
			merge_ID_target_at_the_upstream = 70
		else:
			print('something wrong in definint the reference case for TS at low pressure')
			exit()

		all_j=find_index_of_file(merge_ID_target_at_the_upstream,df_settings,df_log,only_OES=True)
		target_chamber_pressure_case_upstream = []
		target_OES_distance_case_upstream = []
		for j in all_j:
			target_chamber_pressure_case_upstream.append(df_log.loc[j,['p_n [Pa]']])
			target_OES_distance_case_upstream.append(df_log.loc[j,['T_axial']])
		target_chamber_pressure_case_upstream = np.nanmean(target_chamber_pressure_case_upstream)	# Pa
		target_OES_distance_case_upstream = np.nanmean(target_OES_distance_case_upstream)	# Pa
		# Ideal gas law
		max_nH2_from_pressure_at_the_upstream = target_chamber_pressure_case_upstream/(boltzmann_constant_J*300)	# [#/m^3] I suppose ambient temp is ~ 300K

		path_where_to_save_everything_upstream = '/home/ffederic/work/Collaboratory/test/experimental_data/merge' + str(merge_ID_target_at_the_upstream)
		merge_Te_prof_multipulse_upstream = np.load(path_where_to_save_everything_upstream + '/TS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_Te_prof_multipulse']
		merge_dTe_multipulse_upstream = np.load(path_where_to_save_everything_upstream + '/TS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_dTe_multipulse']
		merge_ne_prof_multipulse_upstream = np.load(path_where_to_save_everything_upstream + '/TS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_ne_prof_multipulse']
		merge_dne_multipulse_upstream = np.load(path_where_to_save_everything_upstream + '/TS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_dne_multipulse']
		merge_time_original_upstream = np.load(path_where_to_save_everything_upstream + '/TS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_time']
		merge_time_upstream = time_shift_factor + merge_time_original_upstream

		TS_size = [-4.149230769230769056e+01, 4.416923076923076508e+01]
		TS_r = TS_size[0] + np.linspace(0, 1, 65) * (TS_size[1] - TS_size[0])
		TS_dr = np.median(np.diff(TS_r)) / 1000
		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		profile_upstream_centres = []
		profile_upstream_sigma = []
		profile_upstream_centres_score = []
		for index in range(np.shape(merge_ne_prof_multipulse_upstream)[0]):
			yy = merge_ne_prof_multipulse_upstream[index]
			yy_sigma = merge_dne_multipulse_upstream[index]
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			if np.sum(yy>0)<5:
				profile_upstream_centres.append(0)
				profile_upstream_sigma.append(10)
				profile_upstream_centres_score.append(np.max(TS_r))
				continue
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), 10, 0]
			bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
			profile_upstream_centres.append(fit[0][-1])
			profile_upstream_sigma.append(fit[0][-2])
			profile_upstream_centres_score.append(fit[1][-1, -1])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		profile_upstream_centres = np.array(profile_upstream_centres)
		profile_upstream_sigma = np.array(profile_upstream_sigma)
		profile_upstream_centres_score = np.array(profile_upstream_centres_score)
		# centre = np.nanmean(profile_upstream_centres[profile_upstream_centres_score < 1])
		centre_upstream = np.nansum(profile_upstream_centres/(profile_upstream_centres_score**1))/np.sum(1/profile_upstream_centres_score**1)
		TS_r_new_upstream = (TS_r - centre_upstream) / 1000
		print('TS profile centre at %.3gmm compared to the theoretical centre' %centre)
		# temp_r, temp_t = np.meshgrid(TS_r_new, merge_time)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_Te_prof_multipulse,vmin=0);plt.colorbar().set_label('Te [eV]');plt.pause(0.01)
		# plt.figure();plt.pcolor(temp_t,temp_r,merge_ne_prof_multipulse,vmin=0);plt.colorbar().set_label('ne [10^20 #/m^3]');plt.pause(0.01)

		# This is the mean of Te and ne weighted in their own uncertainties.
		temp1 = np.zeros_like(inverted_profiles[:, 0])
		temp2 = np.zeros_like(inverted_profiles[:, 0])
		temp3 = np.zeros_like(inverted_profiles[:, 0])
		temp4 = np.zeros_like(inverted_profiles[:, 0])
		interp_range_t = max(dt, TS_dt) * 1.5
		interp_range_r = max(dx, TS_dr) * 1.5
		weights_r = (np.zeros_like(merge_Te_prof_multipulse_upstream) + TS_r_new_upstream)
		weights_t = (((np.zeros_like(merge_Te_prof_multipulse_upstream)).T + merge_time_upstream).T)
		for i_t, value_t in enumerate(new_timesteps):
			if np.sum(np.abs(merge_time_upstream - value_t) < interp_range_t) == 0:
				continue
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(np.abs(TS_r_new_upstream) - value_r) < interp_range_r) == 0:
					continue
				elif np.sum(np.logical_and(np.abs(merge_time_upstream - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_upstream, axis=1) > 0)) == 0:
					continue
				selected_values_t = np.logical_and(np.abs(merge_time_upstream - value_t) < interp_range_t,np.sum(merge_Te_prof_multipulse_upstream, axis=1) > 0)
				selected_values_r = np.abs(np.abs(TS_r_new_upstream) - value_r) < interp_range_r
				selecte_values = (np.array([selected_values_t])).T * selected_values_r
				selecte_values[merge_Te_prof_multipulse_upstream == 0] = False
				# weights = 1/(weights_r[selecte_values]-value_r)**2 + 1/(weights_t[selecte_values]-value_t)**2
				weights = 1/((weights_t[selecte_values]-value_t)/interp_range_t)**2 + 1/((weights_r[selecte_values]-value_r)/interp_range_r)**2
				if np.sum(selecte_values) == 0:
					continue
				# temp1[i_t,i_r] = np.mean(merge_Te_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp2[i_t, i_r] = np.max(merge_dTe_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dTe_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp1[i_t, i_r] = np.sum(merge_Te_prof_multipulse_upstream[selecte_values]*weights / merge_dTe_multipulse_upstream[selecte_values]) / np.sum(weights / merge_dTe_multipulse_upstream[selecte_values])
				# temp3[i_t,i_r] = np.mean(merge_ne_prof_multipulse[selected_values_t][:,selected_values_r])
				# temp4[i_t, i_r] = np.max(merge_dne_multipulse[selected_values_t][:,selected_values_r]) / (np.sum(np.isfinite(merge_dne_multipulse[selected_values_t][:,selected_values_r])) ** 0.5)
				temp3[i_t, i_r] = np.sum(merge_ne_prof_multipulse_upstream[selecte_values]*weights / merge_dne_multipulse_upstream[selecte_values]) / np.sum(weights / merge_dne_multipulse_upstream[selecte_values])
				if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					# temp2[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dTe_multipulse[selecte_values]) ** 2)) ** 0.5
					# temp4[i_t, i_r] = (np.sum(selecte_values) / (np.sum(1 / merge_dne_multipulse[selecte_values]) ** 2)) ** 0.5
					temp2[i_t, i_r] = 1/(np.sum(1 / merge_dTe_multipulse_upstream[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse_upstream[selecte_values])/merge_dTe_multipulse_upstream[selecte_values])**2 )**0.5)
					temp4[i_t, i_r] = 1/(np.sum(1 / merge_dne_multipulse_upstream[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse_upstream[selecte_values])/merge_dne_multipulse_upstream[selecte_values])**2 )**0.5)
				else:
					# temp2_temp = 1/(np.sum(1 / merge_dTe_multipulse[selecte_values]))*(np.sum( ((temp1[i_t, i_r]-merge_Te_prof_multipulse[selecte_values])/merge_dTe_multipulse[selecte_values])**2 )**0.5)
					# temp4_temp = 1/(np.sum(1 / merge_dne_multipulse[selecte_values]))*(np.sum( ((temp3[i_t, i_r]-merge_ne_prof_multipulse[selecte_values])/merge_dne_multipulse[selecte_values])**2 )**0.5)
					temp2[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dTe_multipulse_upstream[selecte_values]),(np.max(merge_Te_prof_multipulse_upstream[selecte_values])-np.min(merge_Te_prof_multipulse_upstream[selecte_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					temp4[i_t, i_r] = max(np.sqrt(np.sum(weights**2))/np.sum(weights / merge_dne_multipulse_upstream[selecte_values]),(np.max(merge_ne_prof_multipulse_upstream[selecte_values])-np.min(merge_ne_prof_multipulse_upstream[selecte_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties

		merge_Te_prof_multipulse_interp_upstream = np.array(temp1)
		merge_dTe_prof_multipulse_interp_upstream = np.array(temp2)
		merge_ne_prof_multipulse_interp_upstream = np.array(temp3)
		merge_dne_prof_multipulse_interp_upstream = np.array(temp4)
		temp_r, temp_t = np.meshgrid(r, new_timesteps)

		# I crop to the usefull stuff
		start_time = np.abs(new_timesteps - 0).argmin()
		end_time = np.abs(new_timesteps - 1.5).argmin() + 1
		time_crop = new_timesteps[start_time:end_time]
		start_r = np.abs(r - 0).argmin()
		end_r = np.abs(r - 5).argmin() + 1
		r_crop = r[start_r:end_r]
		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		merge_Te_prof_multipulse_interp_upstream_crop = merge_Te_prof_multipulse_interp_upstream[start_time:end_time,start_r:end_r]
		merge_Te_prof_multipulse_interp_upstream_crop[merge_Te_prof_multipulse_interp_upstream_crop<0]=0
		merge_dTe_prof_multipulse_interp_upstream_crop = merge_dTe_prof_multipulse_interp_upstream[start_time:end_time, start_r:end_r]
		merge_ne_prof_multipulse_interp_upstream_crop = merge_ne_prof_multipulse_interp_upstream[start_time:end_time,start_r:end_r]
		merge_ne_prof_multipulse_interp_upstream_crop[merge_ne_prof_multipulse_interp_upstream_crop<0]=0
		merge_dne_prof_multipulse_interp_upstream_crop = merge_dne_prof_multipulse_interp_upstream[start_time:end_time, start_r:end_r]
		inverted_profiles_crop = inverted_profiles[start_time:end_time, :, start_r:end_r]
		inverted_profiles_crop[np.isnan(inverted_profiles_crop)] = 0
		inverted_profiles_sigma_crop = inverted_profiles_sigma[start_time:end_time, :, start_r:end_r]
		inverted_profiles_sigma_crop[np.isnan(inverted_profiles_sigma_crop)] = 0
		all_fits_crop = all_fits[start_time:end_time]
		# inverted_profiles_crop[inverted_profiles_crop<0] = 0


		if os.path.exists(path_where_to_save_everything_upstream + '/TS_SS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz'):
			merge_Te_prof_multipulse_SS_upstream = np.load(path_where_to_save_everything_upstream + '/TS_SS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_Te_prof_multipulse']
			merge_dTe_multipulse_SS_upstream = np.load(path_where_to_save_everything_upstream + '/TS_SS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_dTe_multipulse']
			merge_ne_prof_multipulse_SS_upstream = np.load(path_where_to_save_everything_upstream + '/TS_SS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_ne_prof_multipulse']
			merge_dne_multipulse_SS_upstream = np.load(path_where_to_save_everything_upstream + '/TS_SS_data_merge_' + str(merge_ID_target_at_the_upstream) + '.npz')['merge_dne_multipulse']

			gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
			yy = merge_ne_prof_multipulse_SS_upstream
			yy_sigma = merge_dne_multipulse_SS_upstream
			if np.sum(np.isfinite(yy_sigma))>0:
				yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
				yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			else:
				yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
			p0 = [np.max(yy), 10, 0]
			bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
			SS_profile_centres_upstream=[fit[0][-1]]
			SS_profile_sigma_upstream=[fit[0][-2]]
			SS_profile_centres_upstream_score=[fit[1][-1, -1]]
			# plt.figure();plt.plot(TS_r,merge_ne_prof_multipulse,label='ne')
			# plt.plot([fit[0][-1],fit[0][-1]],[np.max(merge_ne_prof_multipulse),np.min(merge_ne_prof_multipulse)],'--',label='ne')
			yy = merge_Te_prof_multipulse_SS_upstream
			yy_sigma = merge_dTe_multipulse_SS_upstream
			if np.sum(np.isfinite(yy_sigma))>0:
				yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
				yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			else:
				yy_sigma = np.ones_like(yy)*np.nanmin([np.nanmax(yy),1])
			p0 = [np.max(yy), 10, 0]
			bds = [[0, 0, np.min(TS_r)], [np.inf, TS_size[1], np.max(TS_r)]]
			fit = curve_fit(gauss, TS_r, yy, p0, sigma=yy_sigma, maxfev=100000, bounds=bds)
			SS_profile_centres_upstream = np.append(SS_profile_centres_upstream,[fit[0][-1]],axis=0)
			SS_profile_sigma_upstream = np.append(SS_profile_sigma_upstream,[fit[0][-2]],axis=0)
			SS_profile_centres_upstream_score = np.append(SS_profile_centres_upstream_score,[fit[1][-1, -1]],axis=0)
			SS_centre_upstream = np.nanmean(SS_profile_centres_upstream)
			SS_TS_r_new_upstream = (TS_r - SS_centre_upstream) / 1000
			print('TS profile SS_centre_upstream at %.3gmm compared to the theoretical centre' %centre)

			# This is the mean of Te and ne weighted in their own uncertainties.
			interp_range_r = max(dx, TS_dr) * 1.5
			# weights_r = TS_r_new/interp_range_r
			weights_r = SS_TS_r_new_upstream
			merge_Te_prof_multipulse_SS_upstream_interp = np.zeros_like(merge_Te_prof_multipulse_interp_upstream[ 0])
			merge_dTe_prof_multipulse_SS_upstream_interp = np.zeros_like(merge_Te_prof_multipulse_interp_upstream[ 0])
			merge_ne_prof_multipulse_SS_upstream_interp = np.zeros_like(merge_Te_prof_multipulse_interp_upstream[ 0])
			merge_dne_prof_multipulse_SS_upstream_interp = np.zeros_like(merge_Te_prof_multipulse_interp_upstream[ 0])
			for i_r, value_r in enumerate(np.abs(r)):
				if np.sum(np.abs(np.abs(SS_TS_r_new_upstream) - value_r) < interp_range_r) == 0:
					continue
				selected_values = np.abs(np.abs(SS_TS_r_new_upstream) - value_r) < interp_range_r
				selected_values[merge_Te_prof_multipulse_SS_upstream == 0] = False
				# weights = 1/np.abs(weights_r[selected_values]+1e-5)
				weights = 1/((weights_r[selected_values]-value_r)/interp_range_r)**2
				# weights = np.ones((np.sum(selected_values)))
				if np.sum(selected_values) == 0:
					continue
				merge_Te_prof_multipulse_SS_upstream_interp[i_r] = np.sum(merge_Te_prof_multipulse_SS_upstream[selected_values]*weights / merge_dTe_multipulse_SS_upstream[selected_values]) / np.sum(weights / merge_dTe_multipulse_SS_upstream[selected_values])
				merge_ne_prof_multipulse_SS_upstream_interp[i_r] = np.sum(merge_ne_prof_multipulse_SS_upstream[selected_values]*weights / merge_dne_multipulse_SS_upstream[selected_values]) / np.sum(weights / merge_dne_multipulse_SS_upstream[selected_values])
				if False: 	# suggestion from Daljeet: use the "worst case scenario" in therms of uncertainty
					merge_dTe_prof_multipulse_SS_upstream_interp[i_r] = 1/(np.sum(1 / merge_dTe_multipulse_SS_upstream[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_upstream_interp[i_r]-merge_Te_prof_multipulse_SS_upstream[selected_values])/merge_dTe_multipulse_SS_upstream[selected_values])**2 )**0.5)
					merge_dne_prof_multipulse_SS_upstream_interp[i_r] = 1/(np.sum(1 / merge_dne_multipulse_SS_upstream[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_upstream_interp[i_r]-merge_ne_prof_multipulse_SS_upstream[selected_values])/merge_dne_multipulse_SS_upstream[selected_values])**2 )**0.5)
				else:
					merge_dTe_prof_multipulse_SS_upstream_interp_temp = 1/(np.sum(1 / merge_dTe_multipulse_SS_upstream[selected_values]))*(np.sum( ((merge_Te_prof_multipulse_SS_upstream_interp[i_r]-merge_Te_prof_multipulse_SS_upstream[selected_values])/merge_dTe_multipulse_SS_upstream[selected_values])**2 )**0.5)
					merge_dne_prof_multipulse_SS_upstream_interp_temp = 1/(np.sum(1 / merge_dne_multipulse_SS_upstream[selected_values]))*(np.sum( ((merge_ne_prof_multipulse_SS_upstream_interp[i_r]-merge_ne_prof_multipulse_SS_upstream[selected_values])/merge_dne_multipulse_SS_upstream[selected_values])**2 )**0.5)
					merge_dTe_prof_multipulse_SS_upstream_interp[i_r] = max(merge_dTe_prof_multipulse_SS_upstream_interp_temp,(np.max(merge_Te_prof_multipulse_SS_upstream[selected_values])-np.min(merge_Te_prof_multipulse_SS_upstream[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
					merge_dne_prof_multipulse_SS_upstream_interp[i_r] = max(merge_dne_prof_multipulse_SS_upstream_interp_temp,(np.max(merge_ne_prof_multipulse_SS_upstream[selected_values])-np.min(merge_ne_prof_multipulse_SS_upstream[selected_values]))/2/2 )	# I enlarged the integration range by 1.5, so I reduce the sigma in the second calculation mechanism to compensate for that and get reasonables uncertainties
			temp_r, temp_t = np.meshgrid(r, new_timesteps)

			start_r = np.abs(r - 0).argmin()
			end_r = np.abs(r - 5).argmin() + 1
			r_crop = r[start_r:end_r]
			merge_Te_prof_multipulse_SS_upstream_interp_crop = merge_Te_prof_multipulse_SS_upstream_interp[start_r:end_r]
			merge_dTe_prof_multipulse_SS_upstream_interp_crop = merge_dTe_prof_multipulse_SS_upstream_interp[start_r:end_r]
			merge_ne_prof_multipulse_SS_upstream_interp_crop = merge_ne_prof_multipulse_SS_upstream_interp[start_r:end_r]
			merge_dne_prof_multipulse_SS_upstream_interp_crop = merge_dne_prof_multipulse_SS_upstream_interp[start_r:end_r]

			gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
			yy = merge_ne_prof_multipulse_SS_upstream_interp_crop
			yy_sigma = merge_dne_prof_multipulse_SS_upstream_interp_crop
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), np.max(r_crop)/2, np.min(r_crop)]
			bds = [[0, 0, np.min(r_crop)], [np.inf, np.max(r_crop), np.max(r_crop)]]
			fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma)
			SS_averaged_profile_sigma_upstream=fit[0][-2]
			# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)

			merge_Te_prof_multipulse_interp_upstream_crop[np.max(merge_ne_prof_multipulse_interp_upstream_crop,axis=1)<2] = merge_Te_prof_multipulse_SS_upstream_interp_crop
			merge_dTe_prof_multipulse_interp_upstream_crop[np.max(merge_ne_prof_multipulse_interp_upstream_crop,axis=1)<2] = np.max([merge_Te_prof_multipulse_SS_upstream_interp_crop,merge_dTe_prof_multipulse_SS_upstream_interp_crop],axis=0)
			merge_ne_prof_multipulse_interp_upstream_crop[np.max(merge_ne_prof_multipulse_interp_upstream_crop,axis=1)<2] = merge_ne_prof_multipulse_SS_upstream_interp_crop
			merge_dne_prof_multipulse_interp_upstream_crop[np.max(merge_ne_prof_multipulse_interp_upstream_crop,axis=1)<2] = np.max([merge_ne_prof_multipulse_SS_upstream_interp_crop,merge_dne_prof_multipulse_SS_upstream_interp_crop],axis=0)

		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		averaged_profile_sigma_upstream = []
		for index in range(np.shape(merge_ne_prof_multipulse_interp_upstream_crop)[0]):
			yy = merge_ne_prof_multipulse_interp_upstream_crop[index]
			yy_sigma = merge_dne_prof_multipulse_interp_upstream_crop[index]
			yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
			if (np.sum(yy>0)==0 or np.sum(yy_sigma>0)==0):
				averaged_profile_sigma_upstream.append(0)
				continue
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
			p0 = [np.max(yy), np.max(r_crop)/2, 0]
			bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
			fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma)
			averaged_profile_sigma_upstream.append(fit[0][-2])
		# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
		averaged_profile_sigma_upstream = np.array(averaged_profile_sigma_upstream)

		# x_local = xx - spatial_factor * 17.4 / 1000
		dr_crop = np.median(np.diff(r_crop))

		# merge_dTe_prof_multipulse_interp_crop_limited_upstream = cp.deepcopy(merge_dTe_prof_multipulse_interp_upstream_crop)
		# merge_dTe_prof_multipulse_interp_crop_limited_upstream[merge_Te_prof_multipulse_interp_upstream_crop < 0.1] = 0
		# merge_Te_prof_multipulse_interp_crop_limited_upstream = cp.deepcopy(merge_Te_prof_multipulse_interp_upstream_crop)
		# merge_Te_prof_multipulse_interp_crop_limited_upstream[merge_Te_prof_multipulse_interp_upstream_crop < 0.1] = 0
		# merge_dne_prof_multipulse_interp_crop_limited_upstream = cp.deepcopy(merge_dne_prof_multipulse_interp_crop)
		# merge_dne_prof_multipulse_interp_crop_limited_upstream[merge_ne_prof_multipulse_interp_upstream_crop < 5e-07] = 0
		# merge_ne_prof_multipulse_interp_crop_limited_upstream = cp.deepcopy(merge_ne_prof_multipulse_interp_upstream_crop)
		# merge_ne_prof_multipulse_interp_crop_limited_upstream[merge_ne_prof_multipulse_interp_upstream_crop < 5e-07] = 0
		Te_all_upstream = cp.deepcopy(merge_Te_prof_multipulse_interp_upstream_crop)
		ne_all_upstream = cp.deepcopy(merge_ne_prof_multipulse_interp_upstream_crop)


		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t,temp_r, ne_all_upstream, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'ne upstream (pressure %.3gPa)\n merge %.3g' %(target_chamber_pressure_case_upstream,merge_ID_target_at_the_upstream))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, merge_dne_prof_multipulse_interp_upstream_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'ne uncertainty upstream (pressure %.3gPa)\n merge %.3g' %(target_chamber_pressure_case_upstream,merge_ID_target_at_the_upstream))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, Te_all_upstream, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Te upstream (pressure %.3gPa)\n merge %.3g' %(target_chamber_pressure_case_upstream,merge_ID_target_at_the_upstream))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, merge_dTe_prof_multipulse_interp_upstream_crop, cmap='rainbow');
		plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Te uncertainty upstream (pressure %.3gPa)\n merge %.3g' %(target_chamber_pressure_case_upstream,merge_ID_target_at_the_upstream))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()

		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		heat_inflow_upstream_max = np.sum(area * ne_all_upstream*1e20 * 10000*(ionisation_potential + dissociation_potential + Te_all_upstream + 1*Te_all_upstream)/J_to_eV,axis=1)	# W
		heat_inflow_upstream_min = np.sum(area * ne_all_upstream*1e20 * 1000*(ionisation_potential + dissociation_potential + Te_all_upstream + 1*Te_all_upstream)/J_to_eV,axis=1)	# W
		label_ion_source_at_upstream = "ion source upstream using TS Te/ne at %.3g Pa, merge %.3g" %(target_chamber_pressure,merge_ID_target_at_the_upstream)
		plasma_inflow_upstream_max = np.sum(area * ne_all_upstream*1e20 * 10000,axis=1)	# W
		plasma_inflow_upstream_min = np.sum(area * ne_all_upstream*1e20 * 1000,axis=1)	# W
		upstream_adiabatic_collisional_velocity = ((Te_all_upstream + 5/3 *Te_all_upstream)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
		# specific_heat_inflow_upstream = area * ne_all_upstream*1e20 *(ionisation_potential + dissociation_potential + Te_all_upstream + 1*Te_all_upstream)/J_to_eV	# W / (m/s)
	else:
		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		heat_inflow_upstream_max = np.sum(area * ne_all*1e20 * 10000*(ionisation_potential + dissociation_potential + Te_all + nHp_ne_all*T_Hp*eV_to_K)/J_to_eV,axis=1)	# W
		heat_inflow_upstream_min = np.sum(area * ne_all*1e20 * 1000*(ionisation_potential + dissociation_potential + Te_all + nHp_ne_all*T_Hp*eV_to_K)/J_to_eV,axis=1)	# W
		label_ion_source_at_upstream = "ion source upstream using it's own TS Te and ne at %.3g Pa" %target_chamber_pressure
		plasma_inflow_upstream_max = np.sum(area * ne_all*1e20 * 10000,axis=1)	# W
		plasma_inflow_upstream_min = np.sum(area * ne_all*1e20 * 1000,axis=1)	# W
		upstream_adiabatic_collisional_velocity = ((Te_all + 5/3 *T_Hp*eV_to_K)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
		# specific_heat_inflow_upstream = area * ne_all*1e20 *(ionisation_potential + dissociation_potential + Te_all + nHp_ne_all*T_Hp*eV_to_K)/J_to_eV	# W / (m/s)
		Te_all_upstream = cp.deepcopy(Te_all)
		ne_all_upstream = cp.deepcopy(ne_all)


	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, upstream_adiabatic_collisional_velocity, cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('[m/s]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Sound speed upstream\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	power_pulse_shape_crop = interpolated_power_pulse_shape(time_crop)
	power_pulse_shape_std_crop = interpolated_power_pulse_shape_std(time_crop)
	time_source_power_crop = cp.deepcopy(time_crop)
	# homogeneous_flow_vel = 0.92*10000*power_pulse_shape_crop/heat_inflow_upstream_max
	# # homogeneous_flow_vel[np.logical_not(np.isfinite(homogeneous_flow_vel))]=0
	a_flow_vel = np.sum(0.5*area*hydrogen_mass*ne_all_upstream*1e20,axis=1)
	b_flow_vel = np.sum(5*area*Te_all_upstream/eV_to_K*boltzmann_constant_J*ne_all_upstream*1e20,axis=1) + np.sum(area*ne_all_upstream*1e20*(ionisation_potential + dissociation_potential)/J_to_eV,axis=1)
	c_flow_vel = - power_pulse_shape_crop
	p_flow_vel = b_flow_vel/a_flow_vel
	q_flow_vel = c_flow_vel/a_flow_vel
	homogeneous_flow_vel = (-q_flow_vel/2 + ((q_flow_vel**2)/4 + (p_flow_vel**3)/27)**0.5)**(1/3) - (-(-q_flow_vel/2 - ((q_flow_vel**2)/4 + (p_flow_vel**3)/27)**0.5))**(1/3)
	plt.figure(figsize=(12, 6));
	plt.plot(time_crop,homogeneous_flow_vel)
	plt.plot(time_crop,np.max(upstream_adiabatic_collisional_velocity,axis=1),'--',label='max sound speed')
	plt.grid()
	plt.legend(loc='best', fontsize='xx-small')
	# plt.xlim(left=np.min(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]),right=np.max(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]))
	# plt.xlim(left=1e0,right=1e3)
	# plt.ylim(top=2*homogeneous_flow_vel[power_pulse_shape_crop.argmax()])
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Velocity [m/s]')
	plt.title(pre_title+'Homogeneous flow velocity required for input heat flux\nto match plasma source power (92% efficiency)')	# 92% efficiency from Morgan 2014
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ne_all_upstream*1e20*((homogeneous_flow_vel**2)*hydrogen_mass + (nHp_ne_all*Te_all_upstream/eV_to_K*boltzmann_constant_J + Te_all_upstream/eV_to_K*boltzmann_constant_J).T).T, cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream total pressure homogeneous flow vel\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (ne_all_upstream.T*1e20*((homogeneous_flow_vel**2)*hydrogen_mass)).T, cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream dynamic pressure homogeneous flow vel\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (homogeneous_flow_vel/upstream_adiabatic_collisional_velocity.T).T, cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Mach number [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream mach number for homogeneous flow\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	a_flow_vel = np.sum(0.5*area*hydrogen_mass*ne_all_upstream*1e20*(upstream_adiabatic_collisional_velocity**3),axis=1)
	b_flow_vel = np.sum(5*area*Te_all_upstream/eV_to_K*boltzmann_constant_J*ne_all_upstream*1e20*upstream_adiabatic_collisional_velocity,axis=1) + np.sum(area*ne_all_upstream*1e20*(ionisation_potential + dissociation_potential)/J_to_eV*upstream_adiabatic_collisional_velocity,axis=1)
	c_flow_vel = - power_pulse_shape_crop
	p_flow_vel = b_flow_vel/a_flow_vel
	q_flow_vel = c_flow_vel/a_flow_vel
	homogeneous_mach_number = (-q_flow_vel/2 + ((q_flow_vel**2)/4 + (p_flow_vel**3)/27)**0.5)**(1/3) - (-(-q_flow_vel/2 - ((q_flow_vel**2)/4 + (p_flow_vel**3)/27)**0.5))**(1/3)
	# mach_number = 0.92*power_pulse_shape_crop/np.sum(specific_heat_inflow_upstream*upstream_adiabatic_collisional_velocity,axis=1)
	plt.figure(figsize=(12, 6));
	plt.plot(time_crop,homogeneous_mach_number)
	plt.grid()
	# plt.legend(loc='best', fontsize='xx-small')
	# plt.xlim(left=np.min(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]),right=np.max(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]))
	# plt.xlim(left=1e0,right=1e3)
	# plt.ylim(top=2*homogeneous_flow_vel[power_pulse_shape_crop.argmax()])
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Mach number [au]')
	plt.title(pre_title+'Mach number required for input heat flux\nto match plasma source power (92% efficiency)\nnon homogeneous flow speed')	# 92% efficiency from Morgan 2014
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close('all')

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T, cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('velocity [m/s]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream flow velocity for homogeneous mach number\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ne_all_upstream*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass + (nHp_ne_all*Te_all_upstream/eV_to_K*boltzmann_constant_J + Te_all_upstream/eV_to_K*boltzmann_constant_J)), cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream total pressure homogeneous mach num\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (ne_all_upstream*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass)), cmap='rainbow');
	# plt.plot(time_crop,averaged_profile_sigma_upstream*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	# plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Upstream dynamic pressure homogeneous mach num\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	area_equivalent_to_downstream_peak_total_pressure = merge_ne_prof_multipulse_interp_crop_limited*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass + (nHp_ne_all*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J))
	temp_index = np.sum(area_equivalent_to_downstream_peak_total_pressure*area,axis=-1).argmax()
	area_equivalent_to_downstream_peak_total_pressure = np.sum(area_equivalent_to_downstream_peak_total_pressure*area,axis=-1)/np.max(area_equivalent_to_downstream_peak_total_pressure,axis=-1)
	radious_equivalent_to_downstream_peak_total_pressure = (area_equivalent_to_downstream_peak_total_pressure/3.14)**0.5
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass + (nHp_ne_all*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J)), cmap='rainbow');
	plt.plot(time_crop,radious_equivalent_to_downstream_peak_total_pressure,'--',color='grey',label='radious equivalent to\nmaximum pressure\narea %.3gm2' %(area_equivalent_to_downstream_peak_total_pressure[temp_index]))
	plt.plot([time_crop[temp_index]]*2,[temp_r.min(),temp_r.max()],'--',color='grey')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Downstream total pressure homogeneous mach num\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area_equivalent_to_downstream_peak_pressure = (merge_ne_prof_multipulse_interp_crop_limited*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass))
	temp_index = np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1).argmax()
	area_equivalent_to_downstream_peak_pressure = np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1)/np.max(area_equivalent_to_downstream_peak_pressure,axis=-1)
	radious_equivalent_to_downstream_peak_pressure = (area_equivalent_to_downstream_peak_pressure/3.14)**0.5
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (merge_ne_prof_multipulse_interp_crop_limited*1e20*((((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T)**2)*hydrogen_mass)), cmap='rainbow');
	plt.plot(time_crop,radious_equivalent_to_downstream_peak_pressure,'--',color='grey',label='radious equivalent to\nmaximum pressure\narea %.3gm2' %(area_equivalent_to_downstream_peak_pressure[temp_index]))
	plt.plot([time_crop[temp_index]]*2,[temp_r.min(),temp_r.max()],'--',color='grey')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Downstream dynamic pressure homogeneous mach num\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area_equivalent_to_downstream_peak_pressure = merge_ne_prof_multipulse_interp_crop_limited*1e20*( (nHp_ne_all*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J))
	temp_index = np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1).argmax()
	temp = np.nanmax(np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1)/sum(area))
	# print((np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1)/sum(area)))
	# print(temp)
	area_equivalent_to_downstream_peak_pressure = np.sum(area_equivalent_to_downstream_peak_pressure*area,axis=-1)/np.max(area_equivalent_to_downstream_peak_pressure,axis=-1)
	radious_equivalent_to_downstream_peak_pressure = (area_equivalent_to_downstream_peak_pressure/3.14)**0.5
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited*1e20*( (nHp_ne_all*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J)), cmap='rainbow');
	plt.plot(time_crop,radious_equivalent_to_downstream_peak_pressure,'--',color='grey',label='radious equivalent to\nmaximum tot pressure\narea %.3gm2' %(area_equivalent_to_downstream_peak_pressure[temp_index]))
	plt.plot([time_crop[temp_index]]*2,[temp_r.min(),temp_r.max()],'--',color='grey')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('Pressure [Pa]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Downstream static pressure homogeneous mach num\n'+label_ion_source_at_upstream)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
	# results_summary.loc[merge_ID_target,['area_equiv_max_static_pressure','max_static_pressure']]=area_equivalent_to_downstream_peak_pressure[temp_index],np.max((merge_ne_prof_multipulse_interp_crop_limited*1e20*( (nHp_ne_all*merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J + merge_Te_prof_multipulse_interp_crop_limited/eV_to_K*boltzmann_constant_J)))[temp_index])
	results_summary.loc[merge_ID_target,['area_equiv_max_static_pressure','max_average_static_pressure']]=area_equivalent_to_downstream_peak_pressure[temp_index],temp
	results_summary.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv')

	plt.figure(figsize=(8, 5));
	plt.plot(time_crop, np.max(inverted_profiles_crop[:,0],axis=-1)/np.max(inverted_profiles_crop[:,0]),label='max emissivity n=4');
	plt.plot(time_crop, np.max(inverted_profiles_crop[:,3],axis=-1)/np.max(inverted_profiles_crop[:,3]),label='max emissivity n=7');
	plt.plot(time_crop, merge_ne_prof_multipulse_interp_crop_limited[:,0]/np.max(merge_ne_prof_multipulse_interp_crop_limited[:,0]),label='ne');
	plt.plot(time_crop, merge_Te_prof_multipulse_interp_crop_limited[:,0]/np.max(merge_Te_prof_multipulse_interp_crop_limited[:,0]),label='Te');
	plt.errorbar(time_crop, interpolated_power_pulse_shape(time_crop)/np.max(interpolated_power_pulse_shape(time_crop)),yerr=interpolated_power_pulse_shape_std(time_crop)/np.max(interpolated_power_pulse_shape(time_crop)),label='Source power');
	plt.legend(loc='best', fontsize='xx-small')
	# plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('fraction of peak')
	plt.title(pre_title+'TS / OES /ADC synchronisation check\nstandard time shift between TS and power source of %.5gms' %(shift_TS_to_power_source))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp_time_crop = time_crop[time_crop<=0.9]
	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*temp_time_crop-dt/2,temp_time_crop.max()+dt/2])
	selected = np.ones_like(merge_Te_prof_multipulse_interp_crop_limited).T

	fig, ax = plt.subplots( 3,1,figsize=(5, 7), squeeze=False, sharex=True)
	plot_index = 0
	im = ax[plot_index,0].errorbar(temp_time_crop, interpolated_power_pulse_shape(temp_time_crop)/1000,yerr=interpolated_power_pulse_shape_std(temp_time_crop)/1000)
	ax[plot_index,0].set_ylabel('source power [kW]')
	ax[plot_index,0].set_ylim(bottom=0,top=150)
	ax[plot_index,0].grid()
	plot_index +=1
	im1 = ax[plot_index,0].pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9], cmap='rainbow',vmax=9);
	#fig.colorbar(im, ax=ax[plot_index,0],orientation="vertical").set_label('temperature [eV]')
	ax[plot_index,0].set_ylabel('radial loc [m]')
	plot_index +=1
	im2 = ax[plot_index,0].pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9], cmap='rainbow',vmax=50)
	#fig.colorbar(im, ax=ax[plot_index,0],orientation="vertical").set_label('density [10^20 # m^-3]')
	ax[plot_index,0].set_ylabel('radial loc [m]')
	ax[plot_index,0].set_xlabel('time [ms]')

	fig.subplots_adjust(left=0.07, right=0.87)
	box = ax[1,0].get_position()
	pad, width = 0.02, 0.02
	cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
	# cbar1 = fig.colorbar(im1, cax=cax).set_label('Te [eV]')

	cbar1 = fig.colorbar(im1, cax=cax)
	cbar1.set_label('Te [eV]')
	CS1 = ax[1,0].contour(time_crop[time_crop<=0.9],r_crop,merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9].T,levels=np.linspace(0,merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9].max(),num=6)[1:-1],colors=['k','grey','w','b'],linewidths=2,linestyles='--')
	cbar1.add_lines(CS1)

	box = ax[2,0].get_position()
	pad, width = 0.02, 0.02
	cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
	# cbar2 = fig.colorbar(im2, cax=cax).set_label('ne [10^20 # m^-3]')

	cbar2 = fig.colorbar(im2, cax=cax)
	cbar2.set_label('ne [10^20 # m^-3]')
	CS2 = ax[2,0].contour(time_crop[time_crop<=0.9],r_crop,merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9].T,levels=np.linspace(0,merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9].max(),num=6)[1:-1],colors=['k','grey','w','b'],linewidths=2,linestyles='--')
	cbar2.add_lines(CS2)

	fig.suptitle(pre_title)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	fig, ax = plt.subplots( 3,1,figsize=(5, 7), squeeze=False, sharex=True)
	plot_index = 0
	im = ax[plot_index,0].errorbar(temp_time_crop, interpolated_power_pulse_shape(temp_time_crop)/1000,yerr=interpolated_power_pulse_shape_std(temp_time_crop)/1000)
	ax[plot_index,0].set_ylabel('source power [kW]')
	ax[plot_index,0].set_ylim(bottom=0,top=150)
	ax[plot_index,0].grid()
	plot_index +=1
	im1 = ax[plot_index,0].pcolor(temp_t, temp_r, merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9], cmap='rainbow',vmax=9);
	#fig.colorbar(im, ax=ax[plot_index,0],orientation="vertical").set_label('temperature [eV]')
	ax[plot_index,0].set_ylabel('radial loc [m]')
	plot_index +=1
	im2 = ax[plot_index,0].pcolor(temp_t, temp_r, merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9], cmap='rainbow',vmax=50)
	#fig.colorbar(im, ax=ax[plot_index,0],orientation="vertical").set_label('density [10^20 # m^-3]')
	ax[plot_index,0].set_ylabel('radial loc [m]')
	ax[plot_index,0].set_xlabel('time [ms]')

	fig.subplots_adjust(left=0.07, right=0.87)
	box = ax[1,0].get_position()
	pad, width = 0.02, 0.02
	cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
	# cbar1 = fig.colorbar(im1, cax=cax).set_label('Te [eV]')

	cbar1 = fig.colorbar(im1, cax=cax)
	cbar1.set_label('Te [eV]')
	CS1 = ax[1,0].contour(time_crop[time_crop<=0.9],r_crop,merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9].T,levels=np.linspace(0,merge_Te_prof_multipulse_interp_crop_limited[time_crop<=0.9].max(),num=6)[1:-1],colors='k',linewidths=2,linestyles='--')
	ax[1,0].clabel(CS1, inline=1, fontsize=14,fmt='%.2f',inline_spacing=9)
	# cbar1.add_lines(CS1)

	box = ax[2,0].get_position()
	pad, width = 0.02, 0.02
	cax = fig.add_axes([box.xmax + pad, box.ymin, width, box.height])
	# cbar2 = fig.colorbar(im2, cax=cax).set_label('ne [10^20 # m^-3]')

	cbar2 = fig.colorbar(im2, cax=cax)
	cbar2.set_label('ne [10^20 # m^-3]')
	CS2 = ax[2,0].contour(time_crop[time_crop<=0.9],r_crop,merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9].T,levels=np.linspace(0,merge_ne_prof_multipulse_interp_crop_limited[time_crop<=0.9].max(),num=6)[1:-1],colors='k',linewidths=2,linestyles='--')
	ax[2,0].clabel(CS2, inline=1, fontsize=14,fmt='%.1f',inline_spacing=9)
	# cbar2.add_lines(CS2)

	fig.suptitle(pre_title)
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])

	time_temp = np.arange(-0.5,1.5,0.05)
	bi_gaussian = lambda x,A1,sig1,x10,A2,sig2,x20 : (gauss(x,A1,sig1,x10) + gauss(x,(1-A1)*A2,sig2,x20) + (1-A1)*(1-A2))*np.max(interpolated_power_pulse_shape(time_temp))
	guess = [0.9,0.1,0.3,0.5,0.2,0.4]
	bds = [[0.7,0,0,0,0,0],[1,0.4,0.4,1,0.5,0.8]]
	fit = curve_fit(bi_gaussian,time_temp, interpolated_power_pulse_shape(time_temp),p0=guess,bounds=bds,maxfev=1e4)
	plt.figure(figsize=(8, 5));
	plt.errorbar(time_temp, interpolated_power_pulse_shape(time_temp)/np.max(interpolated_power_pulse_shape(time_temp)),yerr=interpolated_power_pulse_shape_std(time_temp)/np.max(interpolated_power_pulse_shape(time_temp)))
	plt.plot(time_temp, gauss(time_temp,*fit[0][0:3]),':');
	plt.plot(time_temp, gauss(time_temp,(1-fit[0][0])*fit[0][3],*fit[0][4:]),':');
	plt.plot(time_temp, np.ones_like(time_temp)*(1-fit[0][0])*(1-fit[0][3]),':');
	plt.plot(time_temp, bi_gaussian(time_temp,*fit[0])/np.max(interpolated_power_pulse_shape(time_temp)),'--',label='gauss1 A=%.3g%%,sig=%.3gms,t0=%.3gms\ngauss2 A=%.3g%%,sig=%.3gms,t0=%.3gms\nSS=%.3g%%, %% of %.3gW' %(fit[0][0],*tuple(fit[0][1:3]),(1-fit[0][0])*fit[0][3],*tuple(fit[0][4:]),(1-fit[0][0])*(1-fit[0][3]),np.max(interpolated_power_pulse_shape(time_temp))));
	plt.legend(loc='best', fontsize='xx-small')
	# plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('fraction of peak source power')
	plt.title(pre_title+'Fit of power profile with two gaussians, fraction of peak')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()



else:

	# NOTE  the first value of the "_full" arrays is for Lyman alpha. I use it only to estimate n=2 population density
	energy_difference_full = np.array([10.1988,1.88867, 2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520, 3.31917])  # eV
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	statistical_weigth_full = np.array([8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338])  # gi-gk
	einstein_coeff_full = np.array([4.6986e+00, 4.4101e-01, 8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4, 1.8927e-04]) * 1e8  # 1/s

	excitation_full = []
	for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
		if isel==0:
			temp = read_adf15(pecfile_2, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
		else:
			temp = read_adf15(pecfile, isel, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
		temp[np.isnan(temp)] = 0
		temp[np.isinf(temp)] = 0
		temp = temp.reshape((np.shape(Te_all)))
		excitation_full.append(temp)
	excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
	excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
	select = np.logical_or(np.logical_or(ne_all==0,Te_all==0),nH_ne_all==0)
	excitation_full[:,select]=0
	# excitation_full[np.logical_not(np.isfinite(excitation_full))]=0

	recombination_full = []
	for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
		if isel==0:
			temp = read_adf15(pecfile_2, 67, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
		else:
			temp = read_adf15(pecfile, isel, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
		temp[np.isnan(temp)] = 0
		temp = temp.reshape((np.shape(Te_all)))
		recombination_full.append(temp)
	recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
	recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
	select = np.logical_or(np.logical_or(ne_all==0,Te_all==0),nH_ne_all==0)
	recombination_full[:,select]=0

	multiplicative_factor_full = energy_difference_full * einstein_coeff_full / J_to_eV

	population_coefficients = ((excitation_full *  nH_ne_all).T /multiplicative_factor_full).T
	population_coefficients += ((recombination_full * nHp_ne_all).T /multiplicative_factor_full).T
	population_states_atoms = population_coefficients* ne_all**2 * 1e40

	sample = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_all.flatten(),T_Hp.flatten(),T_Hm.flatten(),ne_all.flatten()*1e20,(nHp_ne_all*ne_all).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
	temp1 = np.zeros_like(sample)
	select = np.logical_and(np.logical_and(ne_all>0,Te_all>0),nH_ne_all>0)
	temp1[select.flatten()] = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te_all[select].flatten(),T_Hp[select].flatten(),T_Hm[select].flatten(),ne_all[select].flatten()*1e20,(nHp_ne_all*ne_all)[select].flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
	temp2 = np.zeros_like(sample)
	select = np.logical_and(np.logical_and(ne_all>0,nH2p_ne_all>0),Te_all>0)
	temp2[select.flatten()] = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te_all[select].flatten(),T_H2p[select].flatten(),T_Hm[select].flatten(),ne_all[select].flatten()*1e20,(nH2p_ne_all*ne_all)[select].flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
	population_coefficients += (nHm_ne_all.flatten()*( temp1 + temp2 ).T).reshape((np.shape(population_coefficients)))

	temp = np.zeros_like(sample)
	select = np.logical_and(np.logical_and(ne_all>0,Te_all>0),nH_ne_all>0)
	temp[select.flatten()] = From_H2_pop_coeff_full_extra(np.array([Te_all[select].flatten(),ne_all[select].flatten()*1e20]).T,np.unique(excited_states_From_H2))
	population_coefficients += (nH2_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	temp = np.zeros_like(sample)
	select = np.logical_and(np.logical_and(ne_all>0,Te_all>0),nH_ne_all>0)
	temp[select.flatten()] = From_H2p_pop_coeff_full_extra(np.array([Te_all[select].flatten(),ne_all[select].flatten()*1e20]).T,np.unique(excited_states_From_H2p))
	population_coefficients += (nH2p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	# temp = From_H3p_pop_coeff_full_extra(np.array([Te_all.flatten(),ne_all.flatten()*1e20]).T,np.unique(excited_states_From_H3p))
	# population_coefficients += (nH3p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	population_states = population_coefficients * ne_all**2 * 1e40
	population_states_molecules = population_states - population_states_atoms

	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	population_states_H2 = np.zeros((15,*np.shape(Te_all)))
	for v_index in range(15):
		if v_index==0:
			population_states_H2[0] = ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			population_states_H2[v_index] = ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr

	gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
	averaged_profile_sigma = []
	averaged_profile_sigma_sigma = []
	for index in range(np.shape(ne_all)[0]):
		yy = ne_all[index]
		yy_sigma = merge_dne_prof_multipulse_interp_crop[index]
		yy_sigma[np.isnan(yy_sigma)]=np.nanmax(yy_sigma)
		if (np.sum(yy>0)==0 or np.sum(yy_sigma>0)==0):
			averaged_profile_sigma.append(0)
			averaged_profile_sigma_sigma.append(1)
			continue
		yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
		yy_sigma[yy_sigma==0]=np.nanmax(yy_sigma[yy_sigma!=0])
		p0 = [np.max(yy), np.max(r_crop)/2, 0]
		bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
		fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds, sigma=yy_sigma)
		averaged_profile_sigma.append(fit[0][-2])
		averaged_profile_sigma_sigma.append(fit[1][-2,-2])
	# plt.figure();plt.plot(TS_r,merge_Te_prof_multipulse[index]);plt.plot(TS_r,gauss(TS_r,*fit[0]));plt.pause(0.01)
	averaged_profile_sigma = np.array(averaged_profile_sigma)
	averaged_profile_sigma_sigma = np.array(averaged_profile_sigma_sigma)**0.5


	temp_r, temp_t = np.meshgrid([*r_crop-dx/2,r_crop.max()+dx/2], [*time_crop-dt/2,time_crop.max()+dt/2])
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t,temp_r, ne_all, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)\nmean %.3g' %(np.nanmean(averaged_profile_sigma[averaged_profile_sigma>0]*2.355/2)))
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'ne found with the Bayesian algorithm')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Te_all, cmap='rainbow');
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.colorbar(orientation="horizontal").set_label('temperature [eV]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Te found with the Bayesian algorithm')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, nHp_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'hydrogen ion density / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ne_all * nHp_ne_all,
			   cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'hydrogen ion density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, nH_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'neutral atomic hydrogen density / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all,vmin=max(np.max(ne_all * nH_ne_all)/1e6,1e-3), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'neutral atomic hydrogen density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, nHm_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H- / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, ne_all * nHm_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H- density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, nH2_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2 / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, ne_all * nH2_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2 density, max from chamber pressure= %.4g' % (nH2_from_pressure*1e-20)  + '10^20 # m^-3')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	figure_index += 1
	if np.sum(nH2_ne_all>0)>0:
		plt.figure(figsize=(8, 5));
		# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
		# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
		# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
		# else:
		plt.pcolor(temp_t, temp_r, ne_all * nH2_ne_all, cmap='rainbow',vmin=max(np.max(ne_all * nH2_ne_all)/1e6,1e-3), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'H2 density')
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
		plt.close()


	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, nH2p_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+ density / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, ne_all * nH2p_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+ density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, nH3p_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('[au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H3+ / electron density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	# if np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]) > 0:
	# 	plt.pcolor(temp_t, temp_r, ne_all * nH_ne_all, cmap='rainbow', norm=LogNorm(),
	# 			   vmin=0.1, vmax=np.mean(np.sort((ne_all * nH_ne_all).flatten())[-100:-10]));
	# else:
	plt.pcolor(temp_t, temp_r, ne_all * nH3p_ne_all, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('density [10^20 # m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H3+ density')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	charge_unbalance = -np.ones_like(ne_all) + nHp_ne_all - nHm_ne_all + nH2p_ne_all + nH3p_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, charge_unbalance, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Charge density/ne [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fractional charge unbalance\n(0=neutral, 1=positive charge, -1=negative charge)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, charge_unbalance,vmin=-0.2,vmax=0.2, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Charge density/ne [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Detailed fractional charge unbalance\n(0=neutral, 1=positive charge, -1=negative charge)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.plot(time_crop, np.max(inverted_profiles_crop[:,0],axis=-1)/np.max(inverted_profiles_crop[:,0]),label='max emissivity n=4');
	plt.plot(time_crop, np.max(inverted_profiles_crop[:,3],axis=-1)/np.max(inverted_profiles_crop[:,3]),label='max emissivity n=7');
	plt.plot(time_crop, ne_all[:,0]/np.max(ne_all[:,0]),label='ne');
	plt.plot(time_crop, Te_all[:,0]/np.max(Te_all[:,0]),label='Te');
	plt.errorbar(time_crop, interpolated_power_pulse_shape(time_crop)/np.max(interpolated_power_pulse_shape(time_crop)),yerr=interpolated_power_pulse_shape_std(time_crop)/np.max(interpolated_power_pulse_shape(time_crop)),label='Source power');
	plt.legend(loc='best', fontsize='xx-small')
	# plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('fraction of peak')
	plt.title(pre_title+'Bayesian T/n / OES /ADC synchronisation check\nstandard time shift between TS and power source of %.5gms' %(shift_TS_to_power_source))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	multiplicative_factor = energy_difference[4-4] * einstein_coeff[4-4] / J_to_eV
	calculated_emission_n_4_2 = population_states[4-2]*multiplicative_factor
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, calculated_emission_n_4_2, cmap='rainbow',vmax=np.median(np.sort(inverted_profiles_crop[:,0].flatten())[-5:]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Calculated emissivity Balmer line n=4>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,0]-calculated_emission_n_4_2, cmap='rainbow',vmax=np.median(np.sort((inverted_profiles_crop[:,0]-calculated_emission_n_4_2).flatten())[-5:]),vmin=np.median(np.sort((inverted_profiles_crop[:,0]-calculated_emission_n_4_2).flatten())[:5]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Difference between measured and calculated emissivity\nBalmer line n=4>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	multiplicative_factor = energy_difference[7-4] * einstein_coeff[7-4] / J_to_eV
	calculated_emission_n_7_2 = population_states[7-2]*multiplicative_factor
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, calculated_emission_n_7_2, cmap='rainbow',vmax=np.median(np.sort(inverted_profiles_crop[:,3].flatten())[-5:]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Calculated emissivity Balmer line n=7>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, inverted_profiles_crop[:,3]-calculated_emission_n_7_2, cmap='rainbow',vmax=np.median(np.sort((inverted_profiles_crop[:,3]-calculated_emission_n_7_2).flatten())[-5:]),vmin=np.median(np.sort((inverted_profiles_crop[:,3]-calculated_emission_n_7_2).flatten())[:5]))
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Difference between measured and calculated emissivity\nBalmer line n=7>2')
	# tick_locator = ticker.MaxNLocator(nbins=5)
	# cb.locator = tick_locator
	# cb.update_ticks()
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, residuals_all, cmap='rainbow', vmax=np.max(
		residuals_all[np.logical_and(time_crop > 0.3, time_crop < 0.9)][:, r_crop < 0.007]));
	plt.colorbar(orientation="horizontal").set_label('[W m^-3]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'relative residual unaccounted line emission\nsum = ' + str(np.sum(residuals_all)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	effective_ionisation_rates = (effective_ionisation_rates * (ne_all**2) *1e40* nH_ne_all).astype('float')

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_ionisation_rates, cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e4,1e19), norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('effective_ionisation_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'effective_ionisation_rates from ADAS')
	# plt.title(pre_title+'effective_ionisation_rates')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_ionisation_rates, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('effective_ionisation_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'effective_ionisation_rates from ADAS')
	# plt.title(pre_title+'effective_ionisation_rates')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	temp = read_adf11(acdfile, 'acd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in recombinations m^-3 s-1 / (# / m^3)**2
	effective_recombination_rates = (effective_recombination_rates *nHp_ne_all*1e40* (ne_all**2)).astype('float')

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_recombination_rates, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'effective_recombination_rates\n(three body + radiative) from ADAS')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_recombination_rates, cmap='rainbow',vmin=max(np.max(effective_recombination_rates)/1e4,1e19), norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('effective_recombination_rates [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'effective_recombination_rates\n(three body + radiative) from ADAS')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	fig = plt.figure()
	# fig=plt.figure(merge_ID_target * 100 + 2)
	ax = fig.add_subplot(1, 10, 1)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,Te_all.T,cmap='rainbow')
	# plt.set_sketch_params(scale=2)
	ax.set_aspect(40)
	plt.title(pre_title+'TS\nelectron temperature')
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('[eV]')
	# plt.axes().set_aspect(0.1)


	ax=fig.add_subplot(1, 10, 2)
	im=ax.pcolor(1000*temp_r.T,temp_t.T,ne_all.T,cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'TS\nelectron density')
	plt.xlabel('radial location [mm]')
	plt.yticks([])
	# plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('density [10^20 # m^-3]')

	ax=fig.add_subplot(1, 10, 3)
	im=ax.pcolor(1000*temp_r.T,temp_t.T,inverted_profiles_crop[:, 0].T,cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'OES\nn=4>2 line')
	plt.xlabel('radial location [mm]')
	plt.yticks([])
	# plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')

	ax=fig.add_subplot(1, 10, 4)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_ionisation_rates.T,cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e6,1e19), norm=LogNorm());
	ax.set_aspect(40)
	plt.title(pre_title+'effective ionisation\nrates')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')

	ax=fig.add_subplot(1, 10, 5)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,effective_recombination_rates.T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'effective recombination\nrates')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')

	ax=fig.add_subplot(1, 10, 6)
	if np.sum(nH_ne_all>0)>0:
		im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH_ne_all).T, cmap='rainbow',vmin=max(np.max(ne_all*nH_ne_all)/1e6,1e-3), norm=LogNorm());
	else:
		im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')


	ax=fig.add_subplot(1, 10, 7)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nHp_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H+ density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')


	ax=fig.add_subplot(1, 10, 8)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nHm_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H- density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')


	ax=fig.add_subplot(1, 10, 9)
	if np.sum(nH2_ne_all>0)>0:
		im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH2_ne_all).T, cmap='rainbow',vmin=max(np.max(ne_all*nH2_ne_all)/1e6,1e-3), norm=LogNorm());
	else:
		im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH2_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H2 density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')


	ax=fig.add_subplot(1, 10, 10)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH2p_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H2+ density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')


	ax=fig.add_subplot(1, 6, 6)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,(ne_all*nH3p_ne_all).T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'H3+ density')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('density [# 10^20 m^-3]')

	figure_index += 1
	# plt.pause(0.001)
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	fig = plt.figure()

	ax = fig.add_subplot(1, 5, 1)
	im = ax.pcolor(1000*temp_r.T,temp_t.T,Te_all.T,cmap='rainbow')
	# plt.set_sketch_params(scale=2)
	ax.set_aspect(40)
	plt.title(pre_title+'TS\nelectron temperature')
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('[eV]')
	# plt.axes().set_aspect(0.1)


	ax=fig.add_subplot(1, 5, 2)
	im=ax.pcolor(1000*temp_r.T,temp_t.T,ne_all.T,cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'TS\nelectron density')
	plt.xlabel('radial location [mm]')
	plt.yticks([])
	# plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('density [10^20 # m^-3]')


	ax = fig.add_subplot(1, 5, 3)
	im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_ionisation_rates.T, cmap='rainbow',vmin=max(np.max(effective_ionisation_rates)/1e6,1e19), norm=LogNorm());
	ax.set_aspect(40)
	plt.title(pre_title+'effective ionisation\nrates')
	plt.xlabel('radial location [mm]')
	plt.ylabel('time [ms]')
	# plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('effective ionisation rates [# m^-3 s-1]')

	ax = fig.add_subplot(1, 5, 4)
	im = ax.pcolor(1000 * temp_r.T, temp_t.T, effective_recombination_rates.T, cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'effective recombination\nrates')
	plt.xlabel('radial location [mm]')
	# plt.ylabel('time [ms]')
	plt.yticks([])
	fig.colorbar(im, ax=ax).set_label('effective recombination rates [# m^-3 s-1]')

	ax=fig.add_subplot(1, 5, 5)
	im=ax.pcolor(1000*temp_r.T,temp_t.T,inverted_profiles_crop[:, 0].T,cmap='rainbow')
	ax.set_aspect(40)
	plt.title(pre_title+'OES\nn=4>2 line')
	plt.xlabel('radial location [mm]')
	plt.yticks([])
	# plt.ylabel('time [ms]')
	fig.colorbar(im, ax=ax).set_label('emissivity [W m^-3]')
	figure_index += 1
	# plt.pause(0.001)
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	plt.figure(figsize=(20, 10));
	threshold_radious = 0.005
	plt.plot(time_crop, np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)), label='ionisation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)), label='recombination rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(Te_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(Te_all[:,r_crop<threshold_radious],axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.mean(Te_all[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(ne_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(ne_all[:,r_crop<threshold_radious],axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.mean(ne_all[:,r_crop<threshold_radious],axis=1))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('relative radial average [au]')
	plt.title(pre_title+'Time evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm, '+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(20, 10));
	threshold_radious = 0.015
	plt.plot(time_crop, np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1)), label='ionisation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_ionisation_rates[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1)), label='recombination rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.mean(effective_recombination_rates[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(Te_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(Te_all[:,r_crop<threshold_radious],axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.mean(Te_all[:,r_crop<threshold_radious],axis=1))));
	plt.plot(time_crop, np.mean(ne_all[:,r_crop<threshold_radious],axis=1)/np.max(np.mean(ne_all[:,r_crop<threshold_radious],axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.mean(ne_all[:,r_crop<threshold_radious],axis=1))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('relative radial average [au]')
	plt.title(pre_title+'Time evolution of the radial average, 0>r>'+  '%.3g' % (1000*threshold_radious) +' mm, '+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	ionisation_source = np.sum(volume * effective_ionisation_rates * np.median(np.diff(time_crop))/1000,axis=1)
	recombination_source = np.sum(volume * effective_recombination_rates * np.median(np.diff(time_crop))/1000,axis=1)
	inflow_min = plasma_inflow_upstream_min * np.median(np.diff(time_crop))	#steady state flow from upstream ~1km/s ballpark from CTS (Jonathan)
	inflow_max = plasma_inflow_upstream_max * np.median(np.diff(time_crop))	#peak flow from upstream ~10km/s ballpark from CTS (Jonathan)
	total_e = np.sum(volume * ne_all,axis=1) * 1e20
	total_Hp = np.sum(volume * ne_all * nHp_ne_all,axis=1) * 1e20
	total_H = np.sum(volume * ne_all * nH_ne_all,axis=1) * 1e20
	total_Hm = np.sum(volume * ne_all * nHm_ne_all,axis=1) * 1e20
	total_H2 = np.sum(volume * ne_all * nH2_ne_all,axis=1) * 1e20
	total_H2p = np.sum(volume * ne_all * nH2p_ne_all,axis=1) * 1e20
	total_H3p = np.sum(volume * ne_all * nH3p_ne_all,axis=1) * 1e20
	# plt.figure();
	fig, ax = plt.subplots(figsize=(20, 10))
	ax.fill_between(time_crop, inflow_min,inflow_max, inflow_max>=inflow_min,color='green',alpha=0.1,label='electron inflow from upstream\nflow vel 1-10km/s (CTS ballpark)')#, label='inflow, 1-10km/s (CTS)');
	ax.plot(time_crop, ion_flux_target, label=label_ion_sink_at_target);
	ax.plot(time_crop, ionisation_source, label='total ionisation source');
	ax.plot(time_crop, recombination_source, label='total recombination source');
	ax.plot(time_crop, ionisation_source - recombination_source,'kv', label='ionisation - recombination');
	ax.plot(time_crop, -(ionisation_source - recombination_source),'k^', label='recombination - ionisation');
	ax.plot(time_crop, total_e, label='total number of electrons');
	ax.plot(time_crop, total_Hp, label='total number of H+');
	ax.plot(time_crop, total_H, label='total number of H');
	ax.plot(time_crop, total_Hm, label='total number of H-');
	ax.plot(time_crop, total_H2, label='total number of H2');
	ax.plot(time_crop, total_H2p, label='total number of H2+');
	ax.plot(time_crop, total_H3p, label='total number of H3+');
	ax.legend(loc=1,fontsize='small')
	ax.set_xlabel('time from beginning of pulse [ms]')
	ax.set_ylabel('Particle balance [#]')
	ax.set_title('Particles destroyed and generated via recombination and ionisation, 0>r>'+  '%.3g' % (1000*np.max(r_crop)) +' mm, from target to '+str(target_OES_distance)+'mm from it, chamber pressure %.3gPa' %(target_chamber_pressure))
	ax.set_yscale('log')
	ax.set_ylim(np.max([ionisation_source,recombination_source,total_Hp,inflow_max])*1e-8,np.max([ionisation_source,recombination_source,total_Hp,total_H,inflow_max]))

	ax2 = ax.twinx()
	ax2.plot(time_crop, np.sum(Te_all*area,axis=1)/np.sum(area)/np.max(np.sum(Te_all*area,axis=1)/np.sum(area)),'r--', label='averaged Te TS\n(max='+'%.3g eV)' % np.max(np.sum(Te_all*area,axis=1)/np.sum(area)));
	ax2.plot(time_crop, np.max(Te_all,axis=1)/np.max(np.max(Te_all,axis=1)),'r:', label='peak Te TS\n(max='+'%.3g eV)' % np.max(np.max(Te_all,axis=1)));
	ax2.plot(time_crop, np.sum(ne_all*area,axis=1)/np.sum(area)/np.max(np.sum(ne_all*area,axis=1)/np.sum(area)),'b--', label='averaged ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(np.sum(ne_all*area,axis=1)/np.sum(area)));
	ax2.plot(time_crop, np.max(ne_all,axis=1)/np.max(np.max(ne_all,axis=1)),'b:', label='peak ne TS\n(max='+'%.3g 10^20 # m^-3)' % np.max(np.max(ne_all,axis=1)));
	ax2.set_ylabel('Te and ne divided by max value [au]')
	ax2.legend(loc=4,fontsize='small')


	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.close('all')

	if not os.path.exists(path_where_to_save_everything + mod4 + '/molecules_detailed'):
		os.makedirs(path_where_to_save_everything + mod4 + '/molecules_detailed')


	# reaction rate		e +H(nl) → H- + hν, n ≥ 1
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.1.3
	# info only on e +H(1s) → H- + hν, n ≥ 1
	Eb = 0.754	# eV
	beta = Eb/Te_all
	reaction_rate = 1.17*( 2* np.pi**(1/2) * beta**(3/2) * np.exp(beta)+1 -2*beta* hyp1f1(1,0.5,beta) )*1e-10 * 1e-6		# m^3/s
	reaction_rate[np.isnan(reaction_rate)] = 0
	e_H__Hm = ne_all*1e20 * (ne_all * nH_ne_all* 1e20 - np.sum(population_states,axis=0) ) * reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H__Hm ,vmin=max(np.max(e_H__Hm)*1e-12,np.min(e_H__Hm[e_H__Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H(1s) → H- + hν reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate		e +H- → e +H + e
	if False:
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 3.1.1
		# as mentioned I neglect:
		# e +H- → e +H(n ≥ 2) + e
		# e +H- → e +H+ + 2e
		# because negligible at low temp
		# I'm not sure if only electron energy or also H-
		# I'll assume it is electron energy because usually it is the highest
		T_e_temp = Te_all/eV_to_K	# K
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
		reaction_rate = list(map(internal,(Te_all.flatten())))
		reaction_rate = np.reshape(reaction_rate,np.shape(Te_all))
	elif True:
		# data from https://www-amdis.iaea.org/ALADDIN/
		# I do this just because it's faster to calculate and it produces the same results, given I assume a maxwellian distribution
		interpolation_ev = np.array([0.126,0.16088,0.20542,0.26229,0.3349,0.42761,0.54599,0.69714,0.89014,1.1366,1.4512,1.853,2.3659,3.0209,3.8572,4.925,6.2885,8.0294,10.252,13.09,16.714,21.341,27.25,34.793,44.425,56.724,72.428,92.478,118.08,150.77,192.51,245.8,313.85,400.73,511.67,653.32,834.19,1065.1,1360,1736.5,2217.2,2831,3614.8,4615.5,5893.2,7524.6,9607.8,12268,15664,20000])
		interpolation_reaction_rate = np.array([8.553E-12,3.559E-11,1.1517E-10,3.1067E-10,7.3556E-10,1.5868E-09,3.2008E-09,6.1402E-09,1.1315E-08,2.0133E-08,3.464E-08,5.7584E-08,9.2294E-08,1.4225E-07,2.1026E-07,2.9746E-07,4.0221E-07,5.1956E-07,6.4157E-07,7.585E-07,8.6067E-07,9.4036E-07,9.931E-07,1.0181E-06,1.0178E-06,9.967E-07,9.6031E-07,9.1408E-07,8.6272E-07,8.0988E-07,7.5805E-07,7.0876E-07,6.6267E-07,6.1989E-07,5.801E-07,5.4278E-07,5.073E-07,4.7309E-07,4.397E-07,4.0685E-07,3.7451E-07,3.4285E-07,3.1221E-07,2.8305E-07,2.5587E-07,2.3113E-07,2.0915E-07,1.9015E-07,1.7413E-07,1.6093E-07]) * 1e-6		# m^3/s
		interpolator_reaction_rate = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_reaction_rate),fill_value='extrapolate')
		reaction_rate = np.zeros_like(Te_all)
		reaction_rate[Te_all>0] = np.exp(interpolator_reaction_rate(np.log(Te_all[Te_all>0])))
		reaction_rate[reaction_rate<0] = 0
	e_Hm__e_H_e = ne_all*1e20 *nHm_ne_all*ne_all*1e20*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_Hm__e_H_e ,vmin=max(np.max(e_Hm__e_H_e)*1e-12,np.min(e_Hm__e_H_e[e_Hm__e_H_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H- → e +H + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hp_Hm__H_2_H = ne_all*1e20 *nHm_ne_all*ne_all*1e20*nHp_ne_all*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_Hm__H_2_H ,vmin=max(np.max(Hp_Hm__H_2_H)*1e-12,np.min(Hp_Hm__H_2_H[Hp_Hm__H_2_H>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H- →H(2) +H(1s) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
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
	Hp_Hm__H_3_H = ne_all*1e20 *nHm_ne_all*ne_all*1e20*nHp_ne_all*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_Hm__H_3_H ,vmin=max(np.max(Hp_Hm__H_3_H)*1e-12,np.min(Hp_Hm__H_3_H[Hp_Hm__H_3_H>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H- →H(3) +H(1s) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hp_Hm__H2p_e = ne_all*1e20 *nHm_ne_all*ne_all*1e20*nHp_ne_all*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_Hm__H2p_e ,vmin=max(np.max(Hp_Hm__H2p_e)*1e-12,np.min(Hp_Hm__H2p_e[Hp_Hm__H2p_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H- → H2+(v) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hm_H1s__H1s_H1s_e = ne_all*1e20 *nHm_ne_all*(ne_all*1e20*nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hm_H1s__H1s_H1s_e ,vmin=max(np.max(Hm_H1s__H1s_H1s_e)*1e-12,np.min(Hm_H1s__H1s_H1s_e[Hm_H1s__H1s_H1s_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H- +H(1s) →H2-(B2Σ+g ) →H(1s) +H(1s) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hm_H1s__H2_v_e = ne_all*1e20 *nHm_ne_all*(ne_all*1e20*nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hm_H1s__H2_v_e ,vmin=max(np.max(Hm_H1s__H2_v_e)*1e-12,np.min(Hm_H1s__H2_v_e[Hm_H1s__H2_v_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H- +H(1s) →H2-(X2Σ+u ; B2Σ+g ) →H2(X1Σ+ g ; v) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hp_H_H__Hp_H2 = ne_all*1e20 * nHp_ne_all*((ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))**2)*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_H_H__Hp_H2 ,vmin=max(np.max(Hp_H_H__Hp_H2)*1e-12,np.min(Hp_H_H__Hp_H2[Hp_H_H__Hp_H2>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H(1s) +H(1s) →H+ +H2(v) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate		H +H +H →H +H2(v)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.3.4
	# valid for temperature up to 20000-30000K (~2eV)
	reaction_rate = 1.65/T_H *1e-30 * 1e-6 * 1e-6		# m^6/s
	reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H-20000)).argmin()]
	reaction_rate[T_H>20000]=reaction_rate_max
	H_H_H__H_H2 = ((ne_all*1e20 *nH_ne_all)**3)*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H_H_H__H_H2 ,vmin=max(np.max(H_H_H__H_H2)*1e-12,np.min(H_H_H__H_H2[H_H_H__H_H2>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H(1s) +H(1s) →H+ +H2(v) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate		e+H2(X1Σg;v) → e+H2∗(N1,3Λσ;eps) → e+H(1s)+H(nl)
	if False:
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
		T_e_temp = Te_all/eV_to_K	# K
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
		e_H2_X1Σg_x__e_H_H = np.zeros((15,*np.shape(ne_all)))
		e_H2_X1Σg_x__e_H_H[0] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
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
				e_H2_X1Σg_x__e_H_H[v_index+1] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index+1)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
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
					e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
				else:
					e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr

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
			e_H2_X1Σg_x__e_H_H[0] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
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
				e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
		e_H2_X1Σg_x__e_H_H = np.sum(e_H2_X1Σg_x__e_H_H,axis=0)
	elif True:
		# from AMJUEL
		# 4.10 ReactionReaction 2.2.5g e +H2 → e +H +H
		# this RR is computed for TH2=Te, that is very not true.
		# BUT this calculation is much simpler so I use it.
		coefficients_all = [[-2.702372540584e+01,-3.152103191633e-03,5.990692171729e-03,-3.151252835426e-03,7.457309144890e-04,-9.238664007853e-05,6.222557542845e-06,-2.160024578659e-07,3.028755759836e-09],[1.081756417479e+01,-1.487216964825e-02,1.417396532101e-02,-4.689911797083e-03,7.180338663163e-04,-5.502798587526e-05,1.983066081752e-06,-2.207639762507e-08,-2.116339335271e-10],[-5.368872027676e+00,5.419787589654e-03,-1.747268613395e-02,9.532963297450e-03,-2.196705622859e-03,2.611447288152e-04,-1.695536960581e-05,5.737375510694e-07,-7.940900078995e-09],[1.340684229143e+00,1.058157580038e-02,-3.446019122786e-03,-7.032769815599e-04,4.427959286553e-04,-7.370484189164e-05,5.746786010618e-06,-2.182085196303e-07,3.264045809897e-09],[-1.561644923145e-01,-3.847438570333e-03,3.571477356851e-03,-1.103305795473e-03,1.476712517858e-04,-8.461162952132e-06,9.757111870171e-08,8.130014050833e-09,-2.234996157750e-10],[-1.444731533894e-04,-3.194532513126e-04,-2.987368098475e-04,2.092094838648e-04,-4.339352509941e-05,4.009328699469e-06,-1.762651912129e-07,3.357860444624e-09,-1.857322587267e-11],[2.117693926546e-03,2.679309814780e-04,-1.037559373832e-04,7.297053580368e-06,1.454171585421e-06,-2.251616910293e-07,9.191700327811e-09,-2.052366968228e-11,-3.567738654108e-12],[-2.143738340207e-04,-3.539232757385e-05,1.909399233821e-05,-3.819368125069e-06,3.754063159414e-07,-2.441872829462e-08,1.437490161488e-09,-6.172308568891e-11,1.104905484620e-12],[6.979740947331e-06,1.462031952352e-06,-8.858634506391e-07,2.099830142707e-07,-2.606862169776e-08,2.039813579349e-09,-1.113483084607e-10,3.859777100010e-12,-5.909099891913e-14]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^2
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		reaction_rate[Te_all<min(0.1,Te_all.max()/2)]=0
		e_H2_X1Σg_x__e_H_H = reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2_X1Σg_x__e_H_H ,vmin=max(np.max(e_H2_X1Σg_x__e_H_H)*1e-12,np.min(e_H2_X1Σg_x__e_H_H[e_H2_X1Σg_x__e_H_H>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e+H2(X1Σg;v) → e+H2∗(N1,3Λσ;eps) → e+H(1s)+H(nl), v=0-14, singlets and triplets reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e
	if False:
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
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		e_H2N13Λσ__e_H2pX2Σg_e = np.zeros((15,*np.shape(ne_all)))
		# coefficients: ∆Ev(eV )  NOTE except for v=0 the rest are all guesses
		coefficients_all = np.array([15.42,13.8074509803922,12.6988235294118,11.7917647058824,11.0358823529412,10.3807843137255,9.87686274509804,9.32254901960784,8.86901960784314,8.41549019607843,8.06274509803922,7.76039215686274,7.45803921568627,7.1556862745098])
		for v_index,coefficients in enumerate(coefficients_all):
			x = e_energy/coefficients
			cross_section = ((coefficients/coefficients_all[0])**1.15) * 1.828/x * ((1-1/(x**0.92))**2.19) * np.log(2.05*e_energy) * 1e-16 * 1e-4	# m^2
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
			if v_index==0:
				e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
		e_H2N13Λσ__e_H2pX2Σg_e = np.sum(e_H2N13Λσ__e_H2pX2Σg_e,axis=0)
	elif True:
		# from AMJUEL
		# 4.11 Reaction 2.2.9 e +H2 → 2e +H2+
		# I use this because the to calculate the cross section i use standard H2 vibrational states already in AMJUEL, but this method is much faster
		coefficients_all = [[-35.74773783577,0.3470247049909,-0.09683166540937,0.00195957627625,0.00247936111919,-0.0001196632952666,-0.00001862956119592,0.000001669867158509,-0.000000036737362782],[17.69208985507,-1.311169841222,0.4700486215943,-0.05521175478827,-0.002689651616933,0.0007308915874002,-0.00002920560755694,-0.0000003148831240316,0.00000002514856386324],[-8.291764008409,1.591701525694,-0.5814996025336,0.09160898084105,-0.004770789631868,0.00001994775632224,-0.000007511552245648,0.000001089689676313,-0.00000002920863498031],[2.55571234724,-0.8625268584825,0.2612076696684,-0.03686525285376,0.001945480608139,-0.00003690918356665,0.000004836340453567,-0.0000004165748666929,0.000000009265898224345],[-0.5370404654062,0.2375816996323,-0.0416590877817,0.001732469114063,0.0003693513203529,-0.00004931268184607,0.000002727501534044,-0.0000001081027384449,0.000000002420509440644],[0.07443307905391,-0.03322214182214,-0.002351235556666,0.001723053881691,-0.0002096625925098,0.00001358575558294,-0.000001041586202167,0.00000006928574330531,-0.000000001746656185835],[-0.006391785721973,0.00186255427819,0.001540632467396,-0.0003547150770477,0.00001392157055273,0.000001047463944093,0.00000001513510667993,-0.000000009915499708242,0.0000000003298173891188],[0.0003001729098239,0.00003497202259366,-0.0001742029226138,0.00002296551698214,0.000002357520372192,-0.000000530608551395,0.00000002223137028418,0.00000000033401693098,-0.00000000002560542889504],[-0.000005607182991432,-0.000005779550092391,0.000006495742927455,-0.0000003040011333889,-0.0000002361542565281,0.00000003655056080262,-0.000000001771478792301,0.00000000001334615260635,0.0000000000006831564719957]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^3/s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		# I have to add external bouds to avoid the extrapolation to stop working and give me crazy values
		reaction_rate[Te_all<min(0.5,Te_all.max()/2)]=0
		e_H2N13Λσ__e_H2pX2Σg_e = reaction_rate * (ne_all**2)*1e40*nH2_ne_all

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2N13Λσ__e_H2pX2Σg_e ,vmin=max(np.max(e_H2N13Λσ__e_H2pX2Σg_e)*1e-12,np.min(e_H2N13Λσ__e_H2pX2Σg_e[e_H2N13Λσ__e_H2pX2Σg_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e, v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate
	# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; eps’) +e → H+ +H(1s) + 2e
	# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(B2Σu+; eps’) +e → H+ +H(1s) + 2e
	if False:
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
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_H2N13Λσ__Hp_H_2e = np.zeros((15,*np.shape(ne_all)))
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
				e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
		e_H2N13Λσ__Hp_H_2e = np.sum(e_H2N13Λσ__Hp_H_2e,axis=0)
	elif True:
		# from AMJUEL
		# 4.12 Reaction 2.2.10 e +H2 → 2e +H +H+
		# I use this because the to calculate the cross section i use standard H2 vibrational states already in AMJUEL, but this method is much faster
		coefficients_all = [[-37.93749300315,-0.3333162972531,0.1849601203843,-0.08803945197107,0.02205180180735,-0.002852568161901,0.0001942314738448,-0.000006597388255594,0.00000008798544848606],[12.80249398154,1.028969438485,-0.3271855492638,0.1305597441611,-0.0340843982191,0.004591924060066,-0.0003167471002157,0.00001070920193931,-0.0000001408139742113],[-3.77814855314,-1.415561059533,0.2928509524911,-0.07425165688158,0.02028424685287,-0.003042376564749,0.0002279124955373,-0.000008197224564797,0.0000001130682076163],[0.2499987501522,1.032922656537,-0.1580288004759,0.009934702707539,-0.002450845732158,0.0005716646876513,-0.00005339115778704,0.000002135848413694,-0.00000003072223247387],[0.2480574522949,-0.4372934216955,0.06448433196301,0.00122922293263,-0.0009281410519553,0.00005946235618034,-0.00000008758032156912,-0.00000007270955072707,0.000000001100087131523],[-0.09960628182831,0.1092652428162,-0.01782307798975,0.0001192181214757,0.0002310636556641,-0.00002492990725967,0.000001217600444191,-0.00000003624263301602,0.0000000006139167092128],[0.01709129400742,-0.01574889001363,0.002865310743302,-0.0001700396064727,-0.000001502644504654,0.0000003297869416435,0.0000000006572135289627,0.0000000004269190108005,-0.00000000003666090917669],[-0.001435304503973,0.001203823111704,-0.0002350465388313,0.00002507288189894,-0.000003077975735212,0.0000003748299687254,-0.00000002613600078122,0.0000000008263175463927,-0.000000000008509179497022],[0.00004808639828229,-0.00003761591649539,0.000007490531472388,-0.000001077314971617,0.0000001950247963978,-0.00000002569729600929,0.000000001804377780165,-0.00000000006031847199601,0.0000000000007416020205748]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^3/s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		reaction_rate[Te_all<min(1,Te_all.max()/2)]=0
		e_H2N13Λσ__Hp_H_2e = reaction_rate * (ne_all**2)*1e40*nH2_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2N13Λσ__Hp_H_2e ,vmin=max(np.max(e_H2N13Λσ__Hp_H_2e)*1e-12,np.min(e_H2N13Λσ__Hp_H_2e[e_H2N13Λσ__Hp_H_2e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e, v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		e_H2X1Σg__Hm_H1s = np.zeros((15,*np.shape(ne_all)))
		# coefficients: E th,v (eV ), σ v (10 -16 cm 2 )
		coefficients_all = np.array([[3.72,3.22e-5],[3.21,5.18e-4],[2.72,4.16e-3],[2.26,2.20e-2],[1.83,1.22e-1],[1.43,4.53e-1],[1.36,1.51],[0.713,4.48],[0.397,10.1],[0.113,13.9],[-0.139,11.8],[-0.354,8.87],[-0.529,7.11],[-0.659,5],[-0.736,3.35]])
		for v_index,coefficients in enumerate(coefficients_all):
			cross_section =  coefficients[1]*np.exp(-(e_energy-np.abs(coefficients[0]))/0.45) * 1e-16 * 1e-4	# m^2
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
			if v_index==0:
				e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
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
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		e_H2X1Σg__Hm_H1s = np.zeros((15,*np.shape(ne_all)))
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
				reaction_rate = np.zeros_like(Te_all)
				reaction_rate[Te_all>0] = np.exp(interpolator_reaction_rate(np.log(Te_all[Te_all>0])))
				reaction_rate[reaction_rate<0] = 0
			if v_index==0:
				e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	e_H2X1Σg__Hm_H1s = np.sum(e_H2X1Σg__Hm_H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2X1Σg__Hm_H1s , cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2(X1Σ+g ; v) → H- +H(1s), v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	T_e_temp = Te_all/eV_to_K	# K
	T_e_temp[T_e_temp==0]=300	# ambient temperature
	e_H2X1Σg__e_H1s_H1s = np.zeros((15,*np.shape(ne_all)))
	# via H2- ground state
	# coefficients: a1,..,a6
	coefficients_all = np.array([[-50.862,0.92494,-28.102,-4.5231e-2,0.46439,0.8795],[-48.125,0.91626,-24.873,-4.9898e-2,0.45288,0.87604],[-41.218,0.96738,-23.167,-4.8546e-2,-1.7222,0.19858],[-37.185,0.96391,-21.264,-5.1701e-2,-1.8121,0.19281],[-35.397,0.85294,-18.452,-6.522e-2,-0.56595,8.8997e-2],[-33.861,0.9301,-20.852,-3.016e-2,5.561,0.45548],[-23.751,0.9402,-19.626,-3.276e-2,-0.3982,1.58655],[-19.988,0.83369,-18.7,-3.552e-2,-0.38065,1.74205],[-18.278,0.8204,-17.754,-4.453e-2,-0.10045,2.5025],[-13.589,0.7021,-16.85,-5.012e-2,-0.77502,0.3423],[-11.504,0.84513,-14.603,-6.775e2,-3.2615,0.13666]])
	for v_index,coefficients in enumerate(coefficients_all):
		T_scaled = T_e_temp/1000
		reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) 		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	# via H2- excited state
	# coefficients: a1,..,a6
	coefficients_all = np.array([[-15.76,-0.052659,-84.679,1.0414,-8.2933,0.18756],[-16.162,-0.0498200333333333,-74.3906666666667,1.01586,-6.15236666666667,0.227996666666667],[-16.564,-0.0469810666666667,-64.1023333333333,0.99032,-4.01143333333333,0.268433333333333],[-16.966,-0.0441421,-53.814,0.96478,-1.8705,0.30887],[-16.1206666666667,-0.0490894,-47.1276666666667,0.94422,-1.72766666666667,0.208215033333333],[-15.2753333333333,-0.0540367,-40.4413333333333,0.92366,-1.58483333333333,0.107560066666667],[-14.43,-0.058984,-33.755,0.9031,-1.442,0.0069051],[-14.4276666666667,-0.0575976666666667,-28.0646666666667,0.897233333333333,-1.5259,0.00691206666666667],[-14.4253333333333,-0.0562113333333333,-22.3743333333333,0.891366666666667,-1.6098,0.00691903333333333],[-14.423,-0.054825,-16.684,0.8855,-1.6937,0.006926],[-16.2556666666667,-0.0396174,-26.4876666666667,0.799833333333333,13.6192,0.0993073333333334],[-18.0883333333333,-0.0244098,-36.2913333333333,0.714166666666667,28.9321,0.191688666666667],[-19.921,-0.0092022,-46.095,0.6285,44.245,0.28407]])
	for v_index,coefficients in enumerate(coefficients_all):
		T_scaled = T_e_temp/1000
		reaction_rate = 1e-6 * np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) 		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	e_H2X1Σg__e_H1s_H1s = np.sum(e_H2X1Σg__e_H1s_H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2X1Σg__e_H1s_H1s  ,vmin=max(np.max(e_H2X1Σg__e_H1s_H1s)*1e-12,np.min(e_H2X1Σg__e_H1s_H1s[e_H2X1Σg__e_H1s_H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2(X1Σ+g;v) →H2-(X2Σ+u)→e +H2(X1Σ+;eps)→ e +H(1s) +H(1s), v=0-12\ne +H2(X1Σ+g;v) →H2-(B2Σ+g) → e +H2(b3Σ+;eps)→ e +H(1s) +H(1s) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate H+ +H2(X1Σ+g ; v) →H(1s) +H2+(X2Σ+ g ; v)
	if False:
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
		Hp_H2X1Σg__H1s_H2pX2Σg = np.zeros((15,*np.shape(ne_all)))
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
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
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
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
		Hp_H2X1Σg__H1s_H2pX2Σg = np.sum(Hp_H2X1Σg__H1s_H2pX2Σg,axis=0)
	elif True:	# 27/05/2020
		# from AMJUEL
		# 3.28 Reaction 3.2.3 p +H2(v) →H +H+
		coefficients_all = [[-2.133104980000e+1,2.961905900000e-1,-2.876892150000e-2,-3.323271590000e-2,7.234558340000e-3,2.940230100000e-4,-8.005031610000e-5,0.000000000000e+0,0.000000000000e+0],[2.308461720000e+0,-1.064800460000e+0,2.310120950000e-1,6.809382980000e-2,-4.241210420000e-2,8.271152020000e-3,-6.275988100000e-4,0.000000000000e+0,0.000000000000e+0],[-2.026151710000e+0,1.142806740000e+0,-2.621943460000e-1,-6.877694430000e-2,4.012716970000e-2,-6.143307540000e-3,3.233852920000e-4,0.000000000000e+0,0.000000000000e+0],[1.648000330000e-1,-4.675786500000e-1,1.242261910000e-1,1.774294860000e-2,-1.157658350000e-2,1.311061300000e-3,-1.125957730000e-5,0.000000000000e+0,0.000000000000e+0],[1.651993580000e-1,5.766584690000e-2,-3.659922760000e-2,7.083346120000e-3,3.403537010000e-4,-2.752152790000e-4,2.225165850000e-5,0.000000000000e+0,0.000000000000e+0],[-2.598458070000e-2,1.349144350000e-2,8.871659800000e-3,-5.231162040000e-3,3.324241650000e-4,1.985585660000e-4,-2.813630850000e-5,0.000000000000e+0,0.000000000000e+0],[-4.330453510000e-3,-5.246404340000e-3,-1.636107180000e-3,1.242023150000e-3,-4.524774630000e-5,-6.369415730000e-5,8.679231940000e-6,0.000000000000e+0,0.000000000000e+0],[1.187405610000e-3,6.281964210000e-4,1.740000360000e-4,-1.337853740000e-4,6.784609160000e-7,8.284840740000e-6,-1.075372230000e-6,0.000000000000e+0,0.000000000000e+0],[-6.897815380000e-5,-2.667160440000e-5,-7.528040300000e-6,5.516687380000e-6,1.140207820000e-7,-3.837975410000e-7,4.793672020000e-8,0.000000000000e+0,0.000000000000e+0]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(3/2*T_H2*eV_to_K),np.log(T_Hp*eV_to_K),coefficients_all)) * 1e-6	# m^2
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		Hp_H2X1Σg__H1s_H2pX2Σg = reaction_rate * nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_H2X1Σg__H1s_H2pX2Σg ,vmin=max(np.max(Hp_H2X1Σg__H1s_H2pX2Σg)*1e-12,np.min(Hp_H2X1Σg__H1s_H2pX2Σg[Hp_H2X1Σg__H1s_H2pX2Σg>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H2(X1Σ+g ; v) →H(1s) +H2+(X2Σ+ g ; v), v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hp_H2X1Σg__Hp_H1s_H1s = np.zeros((15,*np.shape(ne_all)))
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
			Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *nHp_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	Hp_H2X1Σg__Hp_H1s_H1s = np.sum(Hp_H2X1Σg__Hp_H1s_H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_H2X1Σg__Hp_H1s_H1s ,vmin=max(np.max(Hp_H2X1Σg__Hp_H1s_H1s)*1e-12,np.min(Hp_H2X1Σg__Hp_H1s_H1s[Hp_H2X1Σg__Hp_H1s_H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H2(X1Σ+ g ; v) → H+ +H(1s) +H(1s), v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
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
		Hm_H2v__H_H2v_e = reaction_rate *nHm_ne_all* ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
	else:
		# data from https://www-amdis.iaea.org/ALADDIN/
		# no info on the vibrational state, I assume it is valid for all
		interpolation_ev = np.array([2.3183E+00,3.2014E+00,4.4210E+00,6.1052E+00,8.4309E+00,1.1643E+01,1.6078E+01,2.2203E+01,3.0661E+01,4.2341E+01,5.8471E+01,8.0746E+01,1.1151E+02,1.5398E+02,2.1264E+02,2.9365E+02,4.0552E+02,5.6000E+02,7.7333E+02,1.0679E+03,1.4748E+03,2.0366E+03,2.8124E+03,3.8838E+03,5.3633E+03,7.4065E+03,1.0228E+04,1.4124E+04,1.9505E+04,2.6935E+04,3.7197E+04,5.1367E+04,7.0935E+04,9.7957E+04,1.3527E+05,1.8681E+05,2.5797E+05,3.5624E+05,4.9196E+05,6.7937E+05,9.3817E+05,1.2956E+06,1.7891E+06,2.4707E+06,3.4119E+06,4.7116E+06,6.5065E+06,8.9852E+06,1.2408E+07,1.7135E+07])	# eV
		interpolation_cross_section = np.array([9.5063E-17,1.5011E-16,2.1010E-16,2.6792E-16,3.1859E-16,3.6021E-16,3.9345E-16,4.2053E-16,4.4425E-16,4.6737E-16,4.9230E-16,5.2099E-16,5.5498E-16,5.9540E-16,6.4289E-16,6.9763E-16,7.5912E-16,8.2608E-16,8.9629E-16,9.6648E-16,1.0324E-15,1.0892E-15,1.1314E-15,1.1542E-15,1.1537E-15,1.1280E-15,1.0773E-15,1.0041E-15,9.1306E-16,8.1021E-16,7.0205E-16,5.9470E-16,4.9323E-16,4.0127E-16,3.2091E-16,2.5285E-16,1.9672E-16,1.5146E-16,1.1562E-16,8.7634E-17,6.6012E-17,4.9419E-17,3.6733E-17,2.7048E-17,1.9657E-17,1.4021E-17,9.7429E-18,6.5296E-18,4.1680E-18,2.4949E-18]) * 1e-4		# m^2
		interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
		cross_section=np.exp(interpolator_cross_section(np.log(Hm_energy)))
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*Hm_velocity*Hm_velocity_PDF,axis=-1)* np.mean(np.diff(Hm_velocity))		# m^3 / s
		Hm_H2v__H_H2v_e = reaction_rate *nHm_ne_all* (ne_all**2)*1e40 * nH2_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hm_H2v__H_H2v_e ,vmin=max(np.max(Hm_H2v__H_H2v_e)*1e-12,np.min(Hm_H2v__H_H2v_e[Hm_H2v__H_H2v_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H- +H2(v) →H +H2(v") + e, v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate H(1s) +H2(v) → H(1s) + 2H(1s)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 6.2.2
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	H1s_H2v__H1s_2H1s = np.zeros((15,*np.shape(ne_all)))
	# coefficients: a1,..,a5
	coefficients_all = np.array([[2.06964e1,7.32149e7,1.7466,4.75874e3,-9.42775e-1],[2.05788e1,4.32679e7,1.6852,1.91812e3,-8.16838e-1],[2.05183e1,5.15169e7,1.73345,3.09006e3,-8.88414e-1],[2.04460e1,1.87116e8,1.87951,9.04442e3,-9.78327e-1],[2.03608e1,4.93688e8,1.99499,2.32656e4,-1.06294],[2.02426e1,1.80194e8,1.92249,1.28777e4,-1.02713],[2.00161e1,2.96945e5,1.31044,9.55214e2,-1.07546],[1.98954e1,4.53104e5,1.37055,3.88065e2,-8.71521e-1],[1.97543e1,5.13174e5,1.39819,3.54272e2,-8.07563e-1],[1.97464e1,9.47230e4,1.24048,2.28283e2,-8.51591e-1],[1.95900e1,6.43990e4,1.22211,1.16196e2,-7.35645e-1],[1.94937e1,3.49017e4,1.20883,1.26329e2,-8.15130e-1],[1.90708e1,1.05971e5,9.91646e-1,1.05518e2,-1.93837e-1],[1.89718e1,7.76046e5,7.84577e-1,1.31409e3,-1.00479e-2],[1.87530e1,5.81508e5,7.35904e-1,1.69328e3,4.47757e-3]])
	for v_index,coefficients in enumerate(coefficients_all):
		reaction_rate = np.exp(-coefficients[0] -coefficients[1]/(T_H**coefficients[2]*(1+coefficients[3]*(T_H**coefficients[4])) )) *1e-6 		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			H1s_H2v__H1s_2H1s[v_index] += reaction_rate * (ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0)) * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			H1s_H2v__H1s_2H1s[v_index] += reaction_rate * (ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0)) * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	H1s_H2v__H1s_2H1s = np.sum(H1s_H2v__H1s_2H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H1s_H2v__H1s_2H1s,vmin=max(np.max(H1s_H2v__H1s_2H1s)*1e-12,np.min(H1s_H2v__H1s_2H1s[H1s_H2v__H1s_2H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H(1s) +H2(v) → H(1s) + 2H(1s), v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
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
	H2v0_H2v__H2v0_2H1s = np.zeros((15,*np.shape(ne_all)))
	for v_index,coefficients in enumerate(coefficients_all):
		reaction_rate_0 = 1.3*(1 +0.023*v_index + 1.93e-5 * (v_index**4.75) + 2.85e-24 *(v_index**21.6)  )
		T_0 = (7.47 - 0.322 *v_index) * 1e3
		reaction_rate = reaction_rate_0 * np.exp(-T_0/T_H2) * 1e-10 *1e-6 		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H2-20000)).argmin()]
		reaction_rate[T_H2>20000]=reaction_rate_max
		if v_index==0:
			H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
		else:
			H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	H2v0_H2v__H2v0_2H1s = np.sum(H2v0_H2v__H2v0_2H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H2v0_H2v__H2v0_2H1s,vmin=max(np.max(H2v0_H2v__H2v0_2H1s)*1e-12,np.min(H2v0_H2v__H2v0_2H1s[H2v0_H2v__H2v0_2H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2(v1 = 0) +H2(v) →H2(v1 = 0) + 2H(1s), v=0-14 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction reates:
	# e +H2+(v) → e +H2+(2pσu) → e +H+ +H(1s)
	# e +H2+(v) → e +H2+(2pπu) → e +H+ +H(n = 2)
	# e +H2+(v) →H2∗∗[(2pσu)2] → e +H+ +H(1s)
	# e +H2+(v) →H2∗Ryd(N1,3Λσ; eps) → e +H+ +H(1s)
	if False:
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 7.1.2a
		# valid from 0.01 to 2000 eV
		# I'm not sure if only electron energy or also H2+
		# I'll assume it is electron energy because usually it is the highest
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		cross_section = 13.2*np.log(np.exp(1) + 2.55e-4 * e_energy)/( (e_energy**0.31)*(1+0.017*(e_energy**0.76)))* 1e-16 * 1e-4
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(Te_all-0.01)).argmin()]
		cross_section[e_energy<0.01]=cross_section_min
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__XXX__e_Hp_H1s = ne_all*1e20 *nH2p_ne_all*ne_all*1e20*reaction_rate
	elif True:
		# from AMJUEL
		# 4.14 Reaction 2.2.12 e +H+2 → e +H +H+
		# I use this because it includes ne in the calculus
		coefficients_all = [[-17.934432746,-0.04932783688604,0.1039088280849,-0.04375935166008,0.009196691651936,-0.001043378648769,0.00006600342421838,-0.000002198466460165,0.00000003004145701249],[2.236108757681,-0.02545406018621,-0.1160421006835,0.04407846563362,-0.008192521304984,0.0008200277386433,-0.00004508284363534,0.000001282824614809,-0.00000001474719350236],[-0.3620018994703,0.0672152768015,0.01564387124002,-0.004939045440424,0.0004263195867947,0.00001034216805418,-0.0000039750286019,0.0000002322116289258,-0.00000000438121715447],[-0.4353922258965,-0.03051033606589,0.03512861172521,-0.01179504564265,0.002091772760029,-0.0001991100044575,0.00001018080238045,-0.0000002597941866088,0.000000002524118386011],[0.1580381801957,0.002493654957203,-0.01601970998119,0.005346709597939,-0.0008711870134835,0.00007542066727545,-0.000003410778344979,0.00000007120460603822,-0.0000000004412295474522],[0.01697880687685,0.0021066759639,0.000452198335817,-0.0003017151690655,0.00006209239389357,-0.000007598119096817,0.0000005523273241689,-0.00000002130508249251,0.0000000003319099650589],[-0.01521914651109,-0.0007527862162788,0.0009095551479381,-0.0002372576223034,0.00003018561480848,-0.000001365255868731,-0.00000004604769733903,0.00000000586791027043,-0.0000000001357779142836],[0.00240627636807,0.00009971361856278,-0.0001760978402353,0.00004877659148871,-0.000006477358351729,0.0000003541106430252,0.00000000130977289967,-0.000000000807290733423,0.00000000002074669430611],[-0.0001219469579955,-0.000004785505675232,0.000009858840337511,-0.000002779210878533,0.0000003720379996058,-0.00000002110289928486,0.00000000003753875073646,0.00000000004024906665497,-0.000000000001075990572574]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^3/s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		e_H2p__XXX__e_Hp_H1s = (ne_all**2)*1e40 *nH2p_ne_all*reaction_rate

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2p__XXX__e_Hp_H1s,vmin=max(np.max(e_H2p__XXX__e_Hp_H1s)*1e-12,np.min(e_H2p__XXX__e_Hp_H1s[e_H2p__XXX__e_Hp_H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2+(v) → e +H2+(2pσu) → e +H+ +H(1s) \n e +H2+(v) → e +H2+(2pπu) → e +H+ +H(n = 2) \n e +H2+(v) →H2∗∗[(2pσu)2] → e +H+ +H(1s) \n e +H2+(v) →H2∗Ryd(N1,3Λσ; eps) → e +H+ +H(1s) \nreaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate e +H2+(v) → {H2∗∗ (eps);H2∗Ryd(N1,3Λσ; eps)} →H(1s) +H(n ≥ 2)
	if False:
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 7.1.3a
		# valid from 0.01 to 10 eV
		# I'm not sure if only electron energy or also H2+
		# I'll assume it is electron energy because usually it is the highest
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		cross_section = 17.3* ( 1/((e_energy**0.665)*(1+1.1*(e_energy**0.512) + 0.011*(e_energy**3.10)) ) + 0.133* np.exp(-0.35*((e_energy - 6.05)**2)) )* 1e-16 * 1e-4		# m^2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(Te_all-0.01)).argmin()]
		cross_section_max = cross_section.flatten()[(np.abs(Te_all-10)).argmin()]
		cross_section[e_energy<0.01]=cross_section_min
		cross_section[e_energy>10]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__H1s_Hn2 = (ne_all**2)*1e40 *nH2p_ne_all*reaction_rate
	elif True:
		# from AMJUEL
		# 4.15 Reaction 2.2.14 e +H+2 →H +H
		# I use this because it includes ne in the calculus
		coefficients_all = [[-16.64335253647,0.08953780953631,-0.1056411030518,0.0447700080869,-0.009729945434357,0.001174456882002,-0.00007987743820637,0.000002842957892768,-0.00000004104508608435],[-0.6005444031657,0.04063933992726,-0.04753947846841,0.02188304031377,-0.005201085606791,0.0006866340394051,-0.00005059940013116,0.000001930213882205,-0.00000002963966822809],[0.0004494812032769,0.00007884508616595,0.0003688007562485,-0.0004659255785539,0.00019071159804,-0.00003434324710145,0.000003067651560323,-0.000000132568946559,0.00000000221249307362],[0.0001632894866655,0.0003108116177617,-0.0003521552580917,-0.0002233169775063,0.0001869415236037,-0.00004329991211511,0.000004465256901322,-0.0000002136296167564,0.000000003873085368404],[-0.00007234142549752,-0.001316311320262,0.001643509328764,-0.0006412764282779,0.0001048891053765,-0.000007018555173322,0.00000004776213235854,0.00000001380537343974,-0.0000000004199397846492],[-0.00001504085050039,0.0001315865970237,-0.0001025653773999,0.00005310324781249,-0.00001831888048039,0.000003423755373077,-0.0000003303384352061,0.000000015516270977,-0.0000000002809391819541],[0.00001113923667684,0.00002711411525392,-0.00008495922363727,0.00004026487801017,-0.00000628932447424,0.0000001911447036702,0.00000003638198230235,-0.000000003235540606394,0.00000000007605442050634],[-0.00000184392616225,-0.000001663674537499,0.00001308069926896,-0.000007324021449032,0.000001431739868187,-0.0000001085644779665,0.000000001143164983367,0.0000000002151595003971,-0.000000000007052562220005],[0.00000009864173150662,-0.0000002212261708468,-0.0000004431749501051,0.0000003270530731011,-0.00000007282085521177,0.000000006578253567957,-0.0000000001925258267827,-0.000000000004217474167519,0.0000000000002364754029318]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^3/s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		e_H2p__H1s_Hn2 = (ne_all**2)*1e40 *nH2p_ne_all*reaction_rate

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2p__H1s_Hn2,vmin=max(np.max(e_H2p__H1s_Hn2)*1e-12,np.min(e_H2p__H1s_Hn2[e_H2p__H1s_Hn2>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2+(v) → {H2∗∗ (eps);H2∗Ryd(N1,3Λσ; eps)} →H(1s) +H(n ≥ 2) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate e +H2+(v) → e +H+ +H+ + e
	if False:
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 7.1.4
		# I'm not sure if only electron energy or also H2+
		# I'll assume it is electron energy because usually it is the highest
		T_e_temp = Te_all/eV_to_K	# K
		T_e_temp[T_e_temp==0]=300	# ambient temperature
		e_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_e_temp),200))).T*(2*boltzmann_constant_J*T_e_temp.T/electron_mass)**0.5).T
		e_velocity_PDF = (4*np.pi*(e_velocity.T)**2 * gauss( e_velocity.T, (electron_mass/(2*np.pi*boltzmann_constant_J*T_e_temp.T))**(3/2) , (T_e_temp.T*boltzmann_constant_J/electron_mass)**0.5 ,0)).T
		e_energy = 0.5 * e_velocity**2 * electron_mass * J_to_eV
		cross_section = 7.39/e_energy * np.log(0.18*e_energy) * (1-np.exp(-0.105*((e_energy/15.2 -1)**1.55))) * 1e-16 * 1e-4		# m^2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__e_Hp_Hp_e = ne_all*1e20 *nH2p_ne_all*ne_all*1e20*reaction_rate
	elif True:
		# from AMJUEL
		# 4.13 Reaction 2.2.11 e +H+2 → 2e +H+ +H+
		# I use this because it included also ne other than Te
		coefficients_all = [[-37.08803769397,0.09784233987341,-0.00720036127213,0.006496843022778,-0.00142059081876,0.0001703620321164,-0.000011607389464,0.0000004148222302162,-0.000000006007853385325],[15.61780529774,-0.01673256230592,0.02743322772895,-0.01026956102747,0.001999561527383,-0.0002043607814503,0.00001084177127603,-0.0000002671800995803,0.000000002093182411476],[-6.874406034117,-0.007782929961315,-0.006888773684846,0.002306107197863,-0.0004029222834436,0.00003932152471491,-0.00000209490736415,0.0000000568290706001,-0.000000000632075254561],[2.010540060675,-0.003226785148562,-0.006181192193854,0.002388146990238,-0.0005018901320009,0.00005520233512352,-0.000003080798536641,0.00000007864770315002,-0.0000000006357395371638],[-0.361476890612,0.003710098881765,0.002045814599796,-0.0008523935993991,0.0001751295192861,-0.00001944203941844,0.000001138888354831,-0.00000003256303793266,0.0000000003501794038444],[0.02956861321735,-0.0005524443504504,-0.00002457951062112,0.00003433179945503,-0.000001450208898992,-0.0000002447566480782,0.00000001375679100044,0.0000000004863880510459,-0.00000000003004374374556],[0.0009662490252868,-0.0001548556801431,0.00001417215042439,-0.000006444863591678,-0.000001566028729499,0.0000004152486680818,-0.00000002855068942744,0.0000000006081804811,0.0000000000009512865901179],[-0.0003543571865464,0.00004662969089421,-0.00001471117766355,0.000005235585096328,-0.0000005779667826854,0.00000002139729421817,-0.000000000365604842523,0.00000000003759866326965,-0.000000000001486151370215],[0.00001827109843671,-0.000003179895716088,0.000001432429412413,-0.0000005141065080107,0.00000007734387173369,-0.000000006163336831045,0.0000000003128313515842,-0.00000000001061842444216,0.000000000000177109976964]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te_all),np.log(ne_all*1e6),coefficients_all)) * 1e-6	# m^3/s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		reaction_rate[Te_all<min(0.5,Te_all.max()/2)]=0
		e_H2p__e_Hp_Hp_e = reaction_rate * (ne_all**2)*1e40*nH2p_ne_all

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, e_H2p__e_Hp_Hp_e,vmin=max(np.max(e_H2p__e_Hp_Hp_e)*1e-12,np.min(e_H2p__e_Hp_Hp_e[e_H2p__e_Hp_Hp_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'e +H2+(v) → e +H+ +H+ + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	# NO: I take it from https://www-amdis.iaea.org/ALADDIN/
	# H2+ [1sσg;v=0-9], H [G] → H+, H [G], H [G]
	interpolation_ev1 = np.array([1.0000E-01,1.3216E-01,1.7467E-01,2.3085E-01,3.0509E-01,4.0321E-01,5.3289E-01,7.0428E-01,9.3080E-01,1.2302E+00,1.6258E+00,2.1487E+00,2.8398E+00,3.7531E+00,4.9602E+00,6.5555E+00,8.6638E+00,1.1450E+01,1.5133E+01,2.0000E+01])
	interpolation_ev2 = np.array([0.1,0.4,1,1.5,5,20])
	interpolation_reaction_rate = np.array([[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,7.8250E+00,6.3511E+00,3.6876E+00,1.4091E+00,1.6587E+00,1.5533E-78,1.7749E-61,2.4111E-48,2.5606E-38,9.9662E-31],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,2.8825E+00,1.0492E+00,1.2838E+00,3.2525E+00,5.6641E+00,1.3117E-93,4.8863E-75,2.3583E-60,8.3672E-49,9.1706E-40,1.0101E-32,2.9288E-27,4.8779E-23],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,1.1780E+00,2.5938E+00,2.3298E+00,3.5205E+00,2.0763E+00,6.7252E+00,1.0927E-84,5.6088E-70,4.1677E-58,1.5679E-48,8.3129E-41,1.4278E-34,1.5571E-29,1.8523E-25,3.7074E-22],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,3.2127E+00,9.4328E+00,2.8409E+00,3.1842E+00,2.7473E+00,2.3172E+00,1.5918E-88,5.1672E-74,3.3761E-62,1.4551E-52,1.0872E-44,3.0771E-38,6.1862E-33,1.4606E-28,6.0452E-25,6.0274E-22],[1.0630E+00,5.8307E+00,2.4311E+00,3.0605E+00,2.6228E+00,2.1022E+00,1.4070E+00,4.8355E-86,3.8061E-74,2.3267E-64,2.9679E-56,1.7448E-49,8.8682E-44,6.3858E-39,9.5609E-35,3.9963E-31,5.8340E-28,3.5211E-25,9.9705E-23,1.4563E-20],[1.9214E+00,1.0314E-86,3.5242E-74,3.3516E-64,3.0146E-56,7.0056E-50,9.5280E-45,1.4641E-40,4.2859E-37,3.5980E-34,1.1866E-31,1.9489E-29,1.8981E-27,1.2415E-25,5.9392E-24,2.1959E-22,6.4862E-21,1.5570E-19,3.0578E-18,4.9127E-17]]) * 1e-6		# m^3/s
	interpolator_reaction_rate = interpolate.interp2d(np.log(interpolation_ev1),np.log(interpolation_ev2), interpolation_reaction_rate,copy=False)
	selected = np.logical_and(T_H>0,T_H2p>0)
	reaction_rate = np.zeros_like(Te_all)
	reaction_rate[selected] = np.diag(interpolator_reaction_rate(np.log(T_H2p[selected]*eV_to_K),np.log(T_H[selected]*eV_to_K)))
	H1s_H2pv__Hp_H_H = (ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0)) *nH2p_ne_all*ne_all*1e20*reaction_rate
	plt.figure(figsize=(8, 5));
	if np.sum(H1s_H2pv__Hp_H_H>0)>20:
		if np.nanmax(H1s_H2pv__Hp_H_H)>0:
			plt.pcolor(temp_t, temp_r, H1s_H2pv__Hp_H_H,vmin=max(np.max(H1s_H2pv__Hp_H_H)*1e-12,np.min(H1s_H2pv__Hp_H_H[H1s_H2pv__Hp_H_H>0])), cmap='rainbow',norm=LogNorm());
		else:
			plt.pcolor(temp_t, temp_r, H1s_H2pv__Hp_H_H,vmin=max(np.max(H1s_H2pv__Hp_H_H)*1e-12,np.min(H1s_H2pv__Hp_H_H)), cmap='rainbow');
	else:
		plt.pcolor(temp_t, temp_r, H1s_H2pv__Hp_H_H, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+ [1sσg;v=0-9], H [G] → H+, H [G], H [G] reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	H2pvi_H2v0__Hp_H_H2v1 = nH2_ne_all*ne_all*1e20 *nH2p_ne_all*ne_all*1e20*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H2pvi_H2v0__Hp_H_H2v1,vmin=max(np.max(H2pvi_H2v0__Hp_H_H2v1)*1e-12,np.min(H2pvi_H2v0__Hp_H_H2v1[H2pvi_H2v0__Hp_H_H2v1>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+(vi) +H2(v0) → (H3+∗ +H) →H+ +H +H2(v01) \n H2+(vi) +H2(v0) → [H2+(2pσu/2pπu···) +H2] →H+ +H +H2(v01) \n reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	H2pvi_H2v0__H3p_H1s = np.zeros((15,*np.shape(ne_all)))
	# Population factors F 0v 0 of H 2 + (v 0 ) levels by electron-impact transitions from H 2 ( 1 Î£ + g ; v = 0) calculated in Franck-Condon approximation
	coefficients = np.array([0.092,0.162,0.176,0.155,0.121,0.089,0.063,0.044,0.03,0.021,0.0147,0.0103,0.0072,0.0051,0.0036,0.0024,0.0016,0.0008,0.0002])
	for vi_index in range(19):
		cross_section = coefficients[vi_index]*form_factor*(interpolator_f(np.log(baricenter_impact_energy.flatten()),vi_index).reshape(np.shape(baricenter_impact_energy)))
		cross_section[cross_section<0] = 0
		reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * H2_velocity*H2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(H2_velocity))		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		for v0_index in range(15):
			if vi_index==0:
				H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * nH2p_ne_all*ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(0) / q_vibr
			else:
				H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * nH2p_ne_all*ne_all*1e20 * ne_all*nH2_ne_all*1e20 * 1 * np.exp(-(v0_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	H2pvi_H2v0__H3p_H1s = np.sum(H2pvi_H2v0__H3p_H1s,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H2pvi_H2v0__H3p_H1s ,vmin=max(np.max(H2pvi_H2v0__H3p_H1s)*1e-12,np.min(H2pvi_H2v0__H3p_H1s[H2pvi_H2v0__H3p_H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+(vi) +H2(v0) →H3+(v3) +H(1s), singlets and triplets reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	H2p_Hm__H2N13Λσ_H1s = ne_all*1e20 * nH2p_ne_all*ne_all*1e20 * nHm_ne_all*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H2p_Hm__H2N13Λσ_H1s,vmin=max(np.max(H2p_Hm__H2N13Λσ_H1s)*1e-12,np.min(H2p_Hm__H2N13Λσ_H1s[H2p_Hm__H2N13Λσ_H1s>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H2+(vi) +H- → (H3∗) →H2(N1,3Λσ;v0) +H(1s),N=2,3 reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate H+2 (vi) +H- → (H∗ 3 ) →H+ 3 (v3) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 7.4.2
	H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
	H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
	H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
	Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/(hydrogen_mass))**0.5).T
	Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
	Hm_energy = 0.5 * Hm_velocity**2 * (hydrogen_mass) * J_to_eV
	baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + Hm_energy).T
	# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
	cross_section = 0.38/((baricenter_impact_energy**0.782)*(1+0.039*(baricenter_impact_energy**2.62))) * 1e-16 *1e-4	# m^2
	cross_section[cross_section<0] = 0
	reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	H2p_Hm__H3p_e = ne_all*1e20 * nH2p_ne_all*ne_all*1e20 * nHm_ne_all*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H2p_Hm__H3p_e,vmin=max(np.max(H2p_Hm__H3p_e)*1e-12,np.min(H2p_Hm__H3p_e[H2p_Hm__H3p_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+2 (vi) +H- → (H∗ 3 ) →H+ 3 (v3) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	Hp_H_H__H_H2p = ne_all*1e20 * nHp_ne_all*((ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))**2)*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, Hp_H_H__H_H2p,vmin=max(np.max(Hp_H_H__H_H2p)*1e-12,np.min(Hp_H_H__H_H2p[Hp_H_H__H_H2p>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ +H(1s) +H(1s) →H(1s) +H2+ (ν) reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
	H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
	H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
	H_1_energy = 0.5 * H_1_velocity**2 * (hydrogen_mass) * J_to_eV
	H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(hydrogen_mass))**0.5).T
	H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
	H_2_energy = 0.5 * H_2_velocity**2 * (hydrogen_mass) * J_to_eV
	baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
	cross_section = gauss(baricenter_impact_energy,2.5 * 1e-17 * 1e-4,0.9,3.25)
	reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
	# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s
	H1s_H_2__H2p_e = population_states[0]*(ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H1s_H_2__H2p_e,vmin=max(np.max(H1s_H_2__H2p_e)*1e-12,np.min(H1s_H_2__H2p_e[H1s_H_2__H2p_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H(1s) +H(2) → H2+(v) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
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
		H1s_H_3__H2p_e = population_states[1]*(ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	else:
		# data from https://www-amdis.iaea.org/ALADDIN/
		H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
		H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_1_energy = 0.5 * H_1_velocity**2 * (hydrogen_mass) * J_to_eV
		H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(hydrogen_mass))**0.5).T
		H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_2_energy = 0.5 * H_2_velocity**2 * (hydrogen_mass) * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
		interpolation_ev = np.array([5.5235E-02,7.1967E-02,9.0916E-02,1.1491E-01,1.4010E-01,1.7236E-01,2.0159E-01,2.3787E-01,2.7819E-01,3.2052E-01,3.6487E-01,4.1326E-01,4.5760E-01,5.2211E-01,5.7453E-01,6.2694E-01,6.9145E-01,7.5797E-01,8.3659E-01,9.0110E-01,9.7972E-01,1.0583E+00,1.1491E+00,1.2297E+00,1.3103E+00,1.4010E+00,1.5018E+00,1.5925E+00,1.6933E+00,1.7941E+00,1.8949E+00,1.9957E+00,2.1167E+00,2.2376E+00,2.3586E+00,2.4594E+00,2.5803E+00,2.7214E+00,2.8424E+00,2.9835E+00,3.1045E+00,3.2456E+00,3.3867E+00,3.5278E+00,3.6689E+00,3.8302E+00,3.9713E+00,4.1326E+00,4.2938E+00,4.4551E+00,4.6164E+00,4.7776E+00,4.9591E+00,5.1203E+00,5.3018E+00,5.4630E+00,5.6445E+00,5.8259E+00,5.9872E+00])
		interpolation_cross_section = np.array([7.3500E-16,5.9000E-16,4.2400E-16,3.1400E-16,2.3800E-16,2.0200E-16,3.1400E-16,2.6600E-16,2.1400E-16,2.1900E-16,1.9100E-16,2.0200E-16,2.6300E-16,2.6200E-16,2.3800E-16,2.6200E-16,2.1600E-16,2.3400E-16,2.5400E-16,2.8400E-16,2.7500E-16,2.8900E-16,3.0800E-16,3.0500E-16,3.3000E-16,2.8000E-16,3.1200E-16,2.3200E-16,3.0500E-16,2.7100E-16,2.5000E-16,2.2800E-16,2.4900E-16,2.3100E-16,2.0500E-16,1.9200E-16,2.1400E-16,2.0200E-16,2.1000E-16,2.0100E-16,1.8700E-16,1.8600E-16,2.0800E-16,1.7100E-16,1.6100E-16,1.5200E-16,1.6100E-16,1.2900E-16,1.2500E-16,1.2500E-16,1.2600E-16,1.1000E-16,1.0300E-16,1.1200E-16,1.0600E-16,9.3700E-17,7.7400E-17,8.0400E-17,6.9900E-17]) * 1e-4		# m^2
		interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
		cross_section = np.zeros_like(baricenter_impact_energy)
		cross_section[baricenter_impact_energy>0] = np.exp(interpolator_cross_section(np.log(baricenter_impact_energy[baricenter_impact_energy>0])))
		cross_section[cross_section<0] = 0
		reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
		H1s_H_3__H2p_e = population_states[1]*(ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H1s_H_3__H2p_e,vmin=max(np.max(H1s_H_3__H2p_e)*1e-12,np.min(H1s_H_3__H2p_e[H1s_H_3__H2p_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H(1s) +H(3) → H2+(v) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# reaction rate		H(1s) +H(4) → H2+(v) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# based on baricenter impact energy: I assume 1eV, maxwellian distribution
	# T_H_single = 1/eV_to_K	# K
	# H_velocity = np.logspace(np.log10(0.001),np.log10(10),200)*(2*boltzmann_constant_J*T_H_single/hydrogen_mass)**0.5
	# H_velocity_PDF = 4*np.pi*H_velocity**2 * gauss( H_velocity, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H_single))**(3/2) , (T_H_single*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)
	# H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
	# baricenter_impact_energy = (np.zeros((len(H_energy),len(H_energy)))+H_energy).T + H_energy
	H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
	H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
	H_1_energy = 0.5 * H_1_velocity**2 * (hydrogen_mass) * J_to_eV
	H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(hydrogen_mass))**0.5).T
	H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
	H_2_energy = 0.5 * H_2_velocity**2 * (hydrogen_mass) * J_to_eV
	baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
	cross_section = np.zeros_like(baricenter_impact_energy)
	cross_section[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] += 2.96*(4**4)*1e-19*1e-4/baricenter_impact_energy[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)]
	cross_section[np.logical_and(baricenter_impact_energy<=1,baricenter_impact_energy>0.1)] += 2.96*(4**4)*1e-18*1e-4
	cross_section[baricenter_impact_energy>1] += 2.96*(4**4)*1e-18*1e-4/(baricenter_impact_energy[baricenter_impact_energy>1]**(0.4*4))
	cross_section[cross_section<0] = 0
	reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s
	# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s
	H1s_H_4__H2p_e = population_states[2]*(ne_all*1e20 *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, H1s_H_4__H2p_e,vmin=max(np.max(H1s_H_4__H2p_e)*1e-12,np.min(H1s_H_4__H2p_e[H1s_H_4__H2p_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H(1s) +H(4) → H2+(v) + e reaction rate')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	rate_creation_Hm = e_H__Hm + e_H2X1Σg__Hm_H1s
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_Hm,vmin=max(np.max(rate_creation_Hm)*1e-3,np.min(rate_creation_Hm[rate_creation_Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_Hm = e_Hm__e_H_e + Hp_Hm__H_2_H + Hp_Hm__H_3_H + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + Hm_H2v__H_H2v_e
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_Hm-rate_destruction_Hm,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_Hm-rate_destruction_Hm),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2p = Hp_Hm__H2p_e + e_H2N13Λσ__e_H2pX2Σg_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p,vmin=max(np.max(rate_creation_H2p)*1e-3,np.min(rate_creation_H2p[rate_creation_H2p>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H2+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2p_from_Hm = Hp_Hm__H2p_e/rate_creation_H2p
	rate_creation_H2p_from_Hm[np.logical_not(np.isfinite(rate_creation_H2p_from_Hm))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_Hm,vmax=1, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2+ from H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2p_from_Hp = (Hp_Hm__H2p_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p)/rate_creation_H2p
	rate_creation_H2p_from_Hp[np.logical_not(np.isfinite(rate_creation_H2p_from_Hp))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_Hp,vmax=1, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2+ from H+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2p_from_H = (Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e)/rate_creation_H2p
	rate_creation_H2p_from_H[np.logical_not(np.isfinite(rate_creation_H2p_from_H))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_H,vmax=1, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2+ from H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2p_from_H2 = (e_H2N13Λσ__e_H2pX2Σg_e + Hp_H2X1Σg__H1s_H2pX2Σg)/rate_creation_H2p
	rate_creation_H2p_from_H2[np.logical_not(np.isfinite(rate_creation_H2p_from_H2))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p_from_H2,vmax=1, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2+ from H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_H2p = e_H2p__XXX__e_Hp_H1s + e_H2p__H1s_Hn2 + e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s + H2p_Hm__H3p_e
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of H2+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p-rate_destruction_H2p,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of H2+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_H2p-rate_destruction_H2p),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of H2+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2 = Hm_H1s__H2_v_e + Hp_H_H__Hp_H2 + H_H_H__H_H2 + H2p_Hm__H2N13Λσ_H1s
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2,vmin=max(np.max(rate_creation_H2)*1e-3,np.min(rate_creation_H2[rate_creation_H2>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2_from_Hm = (Hm_H1s__H2_v_e + H2p_Hm__H2N13Λσ_H1s)/rate_creation_H2
	rate_creation_H2_from_Hm[np.logical_not(np.isfinite(rate_creation_H2_from_Hm))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2_from_Hm,vmax=1, cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2 from H-')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2_from_Hp = Hp_H_H__Hp_H2/rate_creation_H2
	rate_creation_H2_from_Hp[np.logical_not(np.isfinite(rate_creation_H2_from_Hp))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2_from_Hp,vmax=1, cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2 from H+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2_from_H = (Hp_H_H__Hp_H2 + H_H_H__H_H2)/rate_creation_H2
	rate_creation_H2_from_H[np.logical_not(np.isfinite(rate_creation_H2_from_H))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2,vmax=1, cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2 from H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H2_from_H2p = H2p_Hm__H2N13Λσ_H1s/rate_creation_H2
	rate_creation_H2_from_H2p[np.logical_not(np.isfinite(rate_creation_H2_from_H2p))]=0
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2_from_H2p,vmax=1, cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('fractional reatcion rate [au]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Fraction of rate of creation of H2 from H2+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_H2 = e_H2_X1Σg_x__e_H_H + e_H2N13Λσ__e_H2pX2Σg_e + e_H2N13Λσ__Hp_H_2e + e_H2X1Σg__Hm_H1s + e_H2X1Σg__e_H1s_H1s + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H2X1Σg__Hp_H1s_H1s + H1s_H2v__H1s_2H1s + H2v0_H2v__H2v0_2H1s + H2pvi_H2v0__H3p_H1s
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_Hm,vmin=max(np.max(rate_destruction_Hm)*1e-3,np.min(rate_destruction_Hm[rate_destruction_Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2-rate_destruction_H2,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_H2-rate_destruction_H2),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of H2')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H3p = H2pvi_H2v0__H3p_H1s + H2p_Hm__H3p_e
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H2p,vmin=max(np.max(rate_creation_H2p)*1e-3,np.min(rate_creation_H2p[rate_creation_H2p>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H3+')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/molecules_detailed' + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_H = e_Hm__e_H_e + 2*Hp_Hm__H_2_H + 2*Hp_Hm__H_3_H + Hm_H1s__H1s_H1s_e + 2*e_H2_X1Σg_x__e_H_H + e_H2N13Λσ__Hp_H_2e + e_H2X1Σg__Hm_H1s + 2*e_H2X1Σg__e_H1s_H1s + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H2X1Σg__Hp_H1s_H1s + Hm_H2v__H_H2v_e + 2*H1s_H2v__H1s_2H1s + 2*H2v0_H2v__H2v0_2H1s + e_H2p__XXX__e_Hp_H1s + 2*e_H2p__H1s_Hn2 + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H,vmin=max(np.max(rate_creation_H)*1e-3,np.min(rate_creation_H[rate_creation_H>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_H = Hm_H1s__H2_v_e + 2*Hp_H_H__Hp_H2 + 2*H_H_H__H_H2 + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e + e_H__Hm
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_H,vmin=max(np.max(rate_destruction_H)*1e-3,np.min(rate_destruction_H[rate_destruction_H>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of H from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_H-rate_destruction_H,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of H from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_H-rate_destruction_H),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of H from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_creation_Hp = e_H2N13Λσ__Hp_H_2e + e_H2p__XXX__e_Hp_H1s + 2*e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_Hp,vmin=max(np.max(rate_creation_Hp)*1e-3,np.min(rate_creation_Hp[rate_creation_Hp>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of H+ from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_Hp = Hp_Hm__H_2_H + Hp_Hm__H_3_H + Hp_Hm__H2p_e + Hp_H2X1Σg__H1s_H2pX2Σg + Hp_H_H__H_H2p
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_Hp,vmin=max(np.max(rate_destruction_Hp)*1e-3,np.min(rate_destruction_Hp[rate_destruction_Hp>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of H+ from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_Hp-rate_destruction_Hp,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of H+ from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_Hp-rate_destruction_Hp),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of H+ from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	rate_creation_e = e_Hm__e_H_e + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + e_H2N13Λσ__e_H2pX2Σg_e + e_H2N13Λσ__Hp_H_2e + Hm_H2v__H_H2v_e + e_H2p__e_Hp_Hp_e + H2p_Hm__H3p_e + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_e,vmin=max(np.max(rate_creation_e)*1e-3,np.min(rate_creation_e[rate_creation_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of creation of e- from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	rate_destruction_e = e_H__Hm + e_H2X1Σg__Hm_H1s + e_H2p__H1s_Hn2
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_destruction_e,vmin=max(np.max(rate_destruction_e)*1e-3,np.min(rate_destruction_e[rate_destruction_e>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Rate of destruction of e- from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, rate_creation_e-rate_destruction_e,vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of creation of e- from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, -(rate_creation_e-rate_destruction_e),vmin=0, cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Net rate of destruction of e- from molecules only')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	length = 0.351+target_OES_distance/1000	# m distance skimmer to OES/TS + OES/TS to target
	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	plt.figure(figsize=(20, 10));
	plt.plot(time_crop, np.sum(area * rate_creation_Hm,axis=1)/np.max(np.sum(area * rate_creation_Hm,axis=1)), label='H- creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_Hm,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_Hm,axis=1)/np.max(np.sum(area * rate_destruction_Hm,axis=1)), label='H- destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_Hm,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_creation_H2p,axis=1)/np.max(np.sum(area * rate_creation_H2p,axis=1)), label='H2+ creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H2p,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_H2p,axis=1)/np.max(np.sum(area * rate_destruction_H2p,axis=1)), label='H2+ destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H2p,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_creation_H2,axis=1)/np.max(np.sum(area * rate_creation_H2,axis=1)), label='H2 creation rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H2,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_H2,axis=1)/np.max(np.sum(area * rate_destruction_H2,axis=1)), label='H2 destruction rate\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H2,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * Te_all,axis=1)/np.max(np.sum(area * Te_all,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * Te_all,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * ne_all,axis=1)/np.max(np.sum(area * ne_all,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * ne_all,axis=1)/np.sum(area))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('relative radial average [au]')
	plt.title(pre_title+'Time evolution of the radially averaged molecular reaction rates '+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	plt.figure(figsize=(20, 10));
	threshold_radious = 0.025
	plt.plot(time_crop, np.sum(area * effective_ionisation_rates,axis=1)/np.max(np.sum(area * effective_ionisation_rates,axis=1)), label='ionisation rate from ADAS\n(max='+'%.3g # m^-3 s-1)' %(np.max(np.sum(area * effective_ionisation_rates,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * effective_recombination_rates,axis=1)/np.max(np.sum(area * effective_recombination_rates,axis=1)), label='recombination rate from ADAS\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * effective_recombination_rates,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_creation_H,axis=1)/np.max(np.sum(area * rate_creation_H,axis=1)), label='H creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_H,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_H,axis=1)/np.max(np.sum(area * rate_destruction_H,axis=1)), label='H destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_H,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_creation_Hp,axis=1)/np.max(np.sum(area * rate_creation_Hp,axis=1)), label='H+ creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_Hp,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_Hp,axis=1)/np.max(np.sum(area * rate_destruction_Hp,axis=1)), label='H+ destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_Hp,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_creation_e,axis=1)/np.max(np.sum(area * rate_creation_e,axis=1)), label='e- creation rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_creation_e,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * rate_destruction_e,axis=1)/np.max(np.sum(area * rate_destruction_e,axis=1)), label='e- destruction rate from molecules\n(max='+'%.3g # m^-3 s-1)' % (np.max(np.sum(area * rate_destruction_e,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * Te_all,axis=1)/np.max(np.sum(area * Te_all,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * Te_all,axis=1)/np.sum(area))));
	plt.plot(time_crop, np.sum(area * ne_all,axis=1)/np.max(np.sum(area * ne_all,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * ne_all,axis=1)/np.sum(area))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('relative radial average [au]')
	plt.title(pre_title+'Time evolution of the radially averaged reaction rates\ncomparison of atomic vs molecular processes\n'+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the ionisation mean free path
	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionization_length_H = thermal_velocity_H/(temp * ne_all * 1e20 )
	ionization_length_H = np.where(np.isnan(ionization_length_H), np.inf, ionization_length_H)
	# ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
	# ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ionization_length_H,vmax=min(np.max(ionization_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/ionization_length_H,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	temp2 = np.cumsum(dx/ionization_length_H,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'ionization length of neutral H from ADAS\ncold H (from simul), ionisation rate from Te/ne')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the ionisation mean free path
	thermal_velocity_H = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionization_length_H = thermal_velocity_H/(temp * ne_all * 1e20 )
	ionization_length_H = np.where(np.isnan(ionization_length_H), np.inf, ionization_length_H)
	# ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
	# ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, ionization_length_H,vmax=min(np.max(ionization_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/ionization_length_H,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	temp2 = np.cumsum(dx/ionization_length_H,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'ionization length of neutral H from ADAS\nhot H (Th = Te, from rec), ionisation rate from Te/ne')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the H destruction mean free path
	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	destruction_length_H = thermal_velocity_H/( rate_destruction_H/(nH_ne_all*ne_all*1e20) )
	destruction_length_H = np.where(np.isnan(destruction_length_H), np.inf, destruction_length_H)
	# destruction_length_H = np.where(np.isinf(destruction_length_H), np.nan, destruction_length_H)
	# destruction_length_H = np.where(np.isnan(destruction_length_H), np.nanmax(destruction_length_H[np.isfinite(destruction_length_H)]), destruction_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, destruction_length_H,vmax=min(np.max(destruction_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/destruction_length_H,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	temp2 = np.cumsum(dx/destruction_length_H,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'"destruction" length of H from Yacora\ncold H (from simul)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the H destruction mean free path
	thermal_velocity_H = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	destruction_length_H = thermal_velocity_H/( rate_destruction_H/(nH_ne_all*ne_all*1e20) )
	destruction_length_H = np.where(np.isnan(destruction_length_H), np.inf, destruction_length_H)
	# destruction_length_H = np.where(np.isinf(destruction_length_H), np.nan, destruction_length_H)
	# destruction_length_H = np.where(np.isnan(destruction_length_H), np.nanmax(destruction_length_H[np.isfinite(destruction_length_H)]), destruction_length_H)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, destruction_length_H,vmax=min(np.max(destruction_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/destruction_length_H,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	temp2 = np.cumsum(dx/destruction_length_H,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'"destruction" length of H from Yacora\nhot H (Th = Te, from rec)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	thermal_velocity_H = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionisation_length_H_CX = thermal_velocity_H/( temp*nHp_ne_all * ne_all * 1e20)
	ionisation_length_H_CX = np.where(np.isnan(ionisation_length_H_CX), np.inf, ionisation_length_H_CX)
	# ionisation_length_H_CX = np.where(np.isinf(ionisation_length_H_CX), np.nan, ionisation_length_H_CX)
	# ionisation_length_H_CX = np.where(np.isnan(ionisation_length_H_CX), np.nanmax(ionisation_length_H_CX[np.isfinite(ionisation_length_H_CX)]), ionisation_length_H_CX)

	temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_charge_exchange_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
	effective_charge_exchange_rates = (effective_charge_exchange_rates * (ne_all**2) *1e40* nH_ne_all*nHp_ne_all).astype('float')

	geometric_factor = ionisation_length_H_CX / np.abs((2*averaged_profile_sigma*np.ones_like(ionisation_length_H_CX.T)).T - r_crop)	# I take diameter = 2 * FWHM
	geometric_factor[geometric_factor>1] = 1

	delta_t = (T_Hp - T_H)
	delta_t[delta_t<0]=0
	E_HCX = 3/2* (delta_t * eV_to_K) * (effective_charge_exchange_rates/nHp_ne_all) / effective_ionisation_rates * geometric_factor	# eV
	E_HCX_max = np.sort(E_HCX[np.isfinite(E_HCX)])[-5]
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, E_HCX, cmap='rainbow',vmin=1, vmax=1e6, norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('enery [eV],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Energy CX cost of atomic hydrogen per ionisation from ADAS\nhot H (TH=Te, from rec), H destruction rate only ionisation')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, P_HCX,vmin=max(np.max(P_HCX)/1e6,1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp2 = np.cumsum(dx/ionisation_length_H_CX,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power for atomic hydrogen charge exchange from ADAS\nhot H (TH=Te, from rec), H destruction rate only ionisation\ntot limit on exit=%.3gJ' %(np.sum(P_HCX*geometric_factor*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	thermal_velocity_H = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	destruction_length_H_CX = thermal_velocity_H/( rate_destruction_H/(nH_ne_all*ne_all*1e20) + temp *nHp_ne_all* ne_all * 1e20)
	destruction_length_H_CX = np.where(np.isnan(destruction_length_H_CX), np.inf, destruction_length_H_CX)
	# destruction_length_H_CX = np.where(np.isinf(destruction_length_H_CX), np.nan, destruction_length_H_CX)
	# destruction_length_H_CX = np.where(np.isnan(destruction_length_H_CX), np.nanmax(destruction_length_H_CX[np.isfinite(destruction_length_H_CX)]), destruction_length_H_CX)

	geometric_factor = destruction_length_H_CX / np.abs((2*averaged_profile_sigma*np.ones_like(destruction_length_H_CX.T)).T - r_crop)	# I take diameter = 2 * FWHM
	geometric_factor[geometric_factor>1] = 1

	E_HCX = 3/2* (delta_t * eV_to_K) * (effective_charge_exchange_rates/nHp_ne_all) / effective_ionisation_rates * geometric_factor	# eV
	E_HCX_max = np.sort(E_HCX[np.isfinite(E_HCX)])[-5]
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, E_HCX, cmap='rainbow',vmin=1, vmax=1e6, norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('enery [eV],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Energy CX cost of atomic hydrogen per ionisation from ADAS\nhot H (TH=Te, from rec), H destruction rate with molecules')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, P_HCX,vmin=max(np.max(P_HCX)/1e6,1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp2 = np.cumsum(dx/destruction_length_H_CX,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power for atomic hydrogen charge exchange from ADAS\nhot H (TH=Te, from rec), H destruction rate with molecules\ntot limit on exit=%.3gJ' %(np.sum(P_HCX*geometric_factor*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	ionisation_length_H_CX = thermal_velocity_H/( temp *nHp_ne_all* ne_all * 1e20)
	ionisation_length_H_CX = np.where(np.isnan(ionisation_length_H_CX), np.inf, ionisation_length_H_CX)
	# ionisation_length_H_CX = np.where(np.isinf(ionisation_length_H_CX), np.nan, ionisation_length_H_CX)
	# ionisation_length_H_CX = np.where(np.isnan(ionisation_length_H_CX), np.nanmax(ionisation_length_H_CX[np.isfinite(ionisation_length_H_CX)]), ionisation_length_H_CX)

	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, P_HCX,vmin=max(np.max(P_HCX)/1e6,1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/ionisation_length_H_CX,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power for atomic hydrogen charge exchange from ADAS\ncold H (from simul), H destruction rate only ionisation\nlimit on enter=%.3gJ' %(np.sum(P_HCX*(temp1>0)*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in ionisations m^-3 s-1 / (# / m^3)**2
	destruction_length_H_CX = thermal_velocity_H/( rate_destruction_H/(nH_ne_all*ne_all*1e20) + temp *nHp_ne_all* ne_all * 1e20)
	destruction_length_H_CX = np.where(np.isnan(destruction_length_H_CX), np.inf, destruction_length_H_CX)
	# destruction_length_H_CX = np.where(np.isinf(destruction_length_H_CX), np.nan, destruction_length_H_CX)
	# destruction_length_H_CX = np.where(np.isnan(destruction_length_H_CX), np.nanmax(destruction_length_H_CX[np.isfinite(destruction_length_H_CX)]), destruction_length_H_CX)

	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, P_HCX,vmin=max(np.max(P_HCX)/1e6,1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/ionisation_length_H_CX,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power for atomic hydrogen charge exchange from ADAS\ncold H (from simul), H destruction rate with molecules\nlimit on enter=%.3gJ' %(np.nansum(P_HCX*(temp1>0)*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of neutrals the CX mean free path
	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
	CX_length_H = thermal_velocity_H/(temp *nHp_ne_all* ne_all * 1e20 )
	CX_length_H = np.where(np.isnan(CX_length_H), np.inf, CX_length_H)
	# CX_length_H = np.where(np.isinf(CX_length_H), np.nan, CX_length_H)
	# CX_length_H = np.where(np.isnan(CX_length_H), np.nanmax(CX_length_H[np.isfinite(CX_length_H)]), CX_length_H)
	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, CX_length_H,vmax=min(np.max(CX_length_H),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/CX_length_H,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'neutrals CX length from ADAS\ncold H (from simul), CX rate from Te/ne\nlimit on enter=%.3gJ' %(np.nansum(P_HCX*(temp1>0)*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of H+ the CX mean free path
	thermal_velocity_Hp = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(Te_all))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
	CX_length_Hp = thermal_velocity_Hp/(temp *nH_ne_all* ne_all * 1e20 )
	CX_length_Hp = np.where(np.isnan(CX_length_Hp), 0, CX_length_Hp)
	# CX_length_H = np.where(np.isinf(CX_length_H), np.nan, CX_length_H)
	# CX_length_H = np.where(np.isnan(CX_length_H), np.nanmax(CX_length_H[np.isfinite(CX_length_H)]), CX_length_H)

	geometric_factor = CX_length_Hp / np.abs((2*averaged_profile_sigma*np.ones_like(destruction_length_H_CX.T)).T - r_crop)	# I take diameter = 2 * FWHM
	geometric_factor[geometric_factor>1] = 1

	P_HCX = 3/2* (delta_t * eV_to_K) * effective_charge_exchange_rates / J_to_eV	# W / m^3
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, CX_length_Hp,vmax=min(np.max(CX_length_Hp),1), cmap='rainbow', norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp2 = np.cumsum(dx/CX_length_Hp,axis=1)
	temp2 = 1-temp2
	temp2[temp2<0]=0
	temp2[:,0][temp2[:,0]==0]=1
	plt.plot(time_crop,(np.shape(temp2)[1]-np.flip(temp2==0,axis=1).argmin(axis=1))*dx,'--',color='r',label='limit full absorption\nfrom inside outward')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('CX length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'H+ CX length from ADAS\nTH+=Te, CX rate from Te/ne\ntot limit on exit=%.3gJ' %(np.sum(P_HCX*geometric_factor*area*length*dt/1000)))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# calculation of the H depleted on its way toward the axis and how the hot hydrogen generated is depleted on the way out
	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	thermal_velocity_Hp = ( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_charge_exchange_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
	effective_charge_exchange_rates = (effective_charge_exchange_rates * (ne_all) *1e20 * nHp_ne_all).astype('float')
	temp = (rate_destruction_H + effective_ionisation_rates)/(nH_ne_all*ne_all*1e20)
	temp[np.isnan(temp)] = 0.
	effective_H_destruction_rate = effective_charge_exchange_rates+ temp
	H_inflow = effective_H_destruction_rate*dr/thermal_velocity_H
	H_inflow = 2*target_chamber_pressure/(boltzmann_constant_J*300)* ( (300*boltzmann_constant_J)/ hydrogen_mass)**0.5 * np.exp(-np.flip(np.cumsum(np.flip(H_inflow,axis=1),axis=1),axis=1)-np.flip(np.cumsum(np.flip(effective_H_destruction_rate,axis=1),axis=1),axis=1)*dr/thermal_velocity_Hp) / thermal_velocity_H
	delta_t = (T_Hp - T_H)
	delta_t[delta_t<0]=0
	P_HCX = 3/2* (delta_t * boltzmann_constant_J) * effective_charge_exchange_rates * H_inflow
	E_HCX = np.sum(2*np.pi*r_crop*P_HCX* dr)*dt/1000*length

	if np.nanmax(P_HCX)==0:
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, P_HCX, cmap='rainbow',vmin=max(np.max(P_HCX)/1e6,1), norm=LogNorm());
		plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		plt.legend(loc='best', fontsize='xx-small')
		# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
		cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Power for atomic hydrogen CX from ADAS (inflow then outflow)\ncold H (from simul), total H destruction rate (atomic, molecular, CX)\nlimit on enter=%.3gJ' %(E_HCX))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
	temp = read_adf11(ccdfile, 'ccd', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_charge_exchange_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # in CX m^-3 s-1 / (# / m^3)**2
	effective_charge_exchange_rates = (effective_charge_exchange_rates * (ne_all) *1e20 * nHp_ne_all).astype('float')
	temp = (rate_destruction_H + effective_ionisation_rates)/(nH_ne_all*ne_all*1e20)
	temp[np.isnan(temp)] = 0.
	effective_H_destruction_rate = temp
	H_inflow = effective_H_destruction_rate*dr/thermal_velocity_H
	H_inflow = 2*target_chamber_pressure/(boltzmann_constant_J*300)* ( (300*boltzmann_constant_J)/ hydrogen_mass)**0.5 * np.exp(-np.flip(np.cumsum(np.flip(H_inflow,axis=1),axis=1),axis=1)-np.flip(np.cumsum(np.flip(effective_H_destruction_rate,axis=1),axis=1),axis=1)*dr/thermal_velocity_Hp) / thermal_velocity_H
	delta_t = (T_Hp - T_H)
	delta_t[delta_t<0]=0
	P_HCX = 3/2* (delta_t * boltzmann_constant_J) * effective_charge_exchange_rates * H_inflow
	E_HCX = np.sum(2*np.pi*r_crop*P_HCX* dr)*dt/1000*length

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, P_HCX, cmap='rainbow',vmin=max(np.max(P_HCX)/1e6,1), norm=LogNorm());
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
	cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('Power [W/m3],')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power for atomic hydrogen charge exchange from ADAS (inflow then outflow)\ncold H (from simul), H destruction rate (atomic, molecular)\nlimit on enter=%.3gJ' %(E_HCX))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	# Calculation of the H- destruction mean free path
	thermal_velocity_Hm = ( (T_Hm*boltzmann_constant_J)/ hydrogen_mass)**0.5
	destruction_length_Hm = thermal_velocity_Hm/( rate_destruction_Hm/(nHm_ne_all*ne_all*1e20) )
	destruction_length_Hm = np.where(np.isnan(destruction_length_Hm), np.inf, destruction_length_Hm)
	# destruction_length_Hm = np.where(np.isinf(destruction_length_Hm), np.nan, destruction_length_Hm)
	# destruction_length_Hm = np.where(np.isnan(destruction_length_Hm), np.nanmax(destruction_length_Hm[np.isfinite(destruction_length_Hm)]), destruction_length_Hm)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, destruction_length_Hm,vmax=min(np.max(destruction_length_Hm),1), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'"destruction" length of H- from Yacora\ncold H- (from simul)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the H2+ destruction mean free path
	thermal_velocity_H2p = ( (T_H2p*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5
	destruction_length_H2p = thermal_velocity_H2p/( rate_destruction_H2p/(nH2p_ne_all*ne_all*1e20) )
	destruction_length_H2p = np.where(np.isnan(destruction_length_H2p), np.inf, destruction_length_H2p)
	# destruction_length_H2p = np.where(np.isinf(destruction_length_H2p), np.nan, destruction_length_H2p)
	# destruction_length_H2p = np.where(np.isnan(destruction_length_H2p), np.nanmax(destruction_length_H2p[np.isfinite(destruction_length_H2p)]), destruction_length_H2p)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, destruction_length_H2p,vmax=min(np.max(destruction_length_H2p),1), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'"destruction" length of H2+ from Yacora\ncold H2+ (from simul)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	# Calculation of the H2 destruction mean free path
	thermal_velocity_H2 = ( (T_H2*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5
	destruction_length_H2 = thermal_velocity_H2/( rate_destruction_H2/(nH2_ne_all*ne_all*1e20) )
	destruction_length_H2 = np.where(np.isnan(destruction_length_H2), np.inf, destruction_length_H2)
	# destruction_length_H2 = np.where(np.isinf(destruction_length_H2), np.nan, destruction_length_H2)
	# destruction_length_H2 = np.where(np.isnan(destruction_length_H2), np.nanmax(destruction_length_H2[np.isfinite(destruction_length_H2)]), destruction_length_H2)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, destruction_length_H2,vmax=min(np.max(destruction_length_H2),1), cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
	temp1 = np.flip(np.cumsum(np.flip(dx/destruction_length_H2,axis=1),axis=1),axis=1)
	temp1 = 1-temp1
	temp1[temp1<0]=0
	temp1[:,-1][temp1[:,-1]==1]=1
	plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'"destruction" length of H2 from Yacora\ncold H2 (from simul)')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	length = 0.351+target_OES_distance/1000	# m distance skimmer to OES/TS + OES/TS to target
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
		time_rate_flow_target_e=np.append(time_rate_flow_target_e,np.median(np.diff(r_crop))*np.sum(2*np.pi*r_crop[:ii_plasma_radius]*0.5*ne_all[index,:ii_plasma_radius]*1e20*(boltzmann_constant_J*(1+5/3)*Te_all[index,:ii_plasma_radius]/(eV_to_K*hydrogen_mass))**0.5))
		time_rate_creation_e=np.append(time_rate_creation_e,length*np.median(np.diff(r_crop))*np.sum(rate_creation_e[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
		time_rate_destruction_e=np.append(time_rate_destruction_e,length*np.median(np.diff(r_crop))*np.sum(rate_destruction_e[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
		time_effective_ionisation_rates=np.append(time_effective_ionisation_rates,length*np.median(np.diff(r_crop))*np.sum(effective_ionisation_rates[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
		time_effective_recombination_rates=np.append(time_effective_recombination_rates,length*np.median(np.diff(r_crop))*np.sum(effective_recombination_rates[index,:ii_plasma_radius]*2*np.pi*r_crop[:ii_plasma_radius]))
	plt.figure(figsize=(20, 10));
	plt.plot(time_crop, time_rate_flow_target_e, label='sheath absorption at target (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_flow_target_e)));
	plt.plot(time_crop, time_rate_creation_e, label='creation via molecules (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_creation_e)));
	plt.plot(time_crop, time_rate_destruction_e, label='destruction via molecules (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_rate_destruction_e)));
	plt.plot(time_crop, time_effective_ionisation_rates, label='ionisation from H (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_effective_ionisation_rates)));
	plt.plot(time_crop, time_effective_recombination_rates, label='recombination fromH+ (tot=%.3g #)' %(np.median(np.diff(time_crop))*np.sum(time_effective_recombination_rates)));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('creation/destruction rate [#/s]')
	plt.title(pre_title+'Total electron creation/destruction rates skimmer to target vs target sheath condition\n'+'total sheath %.3g#, total volume %.3g' %(np.median(np.diff(time_crop))*np.sum(time_rate_flow_target_e),np.median(np.diff(time_crop))*np.sum( time_rate_creation_e-time_rate_destruction_e+time_effective_ionisation_rates-time_effective_recombination_rates ))+'\n'+str(target_OES_distance)+'mm from the target, %.3gPa target chamber pressure' %(target_chamber_pressure))
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	MAR_via_H2p = np.min([Hp_H2X1Σg__H1s_H2pX2Σg,e_H2p__H1s_Hn2],axis=0)
	# MAR_via_H2p = e_H2p__H1s_Hn2
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, MAR_via_H2p,vmin=max(np.max(MAR_via_H2p)*1e-3,np.min(MAR_via_H2p[MAR_via_H2p>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Molecular activated recombination H2+ branch\nH2(v) + H+ → H2+ +H , H2+ + e− → H(*) + H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, MAR_via_H2p,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Molecular activated recombination H2+ branch\nH2(v) + H+ → H2+ +H , H2+ + e− → H(*) + H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	MAR_via_Hm = np.min([e_H2X1Σg__Hm_H1s,Hp_Hm__H_2_H+Hp_Hm__H_3_H],axis=0)
	# MAR_via_Hm = Hp_Hm__H_2_H+Hp_Hm__H_3_H
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, MAR_via_Hm,vmin=max(np.max(MAR_via_Hm)*1e-3,np.min(MAR_via_Hm[MAR_via_Hm>0])), cmap='rainbow',norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Molecular activated recombination H- branch\nH2(v) + e− → H− + H , H− + H+ → H(*) + H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, MAR_via_Hm,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('reatcion rate [# m^-3 s-1]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Molecular activated recombination H- branch\nH2(v) + e− → H− + H , H− + H+ → H(*) + H')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	temp = read_adf11(pltfile, 'plt', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # radiated power W m^-3 s-1 / (# / m^3)**2
	power_rad_excit = temp * (ne_all**2) * 1e40 *nH_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, power_rad_excit*area*length,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Power [W]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Radiative power sink due to H excitation')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	temp = read_adf11(prbfile, 'prb', 1, 1, 1, Te_all.flatten(),(ne_all * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop))) * (10 ** -6)  # radiated power W m^-3 s-1 / (# / m^3)**2
	power_rad_rec_bremm = temp * (ne_all**2) * 1e40 *nHp_ne_all
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, power_rad_rec_bremm*area*length,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Power [W]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Radiative power sink due to recombination and Bremsstrahlung')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	einstein_coeff_full_full = np.array([[4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04],[0,0.44101,0.084193,0.0253044,0.009732,0.0043889,0.0022148,0.0012156,0.00071225,0.00043972,0.00028337,0.00018927],[0,0,8.9860e-02,2.2008e-02,7.7829e-03,3.3585e-03,1.6506e-03,8.9050e-04,5.1558e-04,3.1558e-04,2.0207e-04,1.3431e-04],[0,0,0,2.6993e-02,7.7110e-03,3.0415e-03,1.4242e-03,7.4593e-04,4.2347e-04,2.5565e-04,1.6205e-04,1.0689e-04],[0,0,0,0,1.0254e-02,3.2528e-03,1.3877e-03,6.9078e-04,3.7999e-04,2.2460e-04,1.4024e-04,9.1481e-05],[0,0,0,0,0,4.5608e-03,1.5609e-03,7.0652e-04,3.6881e-04,2.1096e-04,1.2884e-04,8.2716e-05],[0,0,0,0,0,0,2.2720e-03,8.2370e-04,3.9049e-04,2.1174e-04,1.2503e-04,7.8457e-05],[0,0,0,0,0,0,0,1.2328e-03,4.6762e-04,2.3007e-04,1.2870e-04,7.8037e-05],[0,0,0,0,0,0,0,0,7.1514e-04,2.8131e-04,1.4269e-04,8.1919e-05],[0,0,0,0,0,0,0,0,0,4.3766e-04,1.7740e-04,9.2309e-05],[0,0,0,0,0,0,0,0,0,0,2.7989e-04,1.1633e-04],[0,0,0,0,0,0,0,0,0,0,0,1.8569e-04]]) * 1e8  # 1/s
	level_1 = (np.ones((13,13))*np.arange(1,14)).T
	level_2 = (np.ones((13,13))*np.arange(1,14))
	energy_difference_full_full = 1/level_1**2-1/level_2**2
	energy_difference_full_full = 13.6*energy_difference_full_full[:-1,1:]
	energy_difference_full_full[energy_difference_full_full<0]=0
	multiplicative_factor_full_full = np.sum(energy_difference_full_full * einstein_coeff_full_full / J_to_eV,axis=0)
	power_rad_mol = np.sum((population_states_molecules.T * multiplicative_factor_full_full).T,axis=0)
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, power_rad_mol*area*length,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Power [W]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Radiative power sink due molecule induced H de-excitation')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_ionisation_rates*13.6*area*length/J_to_eV,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Power [W]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power sink due to atomic ionisation')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, (power_rad_excit*J_to_eV+effective_ionisation_rates*13.6)/effective_ionisation_rates,vmax=100,cmap='rainbow', norm=LogNorm());
	plt.colorbar(orientation="horizontal").set_label('energy per ionisation [eV]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Energy excitation cost per atomic ionisation')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	plt.figure(figsize=(8, 5));
	plt.pcolor(temp_t, temp_r, effective_recombination_rates*13.6*area*length/J_to_eV,cmap='rainbow');
	plt.colorbar(orientation="horizontal").set_label('Power [W]')  # ;plt.pause(0.01)
	plt.axes().set_aspect(20)
	plt.xlabel('time [ms]')
	plt.ylabel('radial location [m]      ')
	plt.title(pre_title+'Power sink due to atomic recombination')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	# heat_inflow_upstream_max = np.sum(area * ne_all*1e20 * 10000*(ionisation_potential + dissociation_potential + Te_all + nHp_ne_all*T_Hp*eV_to_K)/J_to_eV,axis=1)	# W
	# heat_inflow_upstream_min = np.sum(area * ne_all*1e20 * 1000*(ionisation_potential + dissociation_potential + Te_all + nHp_ne_all*T_Hp*eV_to_K)/J_to_eV,axis=1)	# W
	# target_adiabatic_collisional_velocity = ((Te_all + 5/3 *T_Hp*eV_to_K)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
	# target_Bohm_adiabatic_flow = 0.5*ne_all*1e20*target_adiabatic_collisional_velocity	# 0.5 time reduction on ne in the presheath
	# electron_charge = 1.60217662e-19	# C
	# electron_mass = 9.10938356e-31	# kg
	# electron_sound_speed = (Te_all/eV_to_K*boltzmann_constant_J/(electron_mass))**0.5	# m/s
	# ion_sound_speed = (T_Hp*boltzmann_constant_J/(hydrogen_mass))**0.5	# m/s
	# sheath_potential_drop = Te_all/eV_to_K*boltzmann_constant_J/electron_charge * np.log(4*ion_sound_speed/electron_sound_speed)
	# sheath_potential_drop[np.isnan(sheath_potential_drop)]=0
	# presheath_potential_drop = Te_all/eV_to_K*boltzmann_constant_J/electron_charge * np.log(0.5)
	# ions_pre_sheath_acceleration = 0.4
	# electrons_pre_sheath_acceleration = 0.2
	# neutrals_natural_reflection = 0.6
	# target_heat_flow = target_Bohm_adiabatic_flow*((2.5*boltzmann_constant_J*T_Hp-electron_charge*sheath_potential_drop-electron_charge*presheath_potential_drop)*(1-ions_pre_sheath_acceleration) + 2*Te_all/eV_to_K*boltzmann_constant_J*(1-electrons_pre_sheath_acceleration) + ionisation_potential + dissociation_potential*(1-neutrals_natural_reflection))

	# power_pulse_shape_peak = power_pulse_shape.argmax()
	# ionisation_peak = time_crop[np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1).argmax()]
	# time_source_power = np.arange(len(power_pulse_shape))*time_resolution*1000	# I work here in ms, while current trace is in s
	# time_source_power = time_source_power-time_source_power[power_pulse_shape_peak]+ionisation_peak
	# power_pulse_shape_crop = power_pulse_shape[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
	# power_pulse_shape_std_crop = power_pulse_shape_std[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
	# time_source_power_crop = time_source_power[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
	power_pulse_shape_crop = interpolated_power_pulse_shape(time_crop)
	power_pulse_shape_std_crop = interpolated_power_pulse_shape_std(time_crop)
	time_source_power_crop = cp.deepcopy(time_crop)

	plt.figure(figsize=(12, 6));
	plt.plot(time_crop, np.sum(power_rad_excit*area*length,axis=1), label='Radiative power sink due to H excitation');
	plt.plot(time_crop, np.sum(power_rad_rec_bremm*area*length,axis=1), label='Radiative power sink due to recombination and Bremsstrahlung');
	plt.plot(time_crop, np.sum(power_rad_mol*area*length,axis=1), label='Radiative power sink due molecule induced H de-excitation');
	plt.plot(time_crop, np.sum((power_rad_mol+power_rad_excit+power_rad_rec_bremm)*area*length,axis=1),'--', label='Sum of radiative losses');
	plt.plot(time_crop, np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1), label='Power sink due to atomic ionisation');
	plt.plot(time_crop, np.sum(effective_recombination_rates*ionisation_potential*area*length/J_to_eV,axis=1), label='Power sink due to atomic recombination');
	plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
	plt.plot(time_crop, heat_inflow_upstream_min,'--k');
	plt.plot(time_crop, np.sum(target_heat_flow*area,axis=1), label='Power sink at target:\n'+label_ion_sink_at_target);
	plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--',label='Power from plasma source')
	plt.legend(loc='best', fontsize='xx-small')
	plt.grid()
	plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Power loss [W]')
	plt.title(pre_title+'Time evolution of the power balance summed over radious')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(12, 6));
	plt.plot(time_crop, np.sum(power_rad_excit*area*length,axis=1), label='Radiative power sink due to H excitation');
	plt.plot(time_crop, np.sum(power_rad_rec_bremm*area*length,axis=1), label='Radiative power sink due to recombination and Bremsstrahlung');
	plt.plot(time_crop, np.sum(power_rad_mol*area*length,axis=1), label='Radiative power sink due molecule induced H de-excitation');
	plt.plot(time_crop, np.sum((power_rad_mol+power_rad_excit+power_rad_rec_bremm)*area*length,axis=1),'--', label='Sum of radiative losses');
	plt.plot(time_crop, np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1), label='Power sink due to atomic ionisation');
	plt.plot(time_crop, np.sum(effective_recombination_rates*ionisation_potential*area*length/J_to_eV,axis=1), label='Power sink due to atomic recombination');
	plt.plot(time_crop, np.sum(effective_recombination_rates*ionisation_potential*area*length/J_to_eV,axis=1) + np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1) + np.sum((power_rad_mol+power_rad_excit+power_rad_rec_bremm)*area*length,axis=1), label='Recombination + ionisation + radiation');
	plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
	plt.plot(time_crop, heat_inflow_upstream_min,'--k');
	plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--',label='Power from plasma source')
	# plt.plot(time_crop, np.sum(target_heat_flow*area,axis=1), label='Power sink at target:\n'+label_ion_sink_at_target);
	plt.legend(loc='best', fontsize='xx-small')
	plt.grid()
	# plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Power loss [W]')
	plt.title(pre_title+'Time evolution of the power balance summed over radious')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	plt.figure(figsize=(12, 6));
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(power_rad_excit*area*length,axis=1)), label='Radiative power sink due to H excitation');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(power_rad_rec_bremm*area*length,axis=1)), label='Radiative power sink due to recombination and Bremsstrahlung');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(power_rad_mol*area*length,axis=1)), label='Radiative power sink due molecule induced H de-excitation');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum((power_rad_mol+power_rad_excit+power_rad_rec_bremm)*area*length,axis=1)),'--', label='Sum of radiative losses');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1)), label='Power sink due to atomic ionisation');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(effective_recombination_rates*ionisation_potential*area*length/J_to_eV,axis=1)), label='Power sink due to atomic recombination');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(np.sum(effective_recombination_rates*ionisation_potential*area*length/J_to_eV,axis=1) + np.sum(effective_ionisation_rates*ionisation_potential*area*length/J_to_eV,axis=1) + np.sum((power_rad_mol+power_rad_excit+power_rad_rec_bremm)*area*length,axis=1)), label='Recombination + ionisation + radiation');
	plt.plot(time_crop, 1e-3*dt*np.cumsum(heat_inflow_upstream_max),'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
	plt.plot(time_crop, 1e-3*dt*np.cumsum(heat_inflow_upstream_min),'--k');
	plt.plot(time_source_power_crop,1e-3*np.mean(np.diff(time_source_power_crop))*np.cumsum(power_pulse_shape_crop),'--',label='Power from plasma source')
	# plt.plot(time_crop, np.sum(target_heat_flow*area,axis=1), label='Power sink at target:\n'+label_ion_sink_at_target);
	plt.legend(loc='best', fontsize='x-small', framealpha=0.1)
	plt.grid()
	# plt.semilogy()
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Energy loss [J]')
	plt.title(pre_title+'Cumulative power balance summed over radious')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()


	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
	# plasma_inflow_upstream_max = np.sum(area * ne_all*1e20 * 10000,axis=1)	# W
	# plasma_inflow_upstream_min = np.sum(area * ne_all*1e20 * 1000,axis=1)	# W
	# target_adiabatic_collisional_velocity = ((Te_all + 5/3 *T_Hp)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
	# target_Bohm_adiabatic_flow = 0.5*ne_all*1e20*target_adiabatic_collisional_velocity	# 0.5 time reduction on ne in the presheath
	plt.figure(figsize=(20, 10));
	threshold_radious = 0.025
	plt.plot(time_crop, np.sum(area * target_Bohm_adiabatic_flow,axis=1),'--', label='target Bohm flow');
	plt.plot(time_crop, plasma_inflow_upstream_max,'--', label='Max upstream inflow');
	plt.plot(time_crop, plasma_inflow_upstream_min,'--', label='Min upstream inflow');
	plt.plot(time_crop, np.sum(area *length* effective_ionisation_rates,axis=1), label='ionisation rate from ADAS');
	plt.plot(time_crop, np.sum(area *length* effective_recombination_rates,axis=1), label='recombination rate from ADAS');
	plt.plot(time_crop, np.sum(area *length* rate_creation_H,axis=1), label='H creation rate from molecules');
	plt.plot(time_crop, np.sum(area *length* rate_destruction_H,axis=1), label='H destruction rate from molecules');
	plt.plot(time_crop, np.sum(area *length* rate_creation_Hp,axis=1), label='H+ creation rate from molecules');
	plt.plot(time_crop, np.sum(area *length* rate_destruction_Hp,axis=1), label='H+ destruction rate from molecules');
	plt.plot(time_crop, np.sum(area *length* rate_creation_e,axis=1), label='e- creation rate from molecules');
	plt.plot(time_crop, np.sum(area *length* rate_destruction_e,axis=1), label='e- destruction rate from molecules');
	# plt.plot(time_crop, np.sum(area * Te_all,axis=1)/np.max(np.sum(area * Te_all,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * Te_all,axis=1)/np.sum(area))));
	# plt.plot(time_crop, np.sum(area * ne_all,axis=1)/np.max(np.sum(area * ne_all,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * ne_all,axis=1)/np.sum(area))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Particles source/sink [#/s]')
	plt.title(pre_title+'Time evolution of the radially averaged reaction rates\ncomparison of atomic vs molecular processes\n'+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
	# plasma_inflow_upstream_max = np.sum(area * ne_all*1e20 * 10000,axis=1)	# W
	# plasma_inflow_upstream_min = np.sum(area * ne_all*1e20 * 1000,axis=1)	# W
	# target_adiabatic_collisional_velocity = ((Te_all + 5/3 *T_Hp)/eV_to_K*boltzmann_constant_J/hydrogen_mass)**0.5
	# target_Bohm_adiabatic_flow = 0.5*ne_all*1e20*target_adiabatic_collisional_velocity	# 0.5 time reduction on ne in the presheath
	plt.figure(figsize=(20, 10));
	threshold_radious = 0.025
	# plt.plot(time_crop, np.sum(area * target_Bohm_adiabatic_flow,axis=1),'--', label='target Bohm flow');
	plt.plot(time_crop, plasma_inflow_upstream_max,'--', label='Max upstream inflow');
	plt.plot(time_crop, plasma_inflow_upstream_min,'--', label='Min upstream inflow');
	plt.plot(time_crop, np.sum(area *length* effective_ionisation_rates,axis=1), label='ionisation rate from ADAS');
	plt.plot(time_crop, np.sum(area *length* effective_recombination_rates,axis=1), label='recombination rate from ADAS');
	# plt.plot(time_crop, np.sum(area *length* rate_creation_H,axis=1), label='H creation rate from molecules');
	# plt.plot(time_crop, np.sum(area *length* rate_destruction_H,axis=1), label='H destruction rate from molecules');
	# plt.plot(time_crop, np.sum(area *length* rate_creation_Hp,axis=1), label='H+ creation rate from molecules');
	# plt.plot(time_crop, np.sum(area *length* rate_destruction_Hp,axis=1), label='H+ destruction rate from molecules');
	# plt.plot(time_crop, np.sum(area *length* rate_creation_e,axis=1), label='e- creation rate from molecules');
	# plt.plot(time_crop, np.sum(area *length* rate_destruction_e,axis=1), label='e- destruction rate from molecules');
	# plt.plot(time_crop, np.sum(area * Te_all,axis=1)/np.max(np.sum(area * Te_all,axis=1)),'--', label='Te TS\n(max='+'%.3g eV)' % (np.max(np.sum(area * Te_all,axis=1)/np.sum(area))));
	# plt.plot(time_crop, np.sum(area * ne_all,axis=1)/np.max(np.sum(area * ne_all,axis=1)),'--', label='ne TS\n(max='+'%.3g 10^20 # m^-3)' % (np.max(np.sum(area * ne_all,axis=1)/np.sum(area))));
	plt.legend(loc='best', fontsize='xx-small')
	plt.xlabel('time from beginning of pulse [ms]')
	plt.ylabel('Particles source/sink [#/s]')
	plt.title(pre_title+'Time evolution of the radially averaged reaction rates\ncomparison of atomic vs molecular processes\n'+str(target_OES_distance)+'mm from the target')
	figure_index += 1
	plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		figure_index) + '.eps', bbox_inches='tight')
	plt.close()

	np.savez_compressed(path_where_to_save_everything + mod4 +'/molecular_results'+str(global_pass),ionization_length_H=ionization_length_H,rate_creation_Hm=rate_creation_Hm,rate_destruction_Hm=rate_destruction_Hm,ionization_length_Hm=ionization_length_Hm,rate_creation_H2p=rate_creation_H2p,rate_destruction_H2p=rate_destruction_H2p,ionization_length_H2p=ionization_length_H2p,rate_creation_H2=rate_creation_H2,rate_destruction_H2=rate_destruction_H2,ionization_length_H2=ionization_length_H2)



	if collect_power_PDF:

		intervals_power_rad_excit = np.zeros_like(Te_all).tolist()
		prob_power_rad_excit = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_excit = np.zeros_like(Te_all).tolist()
		intervals_power_rad_rec_bremm = np.zeros_like(Te_all).tolist()
		prob_power_rad_rec_bremm = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_rec_bremm = np.zeros_like(Te_all).tolist()
		intervals_power_rad_mol = np.zeros_like(Te_all).tolist()
		prob_power_rad_mol = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_mol = np.zeros_like(Te_all).tolist()
		intervals_power_via_ionisation = np.zeros_like(Te_all).tolist()
		prob_power_via_ionisation = np.zeros_like(Te_all).tolist()
		actual_values_power_via_ionisation = np.zeros_like(Te_all).tolist()
		intervals_power_via_recombination = np.zeros_like(Te_all).tolist()
		prob_power_via_recombination = np.zeros_like(Te_all).tolist()
		actual_values_power_via_recombination = np.zeros_like(Te_all).tolist()
		intervals_tot_rad_power = np.zeros_like(Te_all).tolist()
		prob_tot_rad_power = np.zeros_like(Te_all).tolist()
		actual_values_tot_rad_power = np.zeros_like(Te_all).tolist()
		intervals_power_rad_Hm = np.zeros_like(Te_all).tolist()
		prob_power_rad_Hm = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_Hm = np.zeros_like(Te_all).tolist()
		intervals_power_rad_H2 = np.zeros_like(Te_all).tolist()
		prob_power_rad_H2 = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_H2 = np.zeros_like(Te_all).tolist()
		intervals_power_rad_H2p = np.zeros_like(Te_all).tolist()
		prob_power_rad_H2p = np.zeros_like(Te_all).tolist()
		actual_values_power_rad_H2p = np.zeros_like(Te_all).tolist()
		intervals_power_heating_rec = np.zeros_like(Te_all).tolist()
		prob_power_heating_rec = np.zeros_like(Te_all).tolist()
		actual_values_power_heating_rec = np.zeros_like(Te_all).tolist()
		intervals_power_rec_neutral = np.zeros_like(Te_all).tolist()
		prob_power_rec_neutral = np.zeros_like(Te_all).tolist()
		actual_values_power_rec_neutral = np.zeros_like(Te_all).tolist()
		intervals_power_via_brem = np.zeros_like(Te_all).tolist()
		prob_power_via_brem = np.zeros_like(Te_all).tolist()
		actual_values_power_via_brem = np.zeros_like(Te_all).tolist()
		intervals_total_removed_power = np.zeros_like(Te_all).tolist()
		prob_total_removed_power = np.zeros_like(Te_all).tolist()
		actual_values_total_removed_power = np.zeros_like(Te_all).tolist()
		intervals_local_CX = np.zeros_like(Te_all).tolist()
		prob_local_CX = np.zeros_like(Te_all).tolist()
		actual_values_local_CX = np.zeros_like(Te_all).tolist()
		intervals_H_destruction_RR = np.zeros_like(Te_all).tolist()
		prob_H_destruction_RR = np.zeros_like(Te_all).tolist()
		actual_values_H_destruction_RR = np.zeros_like(Te_all).tolist()
		intervals_eff_CX_RR = np.zeros_like(Te_all).tolist()
		prob_eff_CX_RR = np.zeros_like(Te_all).tolist()
		actual_values_eff_CX_RR = np.zeros_like(Te_all).tolist()
		intervals_H2_destruction_RR = np.zeros_like(Te_all).tolist()
		prob_H2_destruction_RR = np.zeros_like(Te_all).tolist()
		actual_values_H2_destruction_RR = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_1 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_1 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_1 = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_2 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_2 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_2 = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_3 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_3 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_3 = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_4 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_4 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_4 = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_5 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_5 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_5 = np.zeros_like(Te_all).tolist()
		intervals_CX_term_1_6 = np.zeros_like(Te_all).tolist()
		prob_CX_term_1_6 = np.zeros_like(Te_all).tolist()
		actual_values_CX_term_1_6 = np.zeros_like(Te_all).tolist()

		for i_t in range(np.shape(Te_all)[0]):
			for i_r in range(np.shape(Te_all)[1]):
				i = i_t*np.shape(Te_all)[1] + i_r
				if power_balance_data[i] == 0:
					intervals_power_rad_excit[i_t][i_r] = [0,0]
					prob_power_rad_excit[i_t][i_r] = [1]
					actual_values_power_rad_excit[i_t][i_r] = [0]
					intervals_power_rad_rec_bremm[i_t][i_r] = [0,0]
					prob_power_rad_rec_bremm[i_t][i_r] = [1]
					actual_values_power_rad_rec_bremm[i_t][i_r] = [0]
					intervals_power_rad_mol[i_t][i_r] = [0,0]
					prob_power_rad_mol[i_t][i_r] = [1]
					actual_values_power_rad_mol[i_t][i_r] = [0]
					intervals_power_via_ionisation[i_t][i_r] = [0,0]
					prob_power_via_ionisation[i_t][i_r] = [1]
					actual_values_power_via_ionisation[i_t][i_r] = [0]
					intervals_power_via_recombination[i_t][i_r] = [0,0]
					prob_power_via_recombination[i_t][i_r] = [1]
					actual_values_power_via_recombination[i_t][i_r] = [0]
					intervals_tot_rad_power[i_t][i_r] = [0,0]
					prob_tot_rad_power[i_t][i_r] = [1]
					actual_values_tot_rad_power[i_t][i_r] = [0]
					intervals_power_rad_Hm[i_t][i_r] = [0,0]
					prob_power_rad_Hm[i_t][i_r] = [1]
					actual_values_power_rad_Hm[i_t][i_r] = [0]
					intervals_power_rad_H2[i_t][i_r] = [0,0]
					prob_power_rad_H2[i_t][i_r] = [1]
					actual_values_power_rad_H2[i_t][i_r] = [0]
					intervals_power_rad_H2p[i_t][i_r] = [0,0]
					prob_power_rad_H2p[i_t][i_r] = [1]
					actual_values_power_rad_H2p[i_t][i_r] = [0]
					intervals_power_heating_rec[i_t][i_r] = [0,0]
					prob_power_heating_rec[i_t][i_r] = [1]
					actual_values_power_heating_rec[i_t][i_r] = [0]
					intervals_power_rec_neutral[i_t][i_r] = [0,0]
					prob_power_rec_neutral[i_t][i_r] = [1]
					actual_values_power_rec_neutral[i_t][i_r] = [0]
					intervals_power_via_brem[i_t][i_r] = [0,0]
					prob_power_via_brem[i_t][i_r] = [1]
					actual_values_power_via_brem[i_t][i_r] = [0]
					intervals_total_removed_power[i_t][i_r] = [0,0]
					prob_total_removed_power[i_t][i_r] = [1]
					actual_values_total_removed_power[i_t][i_r] = [0]
					intervals_local_CX[i_t][i_r] = [0,0]
					prob_local_CX[i_t][i_r] = [1]
					actual_values_local_CX[i_t][i_r] = [0]
					intervals_H_destruction_RR[i_t][i_r] = [0,0]
					prob_H_destruction_RR[i_t][i_r] = [1]
					actual_values_H_destruction_RR[i_t][i_r] = [0]
					intervals_eff_CX_RR[i_t][i_r] = [0,0]
					prob_eff_CX_RR[i_t][i_r] = [1]
					actual_values_eff_CX_RR[i_t][i_r] = [0]
					intervals_H2_destruction_RR[i_t][i_r] = [0,0]
					prob_H2_destruction_RR[i_t][i_r] = [1]
					actual_values_H2_destruction_RR[i_t][i_r] = [0]
					intervals_CX_term_1_1[i_t][i_r] = [0,0]
					prob_CX_term_1_1[i_t][i_r] = [1]
					actual_values_CX_term_1_1[i_t][i_r] = [0]
					intervals_CX_term_1_2[i_t][i_r] = [0,0]
					prob_CX_term_1_2[i_t][i_r] = [1]
					actual_values_CX_term_1_2[i_t][i_r] = [0]
					intervals_CX_term_1_3[i_t][i_r] = [0,0]
					prob_CX_term_1_3[i_t][i_r] = [1]
					actual_values_CX_term_1_3[i_t][i_r] = [0]
					intervals_CX_term_1_4[i_t][i_r] = [0,0]
					prob_CX_term_1_4[i_t][i_r] = [1]
					actual_values_CX_term_1_4[i_t][i_r] = [0]
					intervals_CX_term_1_5[i_t][i_r] = [0,0]
					prob_CX_term_1_5[i_t][i_r] = [1]
					actual_values_CX_term_1_5[i_t][i_r] = [0]
					intervals_CX_term_1_6[i_t][i_r] = [0,0]
					prob_CX_term_1_6[i_t][i_r] = [1]
					actual_values_CX_term_1_6[i_t][i_r] = [0]
				else:
					intervals_power_rad_excit[i_t][i_r] = power_balance_data[i][0]
					prob_power_rad_excit[i_t][i_r] = power_balance_data[i][1]/np.sum(power_balance_data[i][1])
					actual_values_power_rad_excit[i_t][i_r] = power_balance_data[i][2]
					intervals_power_rad_rec_bremm[i_t][i_r] = power_balance_data[i][3]
					prob_power_rad_rec_bremm[i_t][i_r] = power_balance_data[i][4]/np.sum(power_balance_data[i][4])
					actual_values_power_rad_rec_bremm[i_t][i_r] = power_balance_data[i][5]
					intervals_power_rad_mol[i_t][i_r] = power_balance_data[i][6]
					prob_power_rad_mol[i_t][i_r] = power_balance_data[i][7]/np.sum(power_balance_data[i][7])
					actual_values_power_rad_mol[i_t][i_r] = power_balance_data[i][8]
					intervals_power_via_ionisation[i_t][i_r] = power_balance_data[i][9]
					prob_power_via_ionisation[i_t][i_r] = power_balance_data[i][10]/np.sum(power_balance_data[i][10])
					actual_values_power_via_ionisation[i_t][i_r] = power_balance_data[i][11]
					intervals_power_via_recombination[i_t][i_r] = power_balance_data[i][12]
					prob_power_via_recombination[i_t][i_r] = power_balance_data[i][13]/np.sum(power_balance_data[i][13])
					actual_values_power_via_recombination[i_t][i_r] = power_balance_data[i][14]
					intervals_tot_rad_power[i_t][i_r] = power_balance_data[i][15]
					prob_tot_rad_power[i_t][i_r] = power_balance_data[i][16]/np.sum(power_balance_data[i][16])
					actual_values_tot_rad_power[i_t][i_r] = power_balance_data[i][17]
					intervals_power_rad_Hm[i_t][i_r] = power_balance_data[i][18]
					prob_power_rad_Hm[i_t][i_r] = power_balance_data[i][19]/np.sum(power_balance_data[i][19])
					actual_values_power_rad_Hm[i_t][i_r] = power_balance_data[i][20]
					intervals_power_rad_H2[i_t][i_r] = power_balance_data[i][21]
					prob_power_rad_H2[i_t][i_r] = power_balance_data[i][22]/np.sum(power_balance_data[i][22])
					actual_values_power_rad_H2[i_t][i_r] = power_balance_data[i][23]
					intervals_power_rad_H2p[i_t][i_r] = power_balance_data[i][24]
					prob_power_rad_H2p[i_t][i_r] = power_balance_data[i][25]/np.sum(power_balance_data[i][25])
					actual_values_power_rad_H2p[i_t][i_r] = power_balance_data[i][26]
					intervals_power_heating_rec[i_t][i_r] = power_balance_data[i][27]
					prob_power_heating_rec[i_t][i_r] = power_balance_data[i][28]/np.sum(power_balance_data[i][28])
					actual_values_power_heating_rec[i_t][i_r] = power_balance_data[i][29]
					intervals_power_rec_neutral[i_t][i_r] = power_balance_data[i][30]
					prob_power_rec_neutral[i_t][i_r] = power_balance_data[i][31]/np.sum(power_balance_data[i][31])
					actual_values_power_rec_neutral[i_t][i_r] = power_balance_data[i][32]
					intervals_power_via_brem[i_t][i_r] = power_balance_data[i][33]
					prob_power_via_brem[i_t][i_r] = power_balance_data[i][34]/np.sum(power_balance_data[i][34])
					actual_values_power_via_brem[i_t][i_r] = power_balance_data[i][35]
					intervals_total_removed_power[i_t][i_r] = power_balance_data[i][36]
					prob_total_removed_power[i_t][i_r] = power_balance_data[i][37]/np.sum(power_balance_data[i][37])
					actual_values_total_removed_power[i_t][i_r] = power_balance_data[i][38]
					intervals_local_CX[i_t][i_r] = power_balance_data[i][72]
					prob_local_CX[i_t][i_r] = power_balance_data[i][73]/np.sum(power_balance_data[i][73])
					actual_values_local_CX[i_t][i_r] = power_balance_data[i][74]
					intervals_H_destruction_RR[i_t][i_r] = power_balance_data[i][75]
					prob_H_destruction_RR[i_t][i_r] = power_balance_data[i][76]/np.sum(power_balance_data[i][76])
					actual_values_H_destruction_RR[i_t][i_r] = power_balance_data[i][77]
					intervals_eff_CX_RR[i_t][i_r] = power_balance_data[i][78]
					prob_eff_CX_RR[i_t][i_r] = power_balance_data[i][79]/np.sum(power_balance_data[i][79])
					actual_values_eff_CX_RR[i_t][i_r] = power_balance_data[i][80]
					intervals_H2_destruction_RR[i_t][i_r] = power_balance_data[i][81]
					prob_H2_destruction_RR[i_t][i_r] = power_balance_data[i][82]/np.sum(power_balance_data[i][82])
					actual_values_H2_destruction_RR[i_t][i_r] = power_balance_data[i][83]
					intervals_CX_term_1_1[i_t][i_r] = power_balance_data[i][84]
					prob_CX_term_1_1[i_t][i_r] = power_balance_data[i][85]/np.sum(power_balance_data[i][85])
					actual_values_CX_term_1_1[i_t][i_r] = power_balance_data[i][86]
					intervals_CX_term_1_2[i_t][i_r] = power_balance_data[i][87]
					prob_CX_term_1_2[i_t][i_r] = power_balance_data[i][88]/np.sum(power_balance_data[i][88])
					actual_values_CX_term_1_2[i_t][i_r] = power_balance_data[i][89]
					intervals_CX_term_1_3[i_t][i_r] = power_balance_data[i][90]
					prob_CX_term_1_3[i_t][i_r] = power_balance_data[i][91]/np.sum(power_balance_data[i][91])
					actual_values_CX_term_1_3[i_t][i_r] = power_balance_data[i][92]
					intervals_CX_term_1_4[i_t][i_r] = power_balance_data[i][93]
					prob_CX_term_1_4[i_t][i_r] = power_balance_data[i][94]/np.sum(power_balance_data[i][94])
					actual_values_CX_term_1_4[i_t][i_r] = power_balance_data[i][95]
					intervals_CX_term_1_5[i_t][i_r] = power_balance_data[i][96]
					prob_CX_term_1_5[i_t][i_r] = power_balance_data[i][97]/np.sum(power_balance_data[i][97])
					actual_values_CX_term_1_5[i_t][i_r] = power_balance_data[i][98]
					intervals_CX_term_1_6[i_t][i_r] = power_balance_data[i][99]
					prob_CX_term_1_6[i_t][i_r] = power_balance_data[i][100]/np.sum(power_balance_data[i][100])
					actual_values_CX_term_1_6[i_t][i_r] = power_balance_data[i][101]

		most_likely_power_rad_excit = []
		for i_t in range(len(prob_power_rad_excit)):
			temp=[]
			for i_r in range(len(prob_power_rad_excit[i_t])):
				# temp.append((np.add(intervals_power_rad_excit[i_t][i_r][1:],intervals_power_rad_excit[i_t][i_r][:-1])/2)[np.array(prob_power_rad_excit[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_excit[i_t][i_r][np.array(prob_power_rad_excit[i_t][i_r]).argmax()])
			most_likely_power_rad_excit.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_excit,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_excit)),np.max(most_likely_power_rad_excit)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_excit')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_rad_rec_bremm = []
		for i_t in range(len(prob_power_rad_rec_bremm)):
			temp=[]
			for i_r in range(len(prob_power_rad_rec_bremm[i_t])):
				# temp.append((np.add(intervals_power_rad_rec_bremm[i_t][i_r][1:],intervals_power_rad_rec_bremm[i_t][i_r][:-1])/2)[np.array(prob_power_rad_rec_bremm[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_rec_bremm[i_t][i_r][np.array(prob_power_rad_rec_bremm[i_t][i_r]).argmax()])
			most_likely_power_rad_rec_bremm.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_rec_bremm,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_rec_bremm)),np.max(most_likely_power_rad_rec_bremm)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_rec_bremm')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_rad_mol = []
		for i_t in range(len(prob_power_rad_mol)):
			temp=[]
			for i_r in range(len(prob_power_rad_mol[i_t])):
				# temp.append((np.add(intervals_power_rad_mol[i_t][i_r][1:],intervals_power_rad_mol[i_t][i_r][:-1])/2)[np.array(prob_power_rad_mol[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_mol[i_t][i_r][np.array(prob_power_rad_mol[i_t][i_r]).argmax()])
			most_likely_power_rad_mol.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_mol,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_mol)),np.max(most_likely_power_rad_mol)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_mol')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_via_ionisation = []
		for i_t in range(len(prob_power_via_ionisation)):
			temp=[]
			for i_r in range(len(prob_power_via_ionisation[i_t])):
				# temp.append((np.add(intervals_power_via_ionisation[i_t][i_r][1:],intervals_power_via_ionisation[i_t][i_r][:-1])/2)[np.array(prob_power_via_ionisation[i_t][i_r]).argmax()])
				temp.append(actual_values_power_via_ionisation[i_t][i_r][np.array(prob_power_via_ionisation[i_t][i_r]).argmax()])
			most_likely_power_via_ionisation.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_via_ionisation,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_via_ionisation)),np.max(most_likely_power_via_ionisation)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_via_ionisation')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_ionisation_rate = np.array(most_likely_power_via_ionisation)/13.6*J_to_eV
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_ionisation_rate,cmap='rainbow',vmin=max(max(1,np.min(most_likely_ionisation_rate)),np.max(most_likely_ionisation_rate)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('reaction rate [#/m3 s]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of ionisation rate')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		# Calculation of the ionisation mean free path
		thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
		ionization_length_H = thermal_velocity_H/(most_likely_ionisation_rate / (ne_all * 1e20*nH_ne_all) )
		ionization_length_H = np.where(np.isnan(ionization_length_H), np.inf, ionization_length_H)
		# ionization_length_H = np.where(np.isinf(ionization_length_H), np.nan, ionization_length_H)
		# ionization_length_H = np.where(np.isnan(ionization_length_H), np.nanmax(ionization_length_H[np.isfinite(ionization_length_H)]), ionization_length_H)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, ionization_length_H,vmax=min(np.max(ionization_length_H),1), cmap='rainbow', norm=LogNorm());
		plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)')
		temp1 = np.flip(np.cumsum(np.flip(dx/ionization_length_H,axis=1),axis=1),axis=1)
		temp1 = 1-temp1
		temp1[temp1<0]=0
		temp1[:,-1][temp1[:,-1]==1]=1
		plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
		temp2 = np.cumsum(dx/ionization_length_H,axis=1)
		temp2 = 1-temp2
		temp2[temp2<0]=0
		temp2[:,0][temp2[:,0]==0]=1
		plt.legend(loc='best', fontsize='xx-small')
		# plt.colorbar(orientation="horizontal").set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
		cb = plt.colorbar(orientation="horizontal",format='%.3g').set_label('ionization_length [m], limited to 1m')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'ionization length of neutral H from Bayesian ionisation power\ncold H (from simul)')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, (np.array(most_likely_power_rad_excit)*J_to_eV +np.array(most_likely_ionisation_rate)*13.6)/np.array(most_likely_ionisation_rate),cmap='rainbow',vmax=10000, norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('energy per ionisation [eV]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most energy excitation cost per atomic ionisation')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_power_via_recombination = []
		for i_t in range(len(prob_power_via_recombination)):
			temp=[]
			for i_r in range(len(prob_power_via_recombination[i_t])):
				# temp.append((np.add(intervals_power_via_recombination[i_t][i_r][1:],intervals_power_via_recombination[i_t][i_r][:-1])/2)[np.array(prob_power_via_recombination[i_t][i_r]).argmax()])
				temp.append(actual_values_power_via_recombination[i_t][i_r][np.array(prob_power_via_recombination[i_t][i_r]).argmax()])
			most_likely_power_via_recombination.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_via_recombination,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_via_recombination)),np.max(most_likely_power_via_recombination)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_via_recombination')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_recombination_rate = np.array(most_likely_power_via_recombination)/13.6*J_to_eV
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_recombination_rate,cmap='rainbow',vmin=max(max(1,np.min(most_likely_recombination_rate)),np.max(most_likely_recombination_rate)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('reaction rate [#/m3 s]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of recombination rate')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_tot_rad_power = []
		for i_t in range(len(prob_tot_rad_power)):
			temp=[]
			for i_r in range(len(prob_tot_rad_power[i_t])):
				# temp.append((np.add(intervals_tot_rad_power[i_t][i_r][1:],intervals_tot_rad_power[i_t][i_r][:-1])/2)[np.array(prob_tot_rad_power[i_t][i_r]).argmax()])
				temp.append(actual_values_tot_rad_power[i_t][i_r][np.array(prob_tot_rad_power[i_t][i_r]).argmax()])
			most_likely_tot_rad_power.append(temp)
		most_likely_tot_rad_power = np.array(most_likely_tot_rad_power)

		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		averaged_radiated_power_sigma = []
		for index in range(np.shape(most_likely_tot_rad_power)[0]):
			yy = most_likely_tot_rad_power[index]
			if (np.sum(yy>0)==0):
				averaged_radiated_power_sigma.append(0)
				continue
			p0 = [np.max(yy), np.max(r_crop)/2, 0]
			bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
			fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds)#, sigma=yy_sigma)
			averaged_radiated_power_sigma.append(fit[0][-2])
		averaged_radiated_power_sigma = np.array(averaged_radiated_power_sigma)

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_tot_rad_power,cmap='rainbow',vmin=max(max(1,np.min(most_likely_tot_rad_power)),np.max(most_likely_tot_rad_power)*1e-6), norm=LogNorm());
		plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)\nmean %.3g' %(np.nanmean(averaged_profile_sigma[averaged_profile_sigma>0]*2.355/2)))
		plt.plot(time_crop,averaged_radiated_power_sigma*2.355/2,'-.',color='grey',label='tot_rad_power FWHM, mean %.3g' %(np.nanmean(averaged_radiated_power_sigma[averaged_radiated_power_sigma>0]*2.355/2)))
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.legend(loc='best', fontsize='xx-small')
		plt.title(pre_title+'Most likely values of tot_rad_power')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_tot_rad_brightness = []
		for i_r in range(len(r_crop)):
			most_likely_tot_rad_brightness.append( np.trapz(2*most_likely_tot_rad_power[:,i_r+1:] * r_crop[i_r+1:] / ((r_crop[i_r+1:]**2 - r_crop[i_r]**2)**0.5),dx=dr,axis=-1) )
		most_likely_tot_rad_brightness = np.array(most_likely_tot_rad_brightness).T

		gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
		averaged_radiated_power_sigma = []
		for index in range(np.shape(most_likely_tot_rad_brightness)[0]):
			yy = most_likely_tot_rad_brightness[index]
			if (np.sum(yy>0)==0):
				averaged_radiated_power_sigma.append(0)
				continue
			p0 = [np.max(yy), np.max(r_crop)/2, 0]
			bds = [[0, 0, -interp_range_r/1000], [np.inf, np.max(r_crop), interp_range_r/1000]]
			fit = curve_fit(gauss, r_crop, yy, p0, maxfev=100000, bounds=bds)#, sigma=yy_sigma)
			averaged_radiated_power_sigma.append(fit[0][-2])
		averaged_radiated_power_sigma = np.array(averaged_radiated_power_sigma)

		# temp = []
		# for i_t in range(len(time_crop)):
		# 	temp.append((most_likely_tot_rad_brightness[i_t]<max(1e4,np.min(most_likely_tot_rad_brightness[i_t]))).argmax())
		# temp = np.array(temp)

		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_tot_rad_brightness,cmap='rainbow',vmin=max(max(1,np.min(most_likely_tot_rad_brightness)),np.max(most_likely_tot_rad_brightness)*1e-6), norm=LogNorm());
		plt.plot(time_crop,averaged_profile_sigma*2.355/2,'--',color='grey',label='density FWHM\n(gaussian fit)\nmean %.3g' %(np.nanmean(averaged_profile_sigma[averaged_profile_sigma>0]*2.355/2)))
		plt.plot(time_crop,averaged_radiated_power_sigma*2.355/2,'-.',color='grey',label='tot_rad_brightness FWHM, mean %.3g' %(np.nanmean(averaged_radiated_power_sigma[averaged_radiated_power_sigma>0]*2.355/2)))
		plt.colorbar(orientation="horizontal").set_label('power [W/m2]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.legend(loc='best', fontsize='xx-small')
		plt.title(pre_title+'Most likely values of tot_rad_brightness')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		temp = (np.mean(most_likely_tot_rad_brightness,axis=0)<max(1e5,np.min(most_likely_tot_rad_brightness[i_t]))).argmax()

		plt.figure(figsize=(8, 5));
		plt.plot(r_crop,np.mean(most_likely_tot_rad_brightness,axis=0))
		plt.plot([r_crop.min(),r_crop.max()],[1e5]*2,'--k')
		plt.plot([r_crop[temp]]*2,[0,1e6],'--k')
		plt.title(pre_title+'average in time of tot_rad_brightness to find the diameter of the rediating region')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		# temp2 = []
		# for i_t in range(len(time_crop)):
		# 	temp2.append(np.mean(most_likely_tot_rad_brightness[i_t,:temp[i_t]]))
		# temp2 = np.array(temp2)/(np.mean(r_crop[temp[temp>0]])**2)

		plt.figure(figsize=(8, 5));
		plt.plot(time_crop,np.mean(most_likely_tot_rad_brightness[:,:temp],axis=-1)/((2*r_crop[temp])**2))
		plt.xlabel('time [ms]')
		plt.ylabel('average power/diameter [W/m]      ')
		plt.title(pre_title+'average tot_rad_brightness where time average >1e5 divided diameter (%.3gmm) square' %(1000*2*r_crop[temp]))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_power_rad_Hm = []
		for i_t in range(len(prob_power_rad_Hm)):
			temp=[]
			for i_r in range(len(prob_power_rad_Hm[i_t])):
				# temp.append((np.add(intervals_power_rad_Hm[i_t][i_r][1:],intervals_power_rad_Hm[i_t][i_r][:-1])/2)[np.array(prob_power_rad_Hm[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_Hm[i_t][i_r][np.array(prob_power_rad_Hm[i_t][i_r]).argmax()])
			most_likely_power_rad_Hm.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_Hm,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_Hm)),np.max(most_likely_power_rad_Hm)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_Hm')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_rad_H2 = []
		for i_t in range(len(prob_power_rad_H2)):
			temp=[]
			for i_r in range(len(prob_power_rad_H2[i_t])):
				# temp.append((np.add(intervals_power_rad_H2[i_t][i_r][1:],intervals_power_rad_H2[i_t][i_r][:-1])/2)[np.array(prob_power_rad_H2[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_H2[i_t][i_r][np.array(prob_power_rad_H2[i_t][i_r]).argmax()])
			most_likely_power_rad_H2.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_H2,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_H2)),np.max(most_likely_power_rad_H2)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_H2')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_rad_H2p = []
		for i_t in range(len(prob_power_rad_H2p)):
			temp=[]
			for i_r in range(len(prob_power_rad_H2p[i_t])):
				# temp.append((np.add(intervals_power_rad_H2p[i_t][i_r][1:],intervals_power_rad_H2p[i_t][i_r][:-1])/2)[np.array(prob_power_rad_H2p[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rad_H2p[i_t][i_r][np.array(prob_power_rad_H2p[i_t][i_r]).argmax()])
			most_likely_power_rad_H2p.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rad_H2p,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rad_H2p)),np.max(most_likely_power_rad_H2p)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rad_H2p')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_heating_rec = []
		for i_t in range(len(prob_power_heating_rec)):
			temp=[]
			for i_r in range(len(prob_power_heating_rec[i_t])):
				# temp.append((np.add(intervals_power_heating_rec[i_t][i_r][1:],intervals_power_heating_rec[i_t][i_r][:-1])/2)[np.array(prob_power_heating_rec[i_t][i_r]).argmax()])
				temp.append(actual_values_power_heating_rec[i_t][i_r][np.array(prob_power_heating_rec[i_t][i_r]).argmax()])
			most_likely_power_heating_rec.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_heating_rec,cmap='rainbow');
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_heating_rec')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_rec_neutral = []
		for i_t in range(len(prob_power_rec_neutral)):
			temp=[]
			for i_r in range(len(prob_power_rec_neutral[i_t])):
				# temp.append((np.add(intervals_power_rec_neutral[i_t][i_r][1:],intervals_power_rec_neutral[i_t][i_r][:-1])/2)[np.array(prob_power_rec_neutral[i_t][i_r]).argmax()])
				temp.append(actual_values_power_rec_neutral[i_t][i_r][np.array(prob_power_rec_neutral[i_t][i_r]).argmax()])
			most_likely_power_rec_neutral.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_rec_neutral,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_rec_neutral)),np.max(most_likely_power_rec_neutral)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_rec_neutral')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_power_via_brem = []
		for i_t in range(len(prob_power_via_brem)):
			temp=[]
			for i_r in range(len(prob_power_via_brem[i_t])):
				# temp.append((np.add(intervals_power_via_brem[i_t][i_r][1:],intervals_power_via_brem[i_t][i_r][:-1])/2)[np.array(prob_power_via_brem[i_t][i_r]).argmax()])
				temp.append(actual_values_power_via_brem[i_t][i_r][np.array(prob_power_via_brem[i_t][i_r]).argmax()])
			most_likely_power_via_brem.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_power_via_brem,cmap='rainbow',vmin=max(max(1,np.min(most_likely_power_via_brem)),np.max(most_likely_power_via_brem)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of power_via_brem')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()
		most_likely_total_removed_power = []
		for i_t in range(len(prob_total_removed_power)):
			temp=[]
			for i_r in range(len(prob_total_removed_power[i_t])):
				# temp.append((np.add(intervals_total_removed_power[i_t][i_r][1:],intervals_total_removed_power[i_t][i_r][:-1])/2)[np.array(prob_total_removed_power[i_t][i_r]).argmax()])
				temp.append(actual_values_total_removed_power[i_t][i_r][np.array(prob_total_removed_power[i_t][i_r]).argmax()])
			most_likely_total_removed_power.append(temp)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_total_removed_power,cmap='rainbow',vmin=max(max(1,np.min(most_likely_total_removed_power)),np.max(most_likely_total_removed_power)*1e-6), norm=LogNorm());
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of total_removed_power\n(ionisation*pot + rad_mol + rad_excit + recombination*pot + brem + rec_neutral)')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
		ionization_length_H = thermal_velocity_H/(most_likely_ionisation_rate / (ne_all * 1e20*nH_ne_all) )
		ionization_length_H = np.where(np.isnan(ionization_length_H), np.inf, ionization_length_H)
		# ionisation_length_H_CX = np.where(np.isinf(ionisation_length_H_CX), np.nan, ionisation_length_H_CX)
		# ionisation_length_H_CX = np.where(np.isnan(ionisation_length_H_CX), np.nanmax(ionisation_length_H_CX[np.isfinite(ionisation_length_H_CX)]), ionisation_length_H_CX)
		most_likely_local_CX = []
		for i_t in range(len(prob_local_CX)):
			temp=[]
			for i_r in range(len(prob_local_CX[i_t])):
				# temp.append((np.add(intervals_local_CX[i_t][i_r][1:],intervals_local_CX[i_t][i_r][:-1])/2)[np.array(prob_local_CX[i_t][i_r]).argmax()])
				temp.append(actual_values_local_CX[i_t][i_r][np.array(prob_local_CX[i_t][i_r]).argmax()])
			most_likely_local_CX.append(temp)
		most_likely_local_CX = np.array(most_likely_local_CX)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, most_likely_local_CX,cmap='rainbow',vmin=max(max(1,np.min(most_likely_local_CX)),np.max(most_likely_local_CX)*1e-6), norm=LogNorm());
		temp1 = np.flip(np.cumsum(np.flip(dx/ionization_length_H,axis=1),axis=1),axis=1)
		temp1 = 1-temp1
		temp1[temp1<0]=0
		temp1[:,-1][temp1[:,-1]==1]=1
		plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Most likely values of local_CX\nPower for atomic hydrogen CX from Bayesian analysis\ncold H (from simul), H destruction RR only ionisation(ADAS)\nlimit on enter=%.3gJ' %(np.sum(most_likely_local_CX*(temp1>0)*area*length*dt/1000)))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_H_destruction_RR = []
		for i_t in range(len(prob_H_destruction_RR)):
			temp=[]
			for i_r in range(len(prob_H_destruction_RR[i_t])):
				# temp.append((np.add(intervals_H_destruction_RR[i_t][i_r][1:],intervals_H_destruction_RR[i_t][i_r][:-1])/2)[np.array(prob_H_destruction_RR[i_t][i_r]).argmax()])
				temp.append(actual_values_H_destruction_RR[i_t][i_r][np.array(prob_H_destruction_RR[i_t][i_r]).argmax()])
			most_likely_H_destruction_RR.append(temp)
		most_likely_H_destruction_RR = np.array(most_likely_H_destruction_RR)
		thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
		destruction_length_H = thermal_velocity_H/(most_likely_H_destruction_RR)
		destruction_length_H = np.where(np.isnan(destruction_length_H), np.inf, destruction_length_H)
		# destruction_length_H = np.where(np.isinf(destruction_length_H), np.nan, destruction_length_H)
		# destruction_length_H = np.where(np.isnan(destruction_length_H), np.nanmax(destruction_length_H[np.isfinite(destruction_length_H)]), destruction_length_H)
		temp = np.flip(np.cumsum(np.flip(dx/destruction_length_H,axis=1),axis=1),axis=1)
		temp = 1-temp
		temp[temp<0]=0
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, destruction_length_H,cmap='rainbow',vmax=min(np.max(destruction_length_H),1), norm=LogNorm());
		temp1 = np.flip(np.cumsum(np.flip(dx/destruction_length_H,axis=1),axis=1),axis=1)
		temp1 = 1-temp1
		temp1[temp1<0]=0
		temp1[:,-1][temp1[:,-1]==1]=1
		plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('H destruction length [m], limited to 1m')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Destruction length of neutral H from Bayesian destruction RR\ncold H (from simul), H destruction RR ADAS+Yacora+AMJUEL\nlimit on enter=%.3gJ' %(np.sum(most_likely_local_CX*temp*area*length*dt/1000)))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_eff_CX_RR = []
		for i_t in range(len(prob_eff_CX_RR)):
			temp=[]
			for i_r in range(len(prob_eff_CX_RR[i_t])):
				# temp.append((np.add(intervals_eff_CX_RR[i_t][i_r][1:],intervals_eff_CX_RR[i_t][i_r][:-1])/2)[np.array(prob_eff_CX_RR[i_t][i_r]).argmax()])
				temp.append(actual_values_eff_CX_RR[i_t][i_r][np.array(prob_eff_CX_RR[i_t][i_r]).argmax()])
			most_likely_eff_CX_RR.append(temp)
		most_likely_eff_CX_RR = np.array(most_likely_eff_CX_RR)
		thermal_velocity_H = ( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5
		CX_length_H = thermal_velocity_H/(most_likely_eff_CX_RR)
		CX_length_H = np.where(np.isnan(CX_length_H), np.inf, CX_length_H)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, CX_length_H,cmap='rainbow',vmax=min(np.max(CX_length_H),1), norm=LogNorm());
		temp1 = np.flip(np.cumsum(np.flip(dx/CX_length_H,axis=1),axis=1),axis=1)
		temp1 = 1-temp1
		temp1[temp1<0]=0
		temp1[:,-1][temp1[:,-1]==1]=1
		max_CX_energy = np.nansum(most_likely_local_CX*(temp1>0)*area*length*dt/1000)
		plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('CX length [m], limited to 1m')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'CX length of neutral H from Bayesian CX RR\ncold H (from simul), CX RR from ADAS\nlimit on enter=%.3gJ' %(max_CX_energy))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		most_likely_H2_destruction_RR = []
		for i_t in range(len(prob_H2_destruction_RR)):
			temp=[]
			for i_r in range(len(prob_H2_destruction_RR[i_t])):
				# temp.append((np.add(intervals_H2_destruction_RR[i_t][i_r][1:],intervals_H2_destruction_RR[i_t][i_r][:-1])/2)[np.array(prob_H2_destruction_RR[i_t][i_r]).argmax()])
				temp.append(actual_values_H2_destruction_RR[i_t][i_r][np.array(prob_H2_destruction_RR[i_t][i_r]).argmax()])
			most_likely_H2_destruction_RR.append(temp)
		most_likely_H2_destruction_RR = np.array(most_likely_H2_destruction_RR)
		thermal_velocity_H2 = ( (T_H2*boltzmann_constant_J)/ (hydrogen_mass*2))**0.5
		destruction_length_H2 = thermal_velocity_H2/(most_likely_H2_destruction_RR)
		destruction_length_H2 = np.where(np.isnan(destruction_length_H2), np.inf, destruction_length_H2)
		# destruction_length_H = np.where(np.isinf(destruction_length_H), np.nan, destruction_length_H)
		# destruction_length_H = np.where(np.isnan(destruction_length_H), np.nanmax(destruction_length_H[np.isfinite(destruction_length_H)]), destruction_length_H)
		plt.figure(figsize=(8, 5));
		plt.pcolor(temp_t, temp_r, destruction_length_H2,cmap='rainbow',vmax=min(np.max(destruction_length_H2),1), norm=LogNorm());
		temp1 = np.flip(np.cumsum(np.flip(dx/destruction_length_H2,axis=1),axis=1),axis=1)
		temp1 = 1-temp1
		temp1[temp1<0]=0
		temp1[:,-1][temp1[:,-1]==1]=1
		plt.plot(time_crop,((temp1==0).argmin(axis=1))*dx,'--',color='b',label='limit full absorption\nfrom outside inward')
		plt.legend(loc='best', fontsize='xx-small')
		plt.colorbar(orientation="horizontal").set_label('destruction length [m], limited to 1m')  # ;plt.pause(0.01)
		plt.axes().set_aspect(20)
		plt.xlabel('time [ms]')
		plt.ylabel('radial location [m]      ')
		plt.title(pre_title+'Destruction length of neutral H2 from Bayesian Destruction RR\ncold H2 (from simul), H2 destruction RR Yacora+AMJUEL')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close()

		if True:
			def PDF_CX_MC(actual_values_CX_term_1_1,prob_CX_term_1_1,actual_values_CX_term_1_2,prob_CX_term_1_2,actual_values_CX_term_1_3,prob_CX_term_1_3,actual_values_CX_term_1_4,prob_CX_term_1_4,actual_values_CX_term_1_5,prob_CX_term_1_5,actual_values_CX_term_1_6,prob_CX_term_1_6,actual_values_H2_destruction_RR,prob_H2_destruction_RR,intervals=30,samples=100000):
				out_values = []
				out_prob_sum = []
				out_actual_values = []
				CX_term_1_1 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				CX_term_1_2 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				CX_term_1_3 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				CX_term_1_4 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				CX_term_1_5 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				CX_term_1_6 = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				H2_destruction_RR = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],samples))
				for i_t in range(np.shape(Te_all)[0]):
					for i_r in range(np.shape(Te_all)[1]):
						if len(actual_values_CX_term_1_1[i_t][i_r])>1:
							CX_term_1_1[i_t][i_r] = np.random.choice(actual_values_CX_term_1_1[i_t][i_r],size=samples,p=prob_CX_term_1_1[i_t][i_r])
							CX_term_1_2[i_t][i_r] = np.random.choice(actual_values_CX_term_1_2[i_t][i_r],size=samples,p=prob_CX_term_1_2[i_t][i_r])
							CX_term_1_3[i_t][i_r] = np.random.choice(actual_values_CX_term_1_3[i_t][i_r],size=samples,p=prob_CX_term_1_3[i_t][i_r])
							CX_term_1_4[i_t][i_r] = np.random.choice(actual_values_CX_term_1_4[i_t][i_r],size=samples,p=prob_CX_term_1_4[i_t][i_r])
							CX_term_1_5[i_t][i_r] = np.random.choice(actual_values_CX_term_1_5[i_t][i_r],size=samples,p=prob_CX_term_1_5[i_t][i_r])
							CX_term_1_6[i_t][i_r] = np.random.choice(actual_values_CX_term_1_6[i_t][i_r],size=samples,p=prob_CX_term_1_6[i_t][i_r])
							H2_destruction_RR[i_t][i_r] = np.random.choice(actual_values_H2_destruction_RR[i_t][i_r],size=samples,p=prob_H2_destruction_RR[i_t][i_r])

				thermal_velocity_Hp = np.transpose([( (T_Hp*boltzmann_constant_J)/ hydrogen_mass)**0.5]*samples,(1,2,0))
				P_HCX = CX_term_1_3*2*target_chamber_pressure/(boltzmann_constant_J*300)* ( (300*boltzmann_constant_J)/ hydrogen_mass)**0.5 * np.exp(-np.flip(np.cumsum(np.flip(CX_term_1_1,axis=1),axis=1),axis=1)-np.flip(np.cumsum(np.flip(CX_term_1_2,axis=1),axis=1),axis=1)/thermal_velocity_Hp)
				thermal_velocity_H = np.transpose([( (T_H*boltzmann_constant_J)/ hydrogen_mass)**0.5]*samples,(1,2,0))
				n_HCX = 2*target_chamber_pressure/(boltzmann_constant_J*300)* ( (300*boltzmann_constant_J)/ hydrogen_mass)**0.5 *CX_term_1_4 * np.exp(-np.flip(np.cumsum(np.flip(CX_term_1_1,axis=1),axis=1),axis=1))
				thermal_velocity_H2 = np.transpose([( (T_H2*boltzmann_constant_J)/ hydrogen_mass)**0.5]*samples,(1,2,0))
				n_H2CX = target_chamber_pressure/(boltzmann_constant_J*300)* ( (300*boltzmann_constant_J)/ (2*hydrogen_mass))**0.5 *CX_term_1_6 * np.exp(-np.flip(np.cumsum(np.flip(CX_term_1_5,axis=1),axis=1),axis=1))
				P_HCX_prob = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				P_HCX_actual_values = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				P_HCX_intervals = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals+1))
				n_HCX_prob = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				n_HCX_actual_values = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				n_HCX_intervals = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals+1))
				n_H2CX_prob = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				n_H2CX_actual_values = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals))
				n_H2CX_intervals = np.zeros((np.shape(Te_all)[0],np.shape(Te_all)[1],intervals+1))
				for i_t in range(np.shape(Te_all)[0]):
					for i_r in range(np.shape(Te_all)[1]):
						if (len(actual_values_CX_term_1_1[i_t][i_r])>1 and np.nanmax(P_HCX[i_t][i_r])>0):
							P_HCX_prob[i_t][i_r],P_HCX_intervals[i_t][i_r] = np.histogram(P_HCX[i_t][i_r],bins=np.logspace(max(-6+np.log10(P_HCX[i_t][i_r].max()),np.log10(P_HCX[i_t][i_r].min())),np.log10(P_HCX[i_t][i_r].max()),intervals+1))
							P_HCX_prob[i_t][i_r] = P_HCX_prob[i_t][i_r]/np.sum(P_HCX_prob[i_t][i_r])
							n_HCX_prob[i_t][i_r],n_HCX_intervals[i_t][i_r] = np.histogram(n_HCX[i_t][i_r],bins=np.logspace(max(-6+np.log10(n_HCX[i_t][i_r].max()),np.log10(n_HCX[i_t][i_r].min())),np.log10(n_HCX[i_t][i_r].max()),intervals+1))
							n_HCX_prob[i_t][i_r] = n_HCX_prob[i_t][i_r]/np.sum(n_HCX_prob[i_t][i_r])
							n_H2CX_prob[i_t][i_r],n_H2CX_intervals[i_t][i_r] = np.histogram(n_H2CX[i_t][i_r],bins=np.logspace(max(-6+np.log10(n_H2CX[i_t][i_r].max()),np.log10(n_H2CX[i_t][i_r].min())),np.log10(n_H2CX[i_t][i_r].max()),intervals+1))
							n_H2CX_prob[i_t][i_r] = n_H2CX_prob[i_t][i_r]/np.sum(n_H2CX_prob[i_t][i_r])
							temp_actual_values = []
							temp_actual_values2 = []
							temp_actual_values3 = []
							for i in range(intervals):
								if i!=intervals-1:
									temp_actual_values.append(np.nanmax([P_HCX_intervals[i_t][i_r][i],np.mean(P_HCX[i_t][i_r][np.logical_and(P_HCX[i_t][i_r]>=P_HCX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),P_HCX[i_t][i_r]<P_HCX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
									temp_actual_values2.append(np.nanmax([n_HCX_intervals[i_t][i_r][i],np.mean(n_HCX[i_t][i_r][np.logical_and(n_HCX[i_t][i_r]>=n_HCX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),n_HCX[i_t][i_r]<n_HCX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
									temp_actual_values3.append(np.nanmax([n_H2CX_intervals[i_t][i_r][i],np.mean(n_H2CX[i_t][i_r][np.logical_and(n_H2CX[i_t][i_r]>=n_H2CX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),n_H2CX[i_t][i_r]<n_H2CX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
								else:
									temp_actual_values.append(np.nanmax([P_HCX_intervals[i_t][i_r][i],np.mean(P_HCX[i_t][i_r][np.logical_and(P_HCX[i_t][i_r]>=P_HCX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),P_HCX[i_t][i_r]<=P_HCX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
									temp_actual_values2.append(np.nanmax([n_HCX_intervals[i_t][i_r][i],np.mean(n_HCX[i_t][i_r][np.logical_and(n_HCX[i_t][i_r]>=n_HCX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),n_HCX[i_t][i_r]<=n_HCX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
									temp_actual_values3.append(np.nanmax([n_H2CX_intervals[i_t][i_r][i],np.mean(n_H2CX[i_t][i_r][np.logical_and(n_H2CX[i_t][i_r]>=n_H2CX_intervals[i_t][i_r][i]/(1+10*np.finfo(float).eps),n_H2CX[i_t][i_r]<=n_H2CX_intervals[i_t][i_r][i+1]*(1+10*np.finfo(float).eps))])]))
							P_HCX_actual_values[i_t][i_r] = np.array(temp_actual_values)
							n_HCX_actual_values[i_t][i_r] = np.array(temp_actual_values2)
							n_H2CX_actual_values[i_t][i_r] = np.array(temp_actual_values3)
				E_HCX = np.sum(2*np.pi*np.transpose(r_crop*np.transpose(P_HCX,(0,2,1)),(0,2,1))* dr,axis=(0,1))*dt/1000*length
				E_HCX_prob,E_HCX_intervals = np.histogram(E_HCX,bins=np.logspace(np.log10(E_HCX.min()),np.log10(E_HCX.max()),intervals+1))
				E_HCX_prob = E_HCX_prob/np.sum(E_HCX_prob)
				temp_actual_values = []
				for i in range(intervals):
					if i!=intervals-1:
						temp_actual_values.append(np.nanmax([E_HCX_intervals[i],np.mean(E_HCX[np.logical_and(E_HCX>=E_HCX_intervals[i]/(1+10*np.finfo(float).eps),E_HCX<E_HCX_intervals[i+1]*(1+10*np.finfo(float).eps))])]))
					else:
						temp_actual_values.append(np.nanmax([E_HCX_intervals[i],np.mean(E_HCX[np.logical_and(E_HCX>=E_HCX_intervals[i]/(1+10*np.finfo(float).eps),E_HCX<=E_HCX_intervals[i+1]*(1+10*np.finfo(float).eps))])]))
				E_HCX_actual_values = np.array(temp_actual_values)

				P_HCX_prob = P_HCX_prob.tolist()
				P_HCX_actual_values = P_HCX_actual_values.tolist()
				P_HCX_intervals = P_HCX_intervals.tolist()
				n_HCX_prob = n_HCX_prob.tolist()
				n_HCX_actual_values = n_HCX_actual_values.tolist()
				n_HCX_intervals = n_HCX_intervals.tolist()
				n_H2CX_prob = n_H2CX_prob.tolist()
				n_H2CX_actual_values = n_H2CX_actual_values.tolist()
				n_H2CX_intervals = n_H2CX_intervals.tolist()
				for i_t in range(np.shape(Te_all)[0]):
					for i_r in range(np.shape(Te_all)[1]):
						if len(actual_values_CX_term_1_1[i_t][i_r])<=1:
							P_HCX_prob[i_t][i_r] = [1]
							P_HCX_actual_values[i_t][i_r] = [0]
							P_HCX_intervals[i_t][i_r] = [0]
							n_HCX_prob[i_t][i_r] = [1]
							n_HCX_actual_values[i_t][i_r] = [0]
							n_HCX_intervals[i_t][i_r] = [0]
							n_H2CX_prob[i_t][i_r] = [1]
							n_H2CX_actual_values[i_t][i_r] = [0]
							n_H2CX_intervals[i_t][i_r] = [0]

				return P_HCX_intervals,P_HCX_prob,P_HCX_actual_values,E_HCX_intervals,E_HCX_prob,E_HCX_actual_values,n_HCX_intervals,n_HCX_prob,n_HCX_actual_values,n_H2CX_intervals,n_H2CX_prob,n_H2CX_actual_values

			intervals_P_HCX,prob_P_HCX,actual_values_P_HCX,intervals_E_HCX,prob_E_HCX,actual_values_E_HCX,intervals_n_HCX,prob_n_HCX,actual_values_n_HCX,intervals_n_H2CX,prob_n_H2CX,actual_values_n_H2CX = PDF_CX_MC(actual_values_CX_term_1_1,prob_CX_term_1_1,actual_values_CX_term_1_2,prob_CX_term_1_2,actual_values_CX_term_1_3,prob_CX_term_1_3,actual_values_CX_term_1_4,prob_CX_term_1_4,actual_values_CX_term_1_5,prob_CX_term_1_5,actual_values_CX_term_1_6,prob_CX_term_1_6,actual_values_H2_destruction_RR,prob_H2_destruction_RR)

			most_likely_n_H2CX = []
			for i_t in range(len(prob_n_H2CX)):
				temp=[]
				for i_r in range(len(prob_n_H2CX[i_t])):
					# temp.append((np.add(intervals_n_H2CX[i_t][i_r][1:],intervals_n_H2CX[i_t][i_r][:-1])/2)[np.array(prob_n_H2CX[i_t][i_r]).argmax()])
					temp.append(actual_values_n_H2CX[i_t][i_r][np.array(prob_n_H2CX[i_t][i_r]).argmax()])
				most_likely_n_H2CX.append(temp)
			plt.figure(figsize=(8, 5));
			plt.pcolor(temp_t, temp_r, most_likely_n_H2CX,cmap='rainbow',vmin=max(max(1,np.min(most_likely_n_H2CX)),np.max(most_likely_n_H2CX)*1e-6), norm=LogNorm());
			plt.colorbar(orientation="horizontal").set_label('density [#/m3]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]      ')
			plt.title(pre_title+'Molecular hydrogen density from inflow, Bayesian\ncold H2 (from simul), total H2 destruction rate')
			figure_index += 1
			plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index+1) + '.eps', bbox_inches='tight')
			plt.close()

			most_likely_P_HCX = []
			for i_t in range(len(prob_P_HCX)):
				temp=[]
				for i_r in range(len(prob_P_HCX[i_t])):
					# temp.append((np.add(intervals_P_HCX[i_t][i_r][1:],intervals_P_HCX[i_t][i_r][:-1])/2)[np.array(prob_P_HCX[i_t][i_r]).argmax()])
					temp.append(actual_values_P_HCX[i_t][i_r][np.array(prob_P_HCX[i_t][i_r]).argmax()])
				most_likely_P_HCX.append(temp)
			ML_E_HCX = actual_values_E_HCX[prob_E_HCX.argmax()]
			temp = np.cumsum(prob_E_HCX)
			ML_E_HCX_sigma = np.mean([ML_E_HCX-intervals_E_HCX[np.abs(temp-0.159).argmin()+1],intervals_E_HCX[np.abs(temp-1+0.159).argmin()]-ML_E_HCX])
			plt.figure(figsize=(8, 5));
			plt.pcolor(temp_t, temp_r, most_likely_P_HCX,cmap='rainbow',vmin=max(max(1,np.min(most_likely_P_HCX)),np.max(most_likely_P_HCX)*1e-6), norm=LogNorm());
			plt.colorbar(orientation="horizontal").set_label('power [W/m3]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]      ')
			plt.title(pre_title+'Power for atomic hydrogen CX from ADAS (inflow then outflow) Bayesian\ncold H (from simul), total H destruction rate (atomic, molecular, CX)\nlimit on enter=%.3g+/-%.3gJ' %(ML_E_HCX,ML_E_HCX_sigma))
			figure_index += 1
			plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index+1) + '.eps', bbox_inches='tight')
			plt.close()

			most_likely_n_HCX = []
			for i_t in range(len(prob_n_HCX)):
				temp=[]
				for i_r in range(len(prob_n_HCX[i_t])):
					# temp.append((np.add(intervals_n_HCX[i_t][i_r][1:],intervals_n_HCX[i_t][i_r][:-1])/2)[np.array(prob_n_HCX[i_t][i_r]).argmax()])
					temp.append(actual_values_n_HCX[i_t][i_r][np.array(prob_n_HCX[i_t][i_r]).argmax()])
				most_likely_n_HCX.append(temp)
			plt.figure(figsize=(8, 5));
			plt.pcolor(temp_t, temp_r, most_likely_n_HCX,cmap='rainbow',vmin=max(max(1,np.min(most_likely_n_HCX)),np.max(most_likely_n_HCX)*1e-6), norm=LogNorm());
			plt.colorbar(orientation="horizontal").set_label('density [#/m3]')  # ;plt.pause(0.01)
			plt.axes().set_aspect(20)
			plt.xlabel('time [ms]')
			plt.ylabel('radial location [m]      ')
			plt.title(pre_title+'Atomic hydrogen density from inflow, Bayesian\ncold H (from simul), total H destruction rate (atomic, molecular, CX)')
			figure_index += 1
			plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(figure_index+1) + '.eps', bbox_inches='tight')
			plt.close()


		area = 2*np.pi*(r_crop + np.median(np.diff(r_crop))/2) * np.median(np.diff(r_crop))
		length = 0.351+target_OES_distance/1000	# mm distance skimmer to OES/TS + OES/TS to target
		def radial_sum_PDF(intervals_power,prob_power,treshold_sum=1.1,treshold_prob=1e-50):
			out_values = []
			out_prob_sum = []
			for i_t in range(np.shape(Te_all)[0]):
				# plt.figure(figsize=(20, 10));
				for i_r in range(np.shape(Te_all)[1]-1):
					min_interv = []
					max_interv = []
					sum_prob_power = []
					if i_r==0:
						for i_1 in range(len(intervals_power[i_t][i_r])-1):
							for i_2 in range(len(intervals_power[i_t][i_r+1])-1):
								min_interv.append(intervals_power[i_t][i_r][i_1]*area[i_r]+intervals_power[i_t][i_r+1][i_2]*area[i_r+1])
								max_interv.append(intervals_power[i_t][i_r][i_1+1]*area[i_r]+intervals_power[i_t][i_r+1][i_2+1]*area[i_r+1])
								sum_prob_power.append(prob_power[i_t][i_r][i_1]*prob_power[i_t][i_r+1][i_2])
					else:
						for i_1 in range(len(values)-1):
							for i_2 in range(len(intervals_power[i_t][i_r+1])-1):
								min_interv.append(values[i_1]+intervals_power[i_t][i_r+1][i_2]*area[i_r+1])
								max_interv.append(values[i_1+1]+intervals_power[i_t][i_r+1][i_2+1]*area[i_r+1])
								sum_prob_power.append(prob_sum[i_1]*prob_power[i_t][i_r+1][i_2])
					min_interv = np.array(min_interv)
					max_interv = np.array(max_interv)
					d_interv = max_interv-min_interv
					sum_prob_power = np.array(sum_prob_power)
					values = np.unique(np.concatenate((min_interv,max_interv)))
					if len(values)==1:
						values = np.concatenate((min_interv,max_interv))
						prob_sum = sum_prob_power
					else:
						prob_sum = []
						for i_start in range(len(values)-1):
							select = np.logical_and(min_interv<=values[i_start],max_interv>=values[i_start+1])
							prob_sum.append(np.sum(sum_prob_power[select]*(values[i_start+1]-values[i_start])/(d_interv[select])))
					# plt.plot(np.sort(values.tolist()*2)[1:-1],100*np.array([prob_sum]*2).T.flatten(),'--')
					# if i_r==np.shape(Te_all)[1]-2:
					if len(values)>30:
						treshold = 1/1000
						temp = np.cumsum(prob_sum)
						temp1=[0]
						temp2 = []
						for i in range(1,1000):
							loc = np.abs(temp-treshold*i).argmin()
							temp1.append(values[loc+1])
							if i==1:
								temp2.append(temp[loc])
							else:
								temp2.append(temp[loc]-temp[prev_loc])
							prev_loc=loc
						temp1.append(values[-1])
						temp2.append(temp[-1]-np.sum(temp2))
						values = np.array(temp1)
						prob_sum = np.array(temp2)
						done=0
						limit = len(values)*3
						while (done==0 and limit>0):
							limit-=1
							if limit==1:
								print('falied PDF building in t=%.3g, r=%.3g' %(i_t,i_r))
							for i in range(1,len(values)-2):
								# print(values)
								if ((values[i+1]/values[i]<treshold_sum or prob_sum[i]<treshold_prob) and len(values)>10):
									values = np.concatenate((values[:i+1],values[i+2:]))
									prob_sum = np.concatenate((prob_sum[:i],[np.sum(prob_sum[i:i+2])],prob_sum[i+2:]))
									break
								if i==len(values)-3:
									done=1
					# plt.plot(np.sort(values.tolist()*2)[1:-1],100*np.array([prob_sum]*2).T.flatten())
				if len(values)==0:
					out_values.append([0,0])
					out_prob_sum.append([1])
				else:
					out_values.append(length*values)
					out_prob_sum.append(prob_sum)
				# plt.plot(np.sort(values.tolist()*2)[1:-1],100*np.array([prob_sum]*2).T.flatten(),'k--')
				#
				# plt.semilogx()
				# plt.xlim(left=1e-2)
				# plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
				# 	figure_index) + '.eps', bbox_inches='tight')
				# plt.close()
			return out_values,out_prob_sum

		def radial_sum_PDF_MC(actual_values_power,prob_power,intervals=30,samples=100000):
			out_values = []
			out_prob_sum = []
			out_actual_values = []
			for i_t in range(np.shape(Te_all)[0]):
				temp_values = np.zeros((samples))
				for i_r in range(np.shape(Te_all)[1]):
					if len(actual_values_power[i_t][i_r])>1:
						temp_values += np.random.choice(actual_values_power[i_t][i_r],size=samples,p=prob_power[i_t][i_r]) * area[i_r]
				if all(temp_values==0):
					out_values.append(np.array([0,0]))
					out_prob_sum.append(np.array([1]))
					out_actual_values.append(np.array([0]))
				else:
					temp_prob,temp_intervals = np.histogram(temp_values,bins=np.logspace(np.log10(temp_values.min()),np.log10(temp_values.max()),intervals+1))
					temp_actual_values = []
					for i in range(intervals):
						if i!=intervals-1:
							temp_actual_values.append(np.mean(temp_values[np.logical_and(temp_values>=temp_intervals[i]/(1+10*np.finfo(float).eps),temp_values<temp_intervals[i+1]*(1+10*np.finfo(float).eps))]))
						else:
							temp_actual_values.append(np.mean(temp_values[np.logical_and(temp_values>=temp_intervals[i]/(1+10*np.finfo(float).eps),temp_values<=temp_intervals[i+1]*(1+10*np.finfo(float).eps))]))
					temp_actual_values = length*np.array(temp_actual_values)
					temp_actual_values[np.isnan(temp_actual_values)]=0
					out_values.append(length*temp_intervals)
					out_prob_sum.append(temp_prob/np.sum(temp_prob))
					out_actual_values.append(temp_actual_values)
			return out_values,out_prob_sum,out_actual_values


		# intervals_power_rad_excit_r, prob_power_rad_excit_r = radial_sum_PDF(intervals_power_rad_excit,prob_power_rad_excit)
		# intervals_power_rad_rec_bremm_r, prob_power_rad_rec_bremm_r = radial_sum_PDF(intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm)
		# intervals_power_rad_mol_r, prob_power_rad_mol_r = radial_sum_PDF(intervals_power_rad_mol,prob_power_rad_mol)
		# intervals_power_via_ionisation_r, prob_power_via_ionisation_r = radial_sum_PDF(intervals_power_via_ionisation,prob_power_via_ionisation)
		# intervals_power_via_recombination_r, prob_power_via_recombination_r = radial_sum_PDF(intervals_power_via_recombination,prob_power_via_recombination)
		# intervals_tot_rad_power_r, prob_tot_rad_power_r = radial_sum_PDF(intervals_tot_rad_power,prob_tot_rad_power)
		# intervals_power_rad_Hm_r, prob_power_rad_Hm_r = radial_sum_PDF(intervals_power_rad_Hm,prob_power_rad_Hm)
		# intervals_power_rad_H2_r, prob_power_rad_H2_r = radial_sum_PDF(intervals_power_rad_H2,prob_power_rad_H2)
		# intervals_power_rad_H2p_r, prob_power_rad_H2p_r = radial_sum_PDF(intervals_power_rad_H2p,prob_power_rad_H2p)
		# intervals_power_heating_rec_r, prob_power_heating_rec_r = radial_sum_PDF(intervals_power_heating_rec,prob_power_heating_rec)
		# intervals_power_rec_neutral_r, prob_power_rec_neutral_r = radial_sum_PDF(intervals_power_rec_neutral,prob_power_rec_neutral)
		# intervals_power_via_brem_r, prob_power_via_brem_r = radial_sum_PDF(intervals_power_via_brem,prob_power_via_brem)
		# intervals_total_removed_power_r, prob_total_removed_power_r = radial_sum_PDF(intervals_total_removed_power,prob_total_removed_power)
		# intervals_local_CX_r, prob_local_CX_r = radial_sum_PDF(intervals_local_CX,prob_local_CX)
		intervals_power_rad_excit_r, prob_power_rad_excit_r, actual_values_power_rad_excit_r = radial_sum_PDF_MC(actual_values_power_rad_excit,prob_power_rad_excit)
		intervals_power_rad_rec_bremm_r, prob_power_rad_rec_bremm_r, actual_values_power_rad_rec_bremm_r = radial_sum_PDF_MC(actual_values_power_rad_rec_bremm,prob_power_rad_rec_bremm)
		intervals_power_rad_mol_r, prob_power_rad_mol_r, actual_values_power_rad_mol_r = radial_sum_PDF_MC(actual_values_power_rad_mol,prob_power_rad_mol)
		intervals_power_via_ionisation_r, prob_power_via_ionisation_r, actual_values_power_via_ionisation_r = radial_sum_PDF_MC(actual_values_power_via_ionisation,prob_power_via_ionisation)
		intervals_power_via_recombination_r, prob_power_via_recombination_r, actual_values_power_via_recombination_r = radial_sum_PDF_MC(actual_values_power_via_recombination,prob_power_via_recombination)
		intervals_tot_rad_power_r, prob_tot_rad_power_r, actual_values_tot_rad_power_r = radial_sum_PDF_MC(actual_values_tot_rad_power,prob_tot_rad_power)
		intervals_power_rad_Hm_r, prob_power_rad_Hm_r, actual_values_power_rad_Hm_r = radial_sum_PDF_MC(actual_values_power_rad_Hm,prob_power_rad_Hm)
		intervals_power_rad_H2_r, prob_power_rad_H2_r, actual_values_power_rad_H2_r = radial_sum_PDF_MC(actual_values_power_rad_H2,prob_power_rad_H2)
		intervals_power_rad_H2p_r, prob_power_rad_H2p_r, actual_values_power_rad_H2p_r = radial_sum_PDF_MC(actual_values_power_rad_H2p,prob_power_rad_H2p)
		intervals_power_heating_rec_r, prob_power_heating_rec_r, actual_values_power_heating_rec_r = radial_sum_PDF_MC(actual_values_power_heating_rec,prob_power_heating_rec)
		intervals_power_rec_neutral_r, prob_power_rec_neutral_r, actual_values_power_rec_neutral_r = radial_sum_PDF_MC(actual_values_power_rec_neutral,prob_power_rec_neutral)
		intervals_power_via_brem_r, prob_power_via_brem_r, actual_values_power_via_brem_r = radial_sum_PDF_MC(actual_values_power_via_brem,prob_power_via_brem)
		intervals_total_removed_power_r, prob_total_removed_power_r, actual_values_total_removed_power_r = radial_sum_PDF_MC(actual_values_total_removed_power,prob_total_removed_power)
		intervals_local_CX_r, prob_local_CX_r, actual_values_local_CX_r = radial_sum_PDF_MC(actual_values_local_CX,prob_local_CX)


		most_likely_power_rad_excit_r = []
		actual_values_power_rad_excit_r_up=[]
		actual_values_power_rad_excit_r_down=[]
		for i_t in range(len(prob_power_rad_excit_r)):
			most_likely_power_rad_excit_r.append(actual_values_power_rad_excit_r[i_t][np.array(prob_power_rad_excit_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_excit_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_excit_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_excit_r_down.append(intervals_power_rad_excit_r[i_t][loc_down+1])
			actual_values_power_rad_excit_r_up.append(intervals_power_rad_excit_r[i_t][-(loc_up+1)])
		most_likely_power_rad_excit_r = np.array(most_likely_power_rad_excit_r)
		actual_values_power_rad_excit_r_up = np.array(actual_values_power_rad_excit_r_up)
		actual_values_power_rad_excit_r_down = np.array(actual_values_power_rad_excit_r_down)
		most_likely_power_rad_rec_bremm_r = []
		actual_values_power_rad_rec_bremm_r_up=[]
		actual_values_power_rad_rec_bremm_r_down=[]
		for i_t in range(len(prob_power_rad_rec_bremm_r)):
			most_likely_power_rad_rec_bremm_r.append(actual_values_power_rad_rec_bremm_r[i_t][np.array(prob_power_rad_rec_bremm_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_rec_bremm_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_rec_bremm_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_rec_bremm_r_down.append(intervals_power_rad_rec_bremm_r[i_t][loc_down+1])
			actual_values_power_rad_rec_bremm_r_up.append(intervals_power_rad_rec_bremm_r[i_t][-(loc_up+1)])
		most_likely_power_rad_rec_bremm_r = np.array(most_likely_power_rad_rec_bremm_r)
		actual_values_power_rad_rec_bremm_r_up = np.array(actual_values_power_rad_rec_bremm_r_up)
		actual_values_power_rad_rec_bremm_r_down = np.array(actual_values_power_rad_rec_bremm_r_down)
		most_likely_power_rad_mol_r = []
		actual_values_power_rad_mol_r_up=[]
		actual_values_power_rad_mol_r_down=[]
		for i_t in range(len(prob_power_rad_mol_r)):
			most_likely_power_rad_mol_r.append(actual_values_power_rad_mol_r[i_t][np.array(prob_power_rad_mol_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_mol_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_mol_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_mol_r_down.append(intervals_power_rad_mol_r[i_t][loc_down+1])
			actual_values_power_rad_mol_r_up.append(intervals_power_rad_mol_r[i_t][-(loc_up+1)])
		most_likely_power_rad_mol_r = np.array(most_likely_power_rad_mol_r)
		actual_values_power_rad_mol_r_up = np.array(actual_values_power_rad_mol_r_up)
		actual_values_power_rad_mol_r_down = np.array(actual_values_power_rad_mol_r_down)
		most_likely_power_via_ionisation_r = []
		actual_values_power_via_ionisation_r_up=[]
		actual_values_power_via_ionisation_r_down=[]
		for i_t in range(len(prob_power_via_ionisation_r)):
			most_likely_power_via_ionisation_r.append(actual_values_power_via_ionisation_r[i_t][np.array(prob_power_via_ionisation_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_via_ionisation_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_via_ionisation_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_via_ionisation_r_down.append(intervals_power_via_ionisation_r[i_t][loc_down+1])
			actual_values_power_via_ionisation_r_up.append(intervals_power_via_ionisation_r[i_t][-(loc_up+1)])
		most_likely_power_via_ionisation_r = np.array(most_likely_power_via_ionisation_r)
		actual_values_power_via_ionisation_r_up = np.array(actual_values_power_via_ionisation_r_up)
		actual_values_power_via_ionisation_r_down = np.array(actual_values_power_via_ionisation_r_down)
		most_likely_power_via_recombination_r = []
		actual_values_power_via_recombination_r_up=[]
		actual_values_power_via_recombination_r_down=[]
		for i_t in range(len(prob_power_via_recombination_r)):
			most_likely_power_via_recombination_r.append(actual_values_power_via_recombination_r[i_t][np.array(prob_power_via_recombination_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_via_recombination_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_via_recombination_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_via_recombination_r_down.append(intervals_power_via_recombination_r[i_t][loc_down+1])
			actual_values_power_via_recombination_r_up.append(intervals_power_via_recombination_r[i_t][-(loc_up+1)])
		most_likely_power_via_recombination_r = np.array(most_likely_power_via_recombination_r)
		actual_values_power_via_recombination_r_up = np.array(actual_values_power_via_recombination_r_up)
		actual_values_power_via_recombination_r_down = np.array(actual_values_power_via_recombination_r_down)
		most_likely_tot_rad_power_r = []
		actual_values_tot_rad_power_r_up=[]
		actual_values_tot_rad_power_r_down=[]
		for i_t in range(len(prob_tot_rad_power_r)):
			most_likely_tot_rad_power_r.append(actual_values_tot_rad_power_r[i_t][np.array(prob_tot_rad_power_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_tot_rad_power_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_tot_rad_power_r[i_t],axis=0))-0.159).argmin()
			actual_values_tot_rad_power_r_down.append(intervals_tot_rad_power_r[i_t][loc_down+1])
			actual_values_tot_rad_power_r_up.append(intervals_tot_rad_power_r[i_t][-(loc_up+1)])
		most_likely_tot_rad_power_r = np.array(most_likely_tot_rad_power_r)
		actual_values_tot_rad_power_r_up = np.array(actual_values_tot_rad_power_r_up)
		actual_values_tot_rad_power_r_down = np.array(actual_values_tot_rad_power_r_down)
		most_likely_power_rad_Hm_r = []
		actual_values_power_rad_Hm_r_up=[]
		actual_values_power_rad_Hm_r_down=[]
		for i_t in range(len(prob_power_rad_Hm_r)):
			most_likely_power_rad_Hm_r.append(actual_values_power_rad_Hm_r[i_t][np.array(prob_power_rad_Hm_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_Hm_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_Hm_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_Hm_r_down.append(intervals_power_rad_Hm_r[i_t][loc_down+1])
			actual_values_power_rad_Hm_r_up.append(intervals_power_rad_Hm_r[i_t][-(loc_up+1)])
		most_likely_power_rad_Hm_r = np.array(most_likely_power_rad_Hm_r)
		actual_values_power_rad_Hm_r_up = np.array(actual_values_power_rad_Hm_r_up)
		actual_values_power_rad_Hm_r_down = np.array(actual_values_power_rad_Hm_r_down)
		most_likely_power_rad_H2_r = []
		actual_values_power_rad_H2_r_up=[]
		actual_values_power_rad_H2_r_down=[]
		for i_t in range(len(prob_power_rad_H2_r)):
			most_likely_power_rad_H2_r.append(actual_values_power_rad_H2_r[i_t][np.array(prob_power_rad_H2_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_H2_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_H2_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_H2_r_down.append(intervals_power_rad_H2_r[i_t][loc_down+1])
			actual_values_power_rad_H2_r_up.append(intervals_power_rad_H2_r[i_t][-(loc_up+1)])
		most_likely_power_rad_H2_r = np.array(most_likely_power_rad_H2_r)
		actual_values_power_rad_H2_r_up = np.array(actual_values_power_rad_H2_r_up)
		actual_values_power_rad_H2_r_down = np.array(actual_values_power_rad_H2_r_down)
		most_likely_power_rad_H2p_r = []
		actual_values_power_rad_H2p_r_up=[]
		actual_values_power_rad_H2p_r_down=[]
		for i_t in range(len(prob_power_rad_H2p_r)):
			most_likely_power_rad_H2p_r.append(actual_values_power_rad_H2p_r[i_t][np.array(prob_power_rad_H2p_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rad_H2p_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rad_H2p_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rad_H2p_r_down.append(intervals_power_rad_H2p_r[i_t][loc_down+1])
			actual_values_power_rad_H2p_r_up.append(intervals_power_rad_H2p_r[i_t][-(loc_up+1)])
		most_likely_power_rad_H2p_r = np.array(most_likely_power_rad_H2p_r)
		actual_values_power_rad_H2p_r_up = np.array(actual_values_power_rad_H2p_r_up)
		actual_values_power_rad_H2p_r_down = np.array(actual_values_power_rad_H2p_r_down)
		most_likely_power_heating_rec_r = []
		actual_values_power_heating_rec_r_up=[]
		actual_values_power_heating_rec_r_down=[]
		for i_t in range(len(prob_power_heating_rec_r)):
			most_likely_power_heating_rec_r.append(actual_values_power_heating_rec_r[i_t][np.array(prob_power_heating_rec_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_heating_rec_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_heating_rec_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_heating_rec_r_down.append(intervals_power_heating_rec_r[i_t][loc_down+1])
			actual_values_power_heating_rec_r_up.append(intervals_power_heating_rec_r[i_t][-(loc_up+1)])
		most_likely_power_heating_rec_r = np.array(most_likely_power_heating_rec_r)
		actual_values_power_heating_rec_r_up = np.array(actual_values_power_heating_rec_r_up)
		actual_values_power_heating_rec_r_down = np.array(actual_values_power_heating_rec_r_down)
		most_likely_power_rec_neutral_r = []
		actual_values_power_rec_neutral_r_up=[]
		actual_values_power_rec_neutral_r_down=[]
		for i_t in range(len(prob_power_rec_neutral_r)):
			most_likely_power_rec_neutral_r.append(actual_values_power_rec_neutral_r[i_t][np.array(prob_power_rec_neutral_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_rec_neutral_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_rec_neutral_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_rec_neutral_r_down.append(intervals_power_rec_neutral_r[i_t][loc_down+1])
			actual_values_power_rec_neutral_r_up.append(intervals_power_rec_neutral_r[i_t][-(loc_up+1)])
		most_likely_power_rec_neutral_r = np.array(most_likely_power_rec_neutral_r)
		actual_values_power_rec_neutral_r_up = np.array(actual_values_power_rec_neutral_r_up)
		actual_values_power_rec_neutral_r_down = np.array(actual_values_power_rec_neutral_r_down)
		most_likely_power_via_brem_r = []
		actual_values_power_via_brem_r_up=[]
		actual_values_power_via_brem_r_down=[]
		for i_t in range(len(prob_power_via_brem_r)):
			most_likely_power_via_brem_r.append(actual_values_power_via_brem_r[i_t][np.array(prob_power_via_brem_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_power_via_brem_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_power_via_brem_r[i_t],axis=0))-0.159).argmin()
			actual_values_power_via_brem_r_down.append(intervals_power_via_brem_r[i_t][loc_down+1])
			actual_values_power_via_brem_r_up.append(intervals_power_via_brem_r[i_t][-(loc_up+1)])
		most_likely_power_via_brem_r = np.array(most_likely_power_via_brem_r)
		actual_values_power_via_brem_r_up = np.array(actual_values_power_via_brem_r_up)
		actual_values_power_via_brem_r_down = np.array(actual_values_power_via_brem_r_down)
		most_likely_total_removed_power_r = []
		actual_values_total_removed_power_r_up=[]
		actual_values_total_removed_power_r_down=[]
		for i_t in range(len(prob_total_removed_power_r)):
			most_likely_total_removed_power_r.append(actual_values_total_removed_power_r[i_t][np.array(prob_total_removed_power_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_total_removed_power_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_total_removed_power_r[i_t],axis=0))-0.159).argmin()
			actual_values_total_removed_power_r_down.append(intervals_total_removed_power_r[i_t][loc_down+1])
			actual_values_total_removed_power_r_up.append(intervals_total_removed_power_r[i_t][-(loc_up+1)])
		most_likely_total_removed_power_r = np.array(most_likely_total_removed_power_r)
		actual_values_total_removed_power_r_up = np.array(actual_values_total_removed_power_r_up)
		actual_values_total_removed_power_r_down = np.array(actual_values_total_removed_power_r_down)
		most_likely_local_CX_r = []
		actual_values_local_CX_r_up=[]
		actual_values_local_CX_r_down=[]
		for i_t in range(len(prob_local_CX_r)):
			most_likely_local_CX_r.append(actual_values_local_CX_r[i_t][np.array(prob_local_CX_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_local_CX_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_local_CX_r[i_t],axis=0))-0.159).argmin()
			actual_values_local_CX_r_down.append(intervals_local_CX_r[i_t][loc_down+1])
			actual_values_local_CX_r_up.append(intervals_local_CX_r[i_t][-(loc_up+1)])
		most_likely_local_CX_r = np.array(most_likely_local_CX_r)
		actual_values_local_CX_r_up = np.array(actual_values_local_CX_r_up)
		actual_values_local_CX_r_down = np.array(actual_values_local_CX_r_down)

		# def total_lost_power_radially_averaged_PDF(treshold_sum=1.1,treshold_prob=1e-50):
		# 	out_values = []
		# 	out_prob_sum = []
		# 	for i_t in range(len(intervals_power_via_ionisation_r)):
		# 		min_interv = []
		# 		max_interv = []
		# 		sum_prob_power = []
		# 		for i_1 in range(len(intervals_power_via_ionisation_r[i_t])-1):
		# 			# for i_2 in range(len(intervals_power_rad_rec_bremm_r[i_t])-1):
		# 			for i_3 in range(len(intervals_tot_rad_power_r[i_t])-1):
		# 				for i_4 in range(len(intervals_power_heating_rec_r[i_t])-1):
		# 					if intervals_power_heating_rec_r[i_t][i_4]>0:
		# 						min_interv.append(intervals_power_via_ionisation_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3]-intervals_power_heating_rec_r[i_t][i_4+1])
		# 						max_interv.append(intervals_power_via_ionisation_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1]-intervals_power_heating_rec_r[i_t][i_4])
		# 					else:
		# 						min_interv.append(intervals_power_via_ionisation_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3]-intervals_power_heating_rec_r[i_t][i_4])
		# 						max_interv.append(intervals_power_via_ionisation_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1]-intervals_power_heating_rec_r[i_t][i_4+1])
		# 					sum_prob_power.append(prob_power_via_ionisation_r[i_t][i_1]*prob_tot_rad_power_r[i_t][i_3]*prob_power_heating_rec_r[i_t][i_4])
		# 		# for i_1 in range(len(intervals_power_via_ionisation_r[i_t])-1):
		# 		# 	for i_3 in range(len(intervals_tot_rad_power_r[i_t])-1):
		# 		# 		min_interv.append(intervals_power_via_ionisation_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3])
		# 		# 		max_interv.append(intervals_power_via_ionisation_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1])
		# 		# 		sum_prob_power.append(prob_power_via_ionisation_r[i_t][i_1]*prob_tot_rad_power_r[i_t][i_3])
		# 		min_interv = np.array(min_interv)
		# 		max_interv = np.array(max_interv)
		# 		d_interv = max_interv-min_interv
		# 		sum_prob_power = np.array(sum_prob_power)
		# 		values = np.unique(np.concatenate((min_interv,max_interv)))
		# 		if len(values)==1:
		# 			values = np.concatenate((min_interv,max_interv))
		# 			prob_sum = sum_prob_power
		# 		else:
		# 			prob_sum = []
		# 			for i_start in range(len(values)-1):
		# 				select = np.logical_and(min_interv<=values[i_start],max_interv>=values[i_start+1])
		# 				prob_sum.append(np.sum(sum_prob_power[select]*(values[i_start+1]-values[i_start])/(d_interv[select])))
		# 		if len(values)>30:
		# 			treshold = 1/1000
		# 			temp = np.cumsum(prob_sum)
		# 			temp1=[0]
		# 			temp2 = []
		# 			for i in range(1,1000):
		# 				loc = np.abs(temp-treshold*i).argmin()
		# 				temp1.append(values[loc+1])
		# 				if i==1:
		# 					temp2.append(temp[loc])
		# 				else:
		# 					temp2.append(temp[loc]-temp[prev_loc])
		# 				prev_loc=loc
		# 			temp1.append(values[-1])
		# 			temp2.append(temp[-1]-np.sum(temp2))
		# 			values = np.array(temp1)
		# 			prob_sum = np.array(temp2)
		# 			done=0
		# 			limit = len(values)*3
		# 			while (done==0 and limit>0):
		# 				limit-=1
		# 				if limit==1:
		# 					print('falied PDF building in t=%.3g, r=%.3g' %(i_t,i_r))
		# 				for i in range(1,len(values)-2):
		# 					# print(values)
		# 					if ((values[i+1]/values[i]<treshold_sum or prob_sum[i]<treshold_prob) and len(values)>10):
		# 						values = np.concatenate((values[:i+1],values[i+2:]))
		# 						prob_sum = np.concatenate((prob_sum[:i],[np.sum(prob_sum[i:i+2])],prob_sum[i+2:]))
		# 						break
		# 					if i==len(values)-3:
		# 						done=1
		# 		if len(values)==0:
		# 			out_values.append([0,0])
		# 			out_prob_sum.append([1])
		# 		else:
		# 			out_values.append(values)
		# 			out_prob_sum.append(prob_sum)
		# 	return out_values,out_prob_sum
		# intervals_total_power_removed_plasma_fluid_r,prob_total_power_removed_plasma_fluid_r = total_lost_power_radially_averaged_PDF()
		# actual_values_total_power_removed_plasma_fluid_r = []
		# actual_values_total_power_removed_plasma_fluid_r_up=[]
		# actual_values_total_power_removed_plasma_fluid_r_down=[]
		# for i_t in range(len(prob_total_power_removed_plasma_fluid_r)):
		# 	actual_values_total_power_removed_plasma_fluid_r.append(((intervals_total_power_removed_plasma_fluid_r[i_t][1:]+intervals_total_power_removed_plasma_fluid_r[i_t][:-1])/2)[np.array(prob_total_power_removed_plasma_fluid_r[i_t]).argmax()])
		# 	loc_down = np.abs(np.cumsum(prob_total_power_removed_plasma_fluid_r[i_t])-0.159).argmin()
		# 	loc_up = np.abs(np.cumsum(np.flip(prob_total_power_removed_plasma_fluid_r[i_t],axis=0))-0.159).argmin()
		# 	actual_values_total_power_removed_plasma_fluid_r_down.append(intervals_total_power_removed_plasma_fluid_r[i_t][loc_down+1])
		# 	actual_values_total_power_removed_plasma_fluid_r_up.append(intervals_total_power_removed_plasma_fluid_r[i_t][-(loc_up+1)])
		# actual_values_total_power_removed_plasma_fluid_r = np.array(actual_values_total_power_removed_plasma_fluid_r)
		# actual_values_total_power_removed_plasma_fluid_r_up = np.array(actual_values_total_power_removed_plasma_fluid_r_up)
		# actual_values_total_power_removed_plasma_fluid_r_down = np.array(actual_values_total_power_removed_plasma_fluid_r_down)

		def net_lost_power_radially_averaged_PDF(treshold_sum=1.1,treshold_prob=1e-50):
			out_values = []
			out_prob_sum = []
			for i_t in range(len(intervals_power_via_ionisation_r)):
				min_interv = []
				max_interv = []
				sum_prob_power = []
				for i_1 in range(len(intervals_power_rec_neutral_r[i_t])-1):
					# for i_2 in range(len(intervals_power_rad_rec_bremm_r[i_t])-1):
					for i_3 in range(len(intervals_tot_rad_power_r[i_t])-1):
						# for i_4 in range(len(intervals_power_heating_rec_r[i_t])-1):
							# if intervals_power_heating_rec_r[i_t][i_4]>0:
						min_interv.append(max(0,intervals_power_rec_neutral_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3]))
						max_interv.append(max(0,intervals_power_rec_neutral_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1]))
							# else:
							# 	min_interv.append(max(0,intervals_power_rec_neutral_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3]-intervals_power_heating_rec_r[i_t][i_4]))
							# 	max_interv.append(max(0,intervals_power_rec_neutral_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1]-intervals_power_heating_rec_r[i_t][i_4+1]))
						sum_prob_power.append(prob_power_rec_neutral_r[i_t][i_1]*prob_tot_rad_power_r[i_t][i_3])
				# for i_1 in range(len(intervals_power_via_ionisation_r[i_t])-1):
				# 	for i_3 in range(len(intervals_tot_rad_power_r[i_t])-1):
				# 		min_interv.append(intervals_power_via_ionisation_r[i_t][i_1]+intervals_tot_rad_power_r[i_t][i_3])
				# 		max_interv.append(intervals_power_via_ionisation_r[i_t][i_1+1]+intervals_tot_rad_power_r[i_t][i_3+1])
				# 		sum_prob_power.append(prob_power_via_ionisation_r[i_t][i_1]*prob_tot_rad_power_r[i_t][i_3])
				min_interv = np.array(min_interv)
				max_interv = np.array(max_interv)
				d_interv = max_interv-min_interv
				sum_prob_power = np.array(sum_prob_power)
				values = np.unique(np.concatenate((min_interv,max_interv)))
				if len(values)==1:
					values = np.concatenate((min_interv,max_interv))
					prob_sum = sum_prob_power
				else:
					prob_sum = []
					for i_start in range(len(values)-1):
						select = np.logical_and(min_interv<=values[i_start],max_interv>=values[i_start+1])
						prob_sum.append(np.sum(sum_prob_power[select]*(values[i_start+1]-values[i_start])/(d_interv[select])))
				if len(values)>30:
					treshold = 1/1000
					temp = np.cumsum(prob_sum)
					temp1=[0]
					temp2 = []
					for i in range(1,1000):
						loc = np.abs(temp-treshold*i).argmin()
						temp1.append(values[loc+1])
						if i==1:
							temp2.append(temp[loc])
						else:
							temp2.append(temp[loc]-temp[prev_loc])
						prev_loc=loc
					temp1.append(values[-1])
					temp2.append(temp[-1]-np.sum(temp2))
					values = np.array(temp1)
					prob_sum = np.array(temp2)
					done=0
					limit = len(values)*3
					while (done==0 and limit>0):
						limit-=1
						if limit==1:
							print('falied PDF building in t=%.3g, r=%.3g' %(i_t,i_r))
						for i in range(1,len(values)-2):
							# print(values)
							if ((values[i+1]/values[i]<treshold_sum or prob_sum[i]<treshold_prob) and len(values)>10):
								values = np.concatenate((values[:i+1],values[i+2:]))
								prob_sum = np.concatenate((prob_sum[:i],[np.sum(prob_sum[i:i+2])],prob_sum[i+2:]))
								break
							if i==len(values)-3:
								done=1
				if len(values)==0:
					out_values.append([0,0])
					out_prob_sum.append([1])
				else:
					out_values.append(values)
					out_prob_sum.append(prob_sum)
			return out_values,out_prob_sum
		# intervals_net_power_removed_plasma_column_r,prob_net_power_removed_plasma_column_r = net_lost_power_radially_averaged_PDF()

		def radial_sum_net_lost_power_PDF_MC(intervals=30,samples=100000):
			out_values = []
			out_prob_sum = []
			out_actual_values = []
			for i_t in range(np.shape(Te_all)[0]):
				temp_values = np.zeros((samples))
				for i_r in range(np.shape(Te_all)[1]):
					if ((len(actual_values_power_rec_neutral[i_t][i_r])>1) and (len(actual_values_tot_rad_power[i_t][i_r])>1)):
						temp_values += (np.random.choice(actual_values_power_rec_neutral[i_t][i_r],size=samples,p=prob_power_rec_neutral[i_t][i_r]) + np.random.choice(actual_values_tot_rad_power[i_t][i_r],size=samples,p=prob_tot_rad_power[i_t][i_r]))* area[i_r]
				if all(temp_values==0):
					out_values.append(np.array([0,0]))
					out_prob_sum.append(np.array([1]))
					out_actual_values.append(np.array([0]))
				else:
					temp_prob,temp_intervals = np.histogram(temp_values,bins=np.logspace(np.log10(temp_values.min()),np.log10(temp_values.max()),intervals+1))
					temp_actual_values = []
					for i in range(intervals):
						if i!=intervals-1:
							temp_actual_values.append(np.mean(temp_values[np.logical_and(temp_values>=temp_intervals[i]/(1+10*np.finfo(float).eps),temp_values<temp_intervals[i+1]*(1+10*np.finfo(float).eps))]))
						else:
							temp_actual_values.append(np.mean(temp_values[np.logical_and(temp_values>=temp_intervals[i]/(1+10*np.finfo(float).eps),temp_values<=temp_intervals[i+1]*(1+10*np.finfo(float).eps))]))
					temp_actual_values = length*np.array(temp_actual_values)
					temp_actual_values[np.isnan(temp_actual_values)]=0
					out_values.append(length*temp_intervals)
					out_prob_sum.append(temp_prob/np.sum(temp_prob))
					out_actual_values.append(temp_actual_values)
			return out_values,out_prob_sum,out_actual_values

		intervals_net_power_removed_plasma_column_r,prob_net_power_removed_plasma_column_r,actual_values_net_power_removed_plasma_column_r = radial_sum_net_lost_power_PDF_MC()

		most_likely_net_power_removed_plasma_column_r = []
		actual_values_net_power_removed_plasma_column_r_up=[]
		actual_values_net_power_removed_plasma_column_r_down=[]
		for i_t in range(len(prob_net_power_removed_plasma_column_r)):
			most_likely_net_power_removed_plasma_column_r.append(actual_values_net_power_removed_plasma_column_r[i_t][np.array(prob_net_power_removed_plasma_column_r[i_t]).argmax()])
			loc_down = np.abs(np.cumsum(prob_net_power_removed_plasma_column_r[i_t])-0.159).argmin()
			loc_up = np.abs(np.cumsum(np.flip(prob_net_power_removed_plasma_column_r[i_t],axis=0))-0.159).argmin()
			actual_values_net_power_removed_plasma_column_r_down.append(intervals_net_power_removed_plasma_column_r[i_t][loc_down+1])
			actual_values_net_power_removed_plasma_column_r_up.append(intervals_net_power_removed_plasma_column_r[i_t][-(loc_up+1)])
		most_likely_net_power_removed_plasma_column_r = np.array(most_likely_net_power_removed_plasma_column_r)
		actual_values_net_power_removed_plasma_column_r_up = np.array(actual_values_net_power_removed_plasma_column_r_up)
		actual_values_net_power_removed_plasma_column_r_down = np.array(actual_values_net_power_removed_plasma_column_r_down)


		# power_pulse_shape_peak = power_pulse_shape.argmax()
		# # ionisation_peak = time_crop[(actual_values_net_power_removed_plasma_column_r).argmax()]
		# ionisation_peak = time_crop[(actual_values_power_via_ionisation_r).argmax()]
		# time_source_power = np.arange(len(power_pulse_shape))*time_resolution*1000	# I work here in ms, while current trace is in s
		# time_source_power = time_source_power-time_source_power[power_pulse_shape_peak]+ionisation_peak
		# power_pulse_shape_crop = power_pulse_shape[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
		# power_pulse_shape_std_crop = power_pulse_shape_std[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
		# time_source_power_crop = time_source_power[np.logical_and(time_source_power>=np.min(time_crop),time_source_power<=np.max(time_crop))]
		power_pulse_shape_crop = interpolated_power_pulse_shape(time_crop)
		power_pulse_shape_std_crop = interpolated_power_pulse_shape_std(time_crop)
		time_source_power_crop = cp.deepcopy(time_crop)


		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_excit_r)):
			if len(prob_power_rad_excit_r[i_t])>1:
				for i_r in range(len(prob_power_rad_excit_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_excit_r[i_t][i_r]),dt,intervals_power_rad_excit_r[i_t][i_r+1]-intervals_power_rad_excit_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_excit_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_excit_r,yerr=[most_likely_power_rad_excit_r-actual_values_power_rad_excit_r_down,actual_values_power_rad_excit_r_up-most_likely_power_rad_excit_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'IRadial sum of power_rad_excit, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_rec_bremm_r)):
			if len(prob_power_rad_rec_bremm_r[i_t])>1:
				for i_r in range(len(prob_power_rad_rec_bremm_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_rec_bremm_r[i_t][i_r]),dt,intervals_power_rad_rec_bremm_r[i_t][i_r+1]-intervals_power_rad_rec_bremm_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_rec_bremm_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_rec_bremm_r,yerr=[most_likely_power_rad_rec_bremm_r-actual_values_power_rad_rec_bremm_r_down,actual_values_power_rad_rec_bremm_r_up-most_likely_power_rad_rec_bremm_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rad_rec_bremm, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_mol_r)):
			if len(prob_power_rad_mol_r[i_t])>1:
				for i_r in range(len(prob_power_rad_mol_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_mol_r[i_t][i_r]),dt,intervals_power_rad_mol_r[i_t][i_r+1]-intervals_power_rad_mol_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_mol_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_mol_r,yerr=[most_likely_power_rad_mol_r-actual_values_power_rad_mol_r_down,actual_values_power_rad_mol_r_up-most_likely_power_rad_mol_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rad_mol, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_via_ionisation_r)):
			if len(prob_power_via_ionisation_r[i_t])>1:
				for i_r in range(len(prob_power_via_ionisation_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_via_ionisation_r[i_t][i_r]),dt,intervals_power_via_ionisation_r[i_t][i_r+1]-intervals_power_via_ionisation_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_via_ionisation_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_via_ionisation_r,yerr=[most_likely_power_via_ionisation_r-actual_values_power_via_ionisation_r_down,actual_values_power_via_ionisation_r_up-most_likely_power_via_ionisation_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_via_ionisation, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_via_recombination_r)):
			if len(prob_power_via_recombination_r[i_t])>1:
				for i_r in range(len(prob_power_via_recombination_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_via_recombination_r[i_t][i_r]),dt,intervals_power_via_recombination_r[i_t][i_r+1]-intervals_power_via_recombination_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_via_recombination_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_via_recombination_r,yerr=[most_likely_power_via_recombination_r-actual_values_power_via_recombination_r_down,actual_values_power_via_recombination_r_up-most_likely_power_via_recombination_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_via_recombination, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_tot_rad_power_r)):
			if len(prob_tot_rad_power_r[i_t])>1:
				for i_r in range(len(prob_tot_rad_power_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_tot_rad_power_r[i_t][i_r]),dt,intervals_tot_rad_power_r[i_t][i_r+1]-intervals_tot_rad_power_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_tot_rad_power_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_tot_rad_power_r,yerr=[most_likely_tot_rad_power_r-actual_values_tot_rad_power_r_down,actual_values_tot_rad_power_r_up-most_likely_tot_rad_power_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of tot_rad_power, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_Hm_r)):
			if len(prob_power_rad_Hm_r[i_t])>1:
				for i_r in range(len(prob_power_rad_Hm_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_Hm_r[i_t][i_r]),dt,intervals_power_rad_Hm_r[i_t][i_r+1]-intervals_power_rad_Hm_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_Hm_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_Hm_r,yerr=[most_likely_power_rad_Hm_r-actual_values_power_rad_Hm_r_down,actual_values_power_rad_Hm_r_up-most_likely_power_rad_Hm_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rad_Hm, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_H2_r)):
			if len(prob_power_rad_H2_r[i_t])>1:
				for i_r in range(len(prob_power_rad_H2_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_H2_r[i_t][i_r]),dt,intervals_power_rad_H2_r[i_t][i_r+1]-intervals_power_rad_H2_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_H2_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_H2_r,yerr=[most_likely_power_rad_H2_r-actual_values_power_rad_H2_r_down,actual_values_power_rad_H2_r_up-most_likely_power_rad_H2_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rad_H2, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rad_H2p_r)):
			if len(prob_power_rad_H2p_r[i_t])>1:
				for i_r in range(len(prob_power_rad_H2p_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rad_H2p_r[i_t][i_r]),dt,intervals_power_rad_H2p_r[i_t][i_r+1]-intervals_power_rad_H2p_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rad_H2p_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rad_H2p_r,yerr=[most_likely_power_rad_H2p_r-actual_values_power_rad_H2p_r_down,actual_values_power_rad_H2p_r_up-most_likely_power_rad_H2p_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rad_H2p, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_heating_rec_r)):
			if len(prob_power_heating_rec_r[i_t])>1:
				for i_r in range(len(prob_power_heating_rec_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_heating_rec_r[i_t][i_r]),dt,intervals_power_heating_rec_r[i_t][i_r+1]-intervals_power_heating_rec_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_heating_rec_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_heating_rec_r,yerr=[most_likely_power_heating_rec_r-actual_values_power_heating_rec_r_down,actual_values_power_heating_rec_r_up-most_likely_power_heating_rec_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_heating_rec, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_rec_neutral_r)):
			if len(prob_power_rec_neutral_r[i_t])>1:
				for i_r in range(len(prob_power_rec_neutral_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_rec_neutral_r[i_t][i_r]),dt,intervals_power_rec_neutral_r[i_t][i_r+1]-intervals_power_rec_neutral_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_rec_neutral_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_rec_neutral_r,yerr=[most_likely_power_rec_neutral_r-actual_values_power_rec_neutral_r_down,actual_values_power_rec_neutral_r_up-most_likely_power_rec_neutral_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_rec_neutral, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_power_via_brem_r)):
			if len(prob_power_via_brem_r[i_t])>1:
				for i_r in range(len(prob_power_via_brem_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_power_via_brem_r[i_t][i_r]),dt,intervals_power_via_brem_r[i_t][i_r+1]-intervals_power_via_brem_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_power_via_brem_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_power_via_brem_r,yerr=[most_likely_power_via_brem_r-actual_values_power_via_brem_r_down,actual_values_power_via_brem_r_up-most_likely_power_via_brem_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of power_via_brem, --=from upstream')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_total_removed_power_r)):
			if len(prob_total_removed_power_r[i_t])>1:
				for i_r in range(len(prob_total_removed_power_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_total_removed_power_r[i_t][i_r]),dt,intervals_total_removed_power_r[i_t][i_r+1]-intervals_total_removed_power_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_total_removed_power_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_total_removed_power_r,yerr=[most_likely_total_removed_power_r-actual_values_total_removed_power_r_down,actual_values_total_removed_power_r_up-most_likely_total_removed_power_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of total_removed_power, --=from upstream\n(ionisation*pot + rad_mol + rad_excit + recombination*pot + brem + rec_neutral)')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_local_CX_r)):
			if len(prob_local_CX_r[i_t])>1:
				for i_r in range(len(prob_local_CX_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_local_CX_r[i_t][i_r]),dt,intervals_local_CX_r[i_t][i_r+1]-intervals_local_CX_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_local_CX_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_local_CX_r,yerr=[most_likely_local_CX_r-actual_values_local_CX_r_down,actual_values_local_CX_r_up-most_likely_local_CX_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of local_CX, --=from upstream\n(ionisation*pot + rad_mol + rad_excit + recombination*pot + brem + rec_neutral)')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		# fig, ax = plt.subplots(1,figsize=(20, 10))
		# for i_t in range(len(prob_total_power_removed_plasma_fluid_r)):
		# 	if len(prob_total_power_removed_plasma_fluid_r[i_t])>1:
		# 		for i_r in range(len(prob_total_power_removed_plasma_fluid_r[i_t])):
		# 			ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_total_power_removed_plasma_fluid_r[i_t][i_r]),dt,intervals_total_power_removed_plasma_fluid_r[i_t][i_r+1]-intervals_total_power_removed_plasma_fluid_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_total_power_removed_plasma_fluid_r[i_t][i_r]))],alpha=0.5)[-1]))
		# plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		# plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		# plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		# plt.errorbar(time_crop,most_likely_total_power_removed_plasma_fluid_r,yerr=[most_likely_total_power_removed_plasma_fluid_r-actual_values_total_power_removed_plasma_fluid_r_down,actual_values_total_power_removed_plasma_fluid_r_up-most_likely_total_power_removed_plasma_fluid_r],color='r',capsize=5)
		# plt.semilogy()
		# plt.ylim(bottom=0.1)
		# plt.xlabel('time from beginning of pulse [ms]')
		# plt.ylabel('Power loss [W]')
		# plt.title(pre_title+'Radial sum of total_power_removed_plasma_fluid, --=from upstream\n(ionisation + radiated +/- recombination heating)')
		# figure_index += 1
		# plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
		# 	figure_index+1) + '.eps', bbox_inches='tight')
		# plt.close('all')

		fig, ax = plt.subplots(1,figsize=(20, 10))
		for i_t in range(len(prob_net_power_removed_plasma_column_r)):
			if len(prob_net_power_removed_plasma_column_r[i_t])>1:
				for i_r in range(len(prob_net_power_removed_plasma_column_r[i_t])):
					ax.add_patch(Rectangle((time_crop[i_t]-dt/2,intervals_net_power_removed_plasma_column_r[i_t][i_r]),dt,intervals_net_power_removed_plasma_column_r[i_t][i_r+1]-intervals_net_power_removed_plasma_column_r[i_t][i_r],facecolor=cm.rainbow([0,1,1/(1-np.log10(prob_net_power_removed_plasma_column_r[i_t][i_r]))],alpha=0.5)[-1]))
		plt.plot(time_crop, heat_inflow_upstream_max,'k--');
		plt.plot(time_crop, heat_inflow_upstream_min,'k--');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,color='y',ls='--',capsize=2)
		plt.errorbar(time_crop,most_likely_net_power_removed_plasma_column_r,yerr=[most_likely_net_power_removed_plasma_column_r-actual_values_net_power_removed_plasma_column_r_down,actual_values_net_power_removed_plasma_column_r_up-most_likely_net_power_removed_plasma_column_r],color='r',capsize=5)
		plt.semilogy()
		plt.ylim(bottom=0.1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Radial sum of net_power_removed_plasma_column, --=from upstream\n(net power removed from plasma column radiated + recombination neutral)')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		energy_variation_dt = np.diff(np.sum((0.5*((homogeneous_mach_number*upstream_adiabatic_collisional_velocity.T).T **2)*hydrogen_mass*J_to_eV +5*Te_all+13.6+2.2)*ne_all*1e20/J_to_eV*area,axis=1)*length)/dt*1000
		if False:	# this is not consistent with what I do for IR traces analysis
			conventional_start_pulse = np.abs(time_crop-0.05).argmin()
			conventional_end_pulse = np.abs(time_crop-0.8).argmin()
		else:
			conventional_start_pulse = power_pulse_shape_crop.argmax() - np.flip((power_pulse_shape_crop[:power_pulse_shape_crop.argmax()]-steady_state_power)>0,axis=0).argmin()
			conventional_end_pulse = power_pulse_shape_crop.argmax() + ((power_pulse_shape_crop[power_pulse_shape_crop.argmax():]-steady_state_power)>0).argmin()
			if (power_pulse_shape_crop[conventional_start_pulse]-steady_state_power)/(power_pulse_shape_crop.max()-steady_state_power)>0.1:
				conventional_start_pulse -= 1
			if (power_pulse_shape_crop[conventional_end_pulse-1]-steady_state_power)/(power_pulse_shape_crop.max()-steady_state_power)>0.1:
				conventional_end_pulse += 1

		plt.figure(figsize=(12, 6));
		# plt.errorbar(time_crop,most_likely_power_rad_excit_r,yerr=[most_likely_power_rad_excit_r-actual_values_power_rad_excit_r_down,actual_values_power_rad_excit_r_up-most_likely_power_rad_excit_r],capsize=5,label='power_rad_excit')
		# plt.errorbar(time_crop,most_likely_power_rad_rec_bremm_r,yerr=[most_likely_power_rad_rec_bremm_r-actual_values_power_rad_rec_bremm_r_down,actual_values_power_rad_rec_bremm_r_up-most_likely_power_rad_rec_bremm_r],capsize=5,label='power_rad_rec_bremm')
		# plt.errorbar(time_crop,most_likely_power_rad_mol_r,yerr=[most_likely_power_rad_mol_r-actual_values_power_rad_mol_r_down,actual_values_power_rad_mol_r_up-most_likely_power_rad_mol_r],capsize=5,label='power_rad_mol')
		plt.plot(time_crop,most_likely_power_rad_excit_r,label='power_rad_excit')
		plt.plot(time_crop,most_likely_power_rad_rec_bremm_r,label='power_rad_rec_bremm')
		plt.plot(time_crop,most_likely_power_rad_mol_r,label='power_rad_mol')
		plt.errorbar(time_crop,most_likely_power_via_ionisation_r,yerr=[most_likely_power_via_ionisation_r-actual_values_power_via_ionisation_r_down,actual_values_power_via_ionisation_r_up-most_likely_power_via_ionisation_r],capsize=5,label='power_via_ionisation (only potential)')
		plt.errorbar(time_crop,most_likely_power_via_recombination_r,yerr=[most_likely_power_via_recombination_r-actual_values_power_via_recombination_r_down,actual_values_power_via_recombination_r_up-most_likely_power_via_recombination_r],capsize=5,label='power_via_recombination (only potential)')
		plt.errorbar(time_crop,most_likely_tot_rad_power_r,yerr=[most_likely_tot_rad_power_r-actual_values_tot_rad_power_r_down,actual_values_tot_rad_power_r_up-most_likely_tot_rad_power_r],capsize=5,label='tot_rad_power')
		plt.errorbar(time_crop,most_likely_net_power_removed_plasma_column_r,yerr=[most_likely_net_power_removed_plasma_column_r-actual_values_net_power_removed_plasma_column_r_down,actual_values_net_power_removed_plasma_column_r_up-most_likely_net_power_removed_plasma_column_r],capsize=5,label='net power removed from plasma column\nradiated + recombination neutral')
		# plt.errorbar(time_crop,most_likely_power_rad_Hm_r,yerr=[most_likely_power_rad_Hm_r-actual_values_power_rad_Hm_r_down,actual_values_power_rad_Hm_r_up-most_likely_power_rad_Hm_r],capsize=5,label='power_rad_Hm')
		# plt.errorbar(time_crop,most_likely_power_rad_H2_r,yerr=[most_likely_power_rad_H2_r-actual_values_power_rad_H2_r_down,actual_values_power_rad_H2_r_up-most_likely_power_rad_H2_r],capsize=5,label='power_rad_H2')
		# plt.errorbar(time_crop,most_likely_power_rad_H2p_r,yerr=[most_likely_power_rad_H2p_r-actual_values_power_rad_H2p_r_down,actual_values_power_rad_H2p_r_up-most_likely_power_rad_H2p_r],capsize=5,label='power_rad_H2p')
		plt.plot(time_crop,most_likely_power_rad_Hm_r,label='power_rad_Hm')
		plt.plot(time_crop,most_likely_power_rad_H2_r,label='power_rad_H2')
		plt.plot(time_crop,most_likely_power_rad_H2p_r,label='power_rad_H2p')
		plt.plot(time_crop,most_likely_power_heating_rec_r,label='power_heating_rec')
		plt.plot(time_crop,most_likely_power_rec_neutral_r,label='power_rec_neutral')
		plt.plot(time_crop,most_likely_power_via_brem_r,label='power_via_brem')
		plt.plot(time_crop,most_likely_local_CX_r,label='local_CX')
		plt.plot(time_crop,most_likely_total_removed_power_r,label='total_removed_power from plasma fliud\nionisation*pot + rad_mol + rad_excit + recombination*pot + brem + rec_neutral')
		plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
		plt.plot(time_crop, heat_inflow_upstream_min,'--k');
		# plt.plot(time_crop, np.sum((merge_Te_prof_multipulse_interp_crop_limited+13.6+2.2)*merge_ne_prof_multipulse_interp_crop_limited*1e20/6.24e18*area,axis=1)*length/dt*1000,'--',label='stored in plasma');
		plt.plot(time_crop[1:], energy_variation_dt,'--',color='gray',label='stored in plasma');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--',label='Power from plasma source')
		plt.grid()
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.legend(loc='best', fontsize='xx-small')
		# plt.ylim(bottom=0,top=max(np.max(actual_values_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		plt.ylim(bottom=0)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Comparison of the true most likely values of radially integrated power loss')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(12, 6));
		plt.errorbar(time_crop,most_likely_power_via_ionisation_r,yerr=[most_likely_power_via_ionisation_r-actual_values_power_via_ionisation_r_down,actual_values_power_via_ionisation_r_up-most_likely_power_via_ionisation_r],capsize=5,label='power_via_ionisation (only potential)')
		plt.errorbar(time_crop,most_likely_power_via_recombination_r,yerr=[most_likely_power_via_recombination_r-actual_values_power_via_recombination_r_down,actual_values_power_via_recombination_r_up-most_likely_power_via_recombination_r],capsize=5,label='power_via_recombination (only potential)')
		# plt.errorbar(time_crop,most_likely_tot_rad_power_r,yerr=[most_likely_tot_rad_power_r-actual_values_tot_rad_power_r_down,actual_values_tot_rad_power_r_up-most_likely_tot_rad_power_r],capsize=5,label='tot_rad_power')
		plt.errorbar(time_crop,most_likely_net_power_removed_plasma_column_r,yerr=[most_likely_net_power_removed_plasma_column_r-actual_values_net_power_removed_plasma_column_r_down,actual_values_net_power_removed_plasma_column_r_up-most_likely_net_power_removed_plasma_column_r],capsize=5,label='net power removed from plasma column\nradiated + recombination neutral')
		plt.errorbar(time_crop,most_likely_total_removed_power_r,yerr=[most_likely_total_removed_power_r-actual_values_total_removed_power_r_down,actual_values_total_removed_power_r_up-most_likely_total_removed_power_r],capsize=5,label='total_removed_power from plasma fliud\nionisation + rad_mol + rad_excit + recombination*pot + brem + rec_neutral')
		plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
		plt.plot(time_crop, heat_inflow_upstream_min,'--k');
		# plt.plot(time_crop, np.sum((merge_Te_prof_multipulse_interp_crop_limited+13.6+2.2)*merge_ne_prof_multipulse_interp_crop_limited*1e20/6.24e18*area,axis=1)*length/dt*1000,'--',label='stored in plasma');
		plt.plot(time_crop[1:], energy_variation_dt,'--',color='gray',label='stored in plasma');
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--',label='Power from plasma source')
		plt.grid()
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.legend(loc='best', fontsize='xx-small')
		# plt.ylim(bottom=0,top=max(np.max(actual_values_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		plt.ylim(bottom=0)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power [W]')
		plt.title(pre_title+'Comparison of the true most likely values of radially integrated power loss\nelements of the local power balance')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(12, 6));
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls=':',label='Power from plasma source')
		plt.plot(time_crop,most_likely_power_via_ionisation_r,color=color[0],label='ionisation')
		plt.plot(time_crop,most_likely_power_via_recombination_r,color=color[1],label='recombination')
		plt.plot(time_crop,most_likely_tot_rad_power_r,color=color[2],label='total radiated power')
		# plt.plot(time_crop,most_likely_power_rad_excit_r,'--',color=color[3],label='power_rad_excit')
		# plt.plot(time_crop,most_likely_power_rad_rec_bremm_r,'--',color=color[4],label='power_rad_rec_bremm')
		# plt.plot(time_crop,most_likely_power_rad_mol_r,'--',color=color[5],label='power_rad_mol')
		# plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
		# plt.plot(time_crop, heat_inflow_upstream_min,'--k');
		# plt.plot(time_crop, np.sum((merge_Te_prof_multipulse_interp_crop_limited+13.6+2.2)*merge_ne_prof_multipulse_interp_crop_limited*1e20/6.24e18*area,axis=1)*length/dt*1000,'--',label='stored in plasma');
		# plt.plot(time_crop[1:], energy_variation_dt,'--',color='gray',label='stored in plasma');
		plt.grid()
		# plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		# plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.legend(loc=1, fontsize='small')
		# plt.ylim(bottom=0,top=max(np.max(actual_values_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		plt.ylim(bottom=0)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power [W]')
		plt.title('Pressure %.3gPa' %(target_chamber_pressure))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(12, 6));
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls=':',label='Power from plasma source')
		# plt.plot(time_crop,most_likely_power_via_ionisation_r,color=color[0],label='power_via_ionisation (only potential)')
		# plt.plot(time_crop,most_likely_power_via_recombination_r,color=color[1],label='power_via_recombination (only potential)')
		# plt.plot(time_crop,most_likely_tot_rad_power_r,color=color[2],label='tot_rad_power')
		plt.plot(time_crop,most_likely_power_rad_excit_r,color=color[3],label='H direct excitation')
		plt.plot(time_crop,most_likely_power_rad_rec_bremm_r,color=color[4],label='H excitation recombination and bremsstrahlung')
		plt.plot(time_crop,most_likely_power_rad_mol_r,color=color[5],label='H excitation from molecular reactions')
		# plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
		# plt.plot(time_crop, heat_inflow_upstream_min,'--k');
		# plt.plot(time_crop, np.sum((merge_Te_prof_multipulse_interp_crop_limited+13.6+2.2)*merge_ne_prof_multipulse_interp_crop_limited*1e20/6.24e18*area,axis=1)*length/dt*1000,'--',label='stored in plasma');
		# plt.plot(time_crop[1:], energy_variation_dt,'--',color='gray',label='stored in plasma');
		plt.grid()
		# plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		# plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.legend(loc=1, fontsize='small')
		# plt.ylim(bottom=0,top=max(np.max(actual_values_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		plt.ylim(bottom=0)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power [W]')
		plt.title('Pressure %.3gPa' %(target_chamber_pressure))
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(12, 6));
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls=':',label='Power from plasma source')
		plt.errorbar(time_crop,most_likely_net_power_removed_plasma_column_r,yerr=[most_likely_net_power_removed_plasma_column_r-actual_values_net_power_removed_plasma_column_r_down,actual_values_net_power_removed_plasma_column_r_up-most_likely_net_power_removed_plasma_column_r],color=color[0],capsize=5,label='net power removed from plasma column\nradiated + recombination neutral')
		plt.plot(time_crop,most_likely_tot_rad_power_r,color=color[1],label='tot_rad_power')
		plt.plot(time_crop,most_likely_power_rad_excit_r,'--',color=color[2],label='power_rad_excit')
		plt.plot(time_crop,most_likely_power_rad_rec_bremm_r,'--',color=color[3],label='power_rad_rec_bremm')
		plt.plot(time_crop,most_likely_power_rad_mol_r,'--',color=color[4],label='power_rad_mol')
		plt.plot(time_crop,most_likely_power_rec_neutral_r,color=color[5],label='power_rec_neutral')
		# plt.plot(time_crop, heat_inflow_upstream_max,'--k', label='Power inflow from upstream:\n'+label_ion_source_at_upstream);
		# plt.plot(time_crop, heat_inflow_upstream_min,'--k');
		# plt.plot(time_crop, np.sum((merge_Te_prof_multipulse_interp_crop_limited+13.6+2.2)*merge_ne_prof_multipulse_interp_crop_limited*1e20/6.24e18*area,axis=1)*length/dt*1000,':',color='gray',label='stored in plasma');
		plt.plot(time_crop[1:], energy_variation_dt,'--',color='gray',label='stored in plasma');
		plt.grid()
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.legend(loc='best', fontsize='xx-small')
		# plt.ylim(bottom=0,top=max(np.max(actual_values_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		plt.ylim(bottom=0)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Comparison of the true most likely values of radially integrated power loss\nelements of the global power balance')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')


		plt.figure(figsize=(10, 5));
		temp = most_likely_power_via_ionisation_r + most_likely_power_via_recombination_r + most_likely_power_rad_mol_r + most_likely_power_rad_excit_r + most_likely_power_via_brem_r + most_likely_power_rec_neutral_r
		labels = ['ionisation (only potential)', 'recombination (only potential)', 'radiated via molecules', 'radiated via excitation', 'bremsstrahlung', 'neutral temp from rec']
		# plt.stackplot(time_crop,most_likely_power_via_ionisation_r/temp,most_likely_power_via_recombination_r/temp,most_likely_power_rad_mol_r/temp,most_likely_power_rad_excit_r/temp,most_likely_power_via_brem_r/temp,most_likely_power_rec_neutral_r/temp,labels=labels)
		plt.stackplot(time_crop,most_likely_power_via_ionisation_r,most_likely_power_via_recombination_r,most_likely_power_rad_mol_r,most_likely_power_rad_excit_r,most_likely_power_via_brem_r,most_likely_power_rec_neutral_r,labels=labels)
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		plt.legend(loc='best', fontsize='xx-small')
		# plt.ylim(bottom=0,top=max(np.max(most_likely_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		# plt.ylim(bottom=1e-1)
		plt.xlabel('time from beginning of pulse [ms]')
		# plt.ylabel('fraction of the total power removed from plasma [au]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Importance of the power loss mechanism in time')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(10, 5));
		temp = most_likely_power_via_ionisation_r + most_likely_power_via_recombination_r + most_likely_power_rad_mol_r + most_likely_power_rad_excit_r + most_likely_power_via_brem_r + most_likely_power_rec_neutral_r
		labels = ['ionisation (only potential)', 'recombination (only potential)', 'radiated via molecules', 'radiated via excitation', 'bremsstrahlung', 'neutral temp from rec']
		plt.stackplot(time_crop,most_likely_power_via_ionisation_r/temp,most_likely_power_via_recombination_r/temp,most_likely_power_rad_mol_r/temp,most_likely_power_rad_excit_r/temp,most_likely_power_via_brem_r/temp,most_likely_power_rec_neutral_r/temp,labels=labels)
		# plt.stackplot(time_crop,most_likely_power_via_ionisation_r,most_likely_power_via_recombination_r,most_likely_power_rad_mol_r,most_likely_power_rad_excit_r,most_likely_power_via_brem_r,most_likely_power_rec_neutral_r,labels=labels)
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,1],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,1],'k--')
		plt.legend(loc='best', fontsize='xx-small')
		plt.ylim(bottom=0,top=1)
		# plt.ylim(bottom=1e-1)
		plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('fraction of the total power removed from plasma [au]')
		# plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Importance of the power loss mechanism in time')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(10, 5));
		temp = most_likely_power_rad_mol_r + most_likely_power_rad_excit_r + most_likely_power_rad_rec_bremm_r + most_likely_power_rec_neutral_r
		labels = ['radiated via molecules', 'radiated via excitation', 'radiated via recombination and bremsstrahlung', 'neutral kin. energy from recombination']
		# plt.stackplot(time_crop,most_likely_power_rad_mol_r/temp,most_likely_power_rad_excit_r/temp,most_likely_power_rad_rec_bremm_r/temp,most_likely_power_rec_neutral_r/temp,labels=labels)
		plt.stackplot(time_crop,most_likely_power_rad_mol_r,most_likely_power_rad_excit_r,most_likely_power_rad_rec_bremm_r,most_likely_power_rec_neutral_r,labels=labels)
		plt.errorbar(time_source_power_crop,power_pulse_shape_crop,yerr=power_pulse_shape_std_crop,ls='--')
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,power_pulse_shape_crop.max()],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,power_pulse_shape_crop.max()],'k--')
		plt.legend(loc='best', fontsize='small')
		# plt.ylim(bottom=0,top=max(np.max(most_likely_power_via_ionisation_r_up),np.max(power_pulse_shape_crop)))
		# plt.ylim(bottom=1e-1)
		plt.xlabel('time from beginning of pulse [ms]')
		# plt.ylabel('fraction of the total power removed from plasma [au]')
		plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Importance of the net power loss from the plasma column in time')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')

		plt.figure(figsize=(10, 5));
		temp = most_likely_power_rad_mol_r + most_likely_power_rad_excit_r + most_likely_power_rad_rec_bremm_r + most_likely_power_rec_neutral_r
		labels = ['radiated via molecules', 'radiated via excitation', 'radiated via recombination and bremsstrahlung', 'neutral kin. energy from recombination']
		plt.stackplot(time_crop,most_likely_power_rad_mol_r/temp,most_likely_power_rad_excit_r/temp,most_likely_power_rad_rec_bremm_r/temp,most_likely_power_rec_neutral_r/temp,labels=labels)
		# plt.stackplot(time_crop,most_likely_power_rad_mol_r,most_likely_power_rad_excit_r,most_likely_power_rad_rec_bremm_r,most_likely_power_rec_neutral_r,labels=labels)
		# plt.semilogy()
		# plt.ylim(bottom=1e0,top=1e6)
		plt.plot([time_crop[conventional_start_pulse]]*2,[0,1],'k--',label='conventional start/end pulse')
		plt.plot([time_crop[conventional_end_pulse-1]]*2,[0,1],'k--')
		plt.legend(loc='best', fontsize='small')
		plt.ylim(bottom=0,top=1)
		# plt.ylim(bottom=1e-1)
		# plt.xlabel('time from beginning of pulse [ms]')
		plt.ylabel('fraction of the total power removed from plasma [au]')
		# plt.ylabel('Power loss [W]')
		plt.title(pre_title+'Importance of the net power loss from the plasma column in time')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')


		def temporal_radial_sum_PDF(intervals_power_ext,prob_power_ext,treshold_sum=1.1,treshold_prob=1e-50,conventional_start_pulse=conventional_start_pulse,conventional_end_pulse=conventional_end_pulse):
			if conventional_start_pulse!=0 or conventional_end_pulse!=len(prob_power_ext):
				intervals_power = intervals_power_ext[conventional_start_pulse:conventional_end_pulse]
				prob_power = prob_power_ext[conventional_start_pulse:conventional_end_pulse]

			for i_t in range(len(intervals_power)-1):
				min_interv = []
				max_interv = []
				sum_prob_power = []
				if i_t==0:
					for i_1 in range(len(intervals_power[i_t])-1):
						for i_2 in range(len(intervals_power[i_t+1])-1):
							min_interv.append(intervals_power[i_t][i_1]+intervals_power[i_t+1][i_2])
							max_interv.append(intervals_power[i_t][i_1+1]+intervals_power[i_t+1][i_2+1])
							sum_prob_power.append(prob_power[i_t][i_1]*prob_power[i_t+1][i_2])
				else:
					for i_1 in range(len(values)-1):
						for i_2 in range(len(intervals_power[i_t+1])-1):
							min_interv.append(values[i_1]+intervals_power[i_t+1][i_2])
							max_interv.append(values[i_1+1]+intervals_power[i_t+1][i_2+1])
							sum_prob_power.append(prob_sum[i_1]*prob_power[i_t+1][i_2])
				min_interv = np.array(min_interv)
				max_interv = np.array(max_interv)
				d_interv = max_interv-min_interv
				sum_prob_power = np.array(sum_prob_power)
				values = np.unique(np.concatenate((min_interv,max_interv)))
				if len(values)==1:
					values = np.concatenate((min_interv,max_interv))
					prob_sum = sum_prob_power
				else:
					prob_sum = []
					for i_start in range(len(values)-1):
						select = np.logical_and(min_interv<=values[i_start],max_interv>=values[i_start+1])
						prob_sum.append(np.sum(sum_prob_power[select]*(values[i_start+1]-values[i_start])/(d_interv[select])))
				if len(values)>30:
					treshold = 1/1000
					temp = np.cumsum(prob_sum)
					temp1=[0]
					temp2 = []
					for i in range(1,1000):
						loc = np.abs(temp-treshold*i).argmin()
						temp1.append(values[loc+1])
						if i==1:
							temp2.append(temp[loc])
						else:
							temp2.append(temp[loc]-temp[prev_loc])
						prev_loc=loc
					temp1.append(values[-1])
					temp2.append(temp[-1]-np.sum(temp2))
					values = np.array(temp1)
					prob_sum = np.array(temp2)
					done=0
					limit = len(values)*3
					while (done==0 and limit>0):
						limit-=1
						if limit==1:
							print('falied PDF building in t=%.3g, r=%.3g' %(i_t,i_r))
						for i in range(1,len(values)-2):
							# print(values)
							if ((values[i+1]/values[i]<treshold_sum or prob_sum[i]<treshold_prob) and len(values)>10):
								values = np.concatenate((values[:i+1],values[i+2:]))
								prob_sum = np.concatenate((prob_sum[:i],[np.sum(prob_sum[i:i+2])],prob_sum[i+2:]))
								break
							if i==len(values)-3:
								done=1

			return values*dt/1000,prob_sum

		def temporal_radial_sum_PDF_MC(actual_values_power_ext,prob_power_ext,intervals=30,samples=100000,conventional_start_pulse=conventional_start_pulse,conventional_end_pulse=conventional_end_pulse):
			if conventional_start_pulse!=0 or conventional_end_pulse!=len(prob_power_ext):
				actual_values_power = actual_values_power_ext[conventional_start_pulse:conventional_end_pulse]
				prob_power = prob_power_ext[conventional_start_pulse:conventional_end_pulse]
			temp_values = np.zeros((samples))
			for i_t in range(len(prob_power)-1):
				if len(actual_values_power[i_t])>1:
					temp_values += np.random.choice(actual_values_power[i_t],size=samples,p=prob_power[i_t])
			if all(temp_values==0):
				out_values=np.array([0,0])
				out_prob_sum=np.array([1])
			else:
				temp_prob,temp_intervals = np.histogram(temp_values,bins=np.logspace(np.log10(temp_values.min()),np.log10(temp_values.max()),intervals+1))
				out_values=temp_intervals*dt/1000
				out_prob_sum=temp_prob/np.sum(temp_prob)
			return out_values,out_prob_sum


		# intervals_power_rad_excit_tr, prob_power_rad_excit_tr = temporal_radial_sum_PDF(intervals_power_rad_excit_r,prob_power_rad_excit_r)
		# intervals_power_rad_rec_bremm_tr, prob_power_rad_rec_bremm_tr = temporal_radial_sum_PDF(intervals_power_rad_rec_bremm_r,prob_power_rad_rec_bremm_r)
		# intervals_power_rad_mol_tr, prob_power_rad_mol_tr = temporal_radial_sum_PDF(intervals_power_rad_mol_r,prob_power_rad_mol_r)
		# intervals_power_via_ionisation_tr, prob_power_via_ionisation_tr = temporal_radial_sum_PDF(intervals_power_via_ionisation_r,prob_power_via_ionisation_r)
		# intervals_power_via_recombination_tr, prob_power_via_recombination_tr = temporal_radial_sum_PDF(intervals_power_via_recombination_r,prob_power_via_recombination_r)
		# intervals_tot_rad_power_tr, prob_tot_rad_power_tr = temporal_radial_sum_PDF(intervals_tot_rad_power_r, prob_tot_rad_power_r)
		# intervals_power_rad_Hm_tr, prob_power_rad_Hm_tr = temporal_radial_sum_PDF(intervals_power_rad_Hm_r,prob_power_rad_Hm_r)
		# intervals_power_rad_H2_tr, prob_power_rad_H2_tr = temporal_radial_sum_PDF(intervals_power_rad_H2_r,prob_power_rad_H2_r)
		# intervals_power_rad_H2p_tr, prob_power_rad_H2p_tr = temporal_radial_sum_PDF(intervals_power_rad_H2p_r,prob_power_rad_H2p_r)
		# intervals_power_heating_rec_tr, prob_power_heating_rec_tr = temporal_radial_sum_PDF(intervals_power_heating_rec_r,prob_power_heating_rec_r)
		# intervals_power_rec_neutral_tr, prob_power_rec_neutral_tr = temporal_radial_sum_PDF(intervals_power_rec_neutral_r,prob_power_rec_neutral_r)
		# intervals_power_via_brem_tr, prob_power_via_brem_tr = temporal_radial_sum_PDF(intervals_power_via_brem_r,prob_power_via_brem_r)
		# intervals_total_removed_power_tr, prob_total_removed_power_tr = temporal_radial_sum_PDF(intervals_total_removed_power_r,prob_total_removed_power_r)
		# intervals_local_CX_tr, prob_local_CX_tr = temporal_radial_sum_PDF(intervals_local_CX_r,prob_local_CX_r)
		# # intervals_total_power_removed_plasma_fluid_tr, prob_total_power_removed_plasma_fluid_tr = temporal_radial_sum_PDF(intervals_total_power_removed_plasma_fluid_r,prob_total_power_removed_plasma_fluid_r)
		# intervals_net_power_removed_plasma_column_tr, prob_net_power_removed_plasma_column_tr = temporal_radial_sum_PDF(intervals_net_power_removed_plasma_column_r,prob_net_power_removed_plasma_column_r)
		intervals_power_rad_excit_tr, prob_power_rad_excit_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_excit_r,prob_power_rad_excit_r)
		intervals_power_rad_rec_bremm_tr, prob_power_rad_rec_bremm_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_rec_bremm_r,prob_power_rad_rec_bremm_r)
		intervals_power_rad_mol_tr, prob_power_rad_mol_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_mol_r,prob_power_rad_mol_r)
		intervals_power_via_ionisation_tr, prob_power_via_ionisation_tr = temporal_radial_sum_PDF_MC(actual_values_power_via_ionisation_r,prob_power_via_ionisation_r)
		intervals_power_via_recombination_tr, prob_power_via_recombination_tr = temporal_radial_sum_PDF_MC(actual_values_power_via_recombination_r,prob_power_via_recombination_r)
		intervals_tot_rad_power_tr, prob_tot_rad_power_tr = temporal_radial_sum_PDF_MC(actual_values_tot_rad_power_r, prob_tot_rad_power_r)
		intervals_power_rad_Hm_tr, prob_power_rad_Hm_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_Hm_r,prob_power_rad_Hm_r)
		intervals_power_rad_H2_tr, prob_power_rad_H2_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_H2_r,prob_power_rad_H2_r)
		intervals_power_rad_H2p_tr, prob_power_rad_H2p_tr = temporal_radial_sum_PDF_MC(actual_values_power_rad_H2p_r,prob_power_rad_H2p_r)
		intervals_power_heating_rec_tr, prob_power_heating_rec_tr = temporal_radial_sum_PDF_MC(actual_values_power_heating_rec_r,prob_power_heating_rec_r)
		intervals_power_rec_neutral_tr, prob_power_rec_neutral_tr = temporal_radial_sum_PDF_MC(actual_values_power_rec_neutral_r,prob_power_rec_neutral_r)
		intervals_power_via_brem_tr, prob_power_via_brem_tr = temporal_radial_sum_PDF_MC(actual_values_power_via_brem_r,prob_power_via_brem_r)
		intervals_total_removed_power_tr, prob_total_removed_power_tr = temporal_radial_sum_PDF_MC(actual_values_total_removed_power_r,prob_total_removed_power_r)
		intervals_local_CX_tr, prob_local_CX_tr = temporal_radial_sum_PDF_MC(actual_values_local_CX_r,prob_local_CX_r)
		# intervals_total_power_removed_plasma_fluid_tr, prob_total_power_removed_plasma_fluid_tr = temporal_radial_sum_PDF_MC(actual_values_total_power_removed_plasma_fluid_r,prob_total_power_removed_plasma_fluid_r)
		intervals_net_power_removed_plasma_column_tr, prob_net_power_removed_plasma_column_tr = temporal_radial_sum_PDF_MC(actual_values_net_power_removed_plasma_column_r,prob_net_power_removed_plasma_column_r)

		ML_power_rad_excit = (np.add(intervals_power_rad_excit_tr[1:],intervals_power_rad_excit_tr[:-1])/2)[np.array(prob_power_rad_excit_tr).argmax()]
		temp = np.cumsum(prob_power_rad_excit_tr)
		ML_power_rad_excit_sigma = np.mean([ML_power_rad_excit-intervals_power_rad_excit_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_excit_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_excit])
		ML_power_rad_rec_bremm = (np.add(intervals_power_rad_rec_bremm_tr[1:],intervals_power_rad_rec_bremm_tr[:-1])/2)[np.array(prob_power_rad_rec_bremm_tr).argmax()]
		temp = np.cumsum(prob_power_rad_rec_bremm_tr)
		ML_power_rad_rec_bremm_sigma = np.mean([ML_power_rad_rec_bremm-intervals_power_rad_rec_bremm_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_rec_bremm_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_rec_bremm])
		ML_power_rad_mol = (np.add(intervals_power_rad_mol_tr[1:],intervals_power_rad_mol_tr[:-1])/2)[np.array(prob_power_rad_mol_tr).argmax()]
		temp = np.cumsum(prob_power_rad_mol_tr)
		ML_power_rad_mol_sigma = np.mean([ML_power_rad_mol-intervals_power_rad_mol_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_mol_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_mol])
		ML_power_via_ionisation = (np.add(intervals_power_via_ionisation_tr[1:],intervals_power_via_ionisation_tr[:-1])/2)[np.array(prob_power_via_ionisation_tr).argmax()]
		temp = np.cumsum(prob_power_via_ionisation_tr)
		ML_power_via_ionisation_sigma = np.mean([ML_power_via_ionisation-intervals_power_via_ionisation_tr[np.abs(temp-0.159).argmin()+1],intervals_power_via_ionisation_tr[np.abs(temp-1+0.159).argmin()]-ML_power_via_ionisation])
		ML_power_via_recombination = (np.add(intervals_power_via_recombination_tr[1:],intervals_power_via_recombination_tr[:-1])/2)[np.array(prob_power_via_recombination_tr).argmax()]
		temp = np.cumsum(prob_power_via_recombination_tr)
		ML_power_via_recombination_sigma = np.mean([ML_power_via_recombination-intervals_power_via_recombination_tr[np.abs(temp-0.159).argmin()+1],intervals_power_via_recombination_tr[np.abs(temp-1+0.159).argmin()]-ML_power_via_recombination])
		ML_tot_rad_power = (np.add(intervals_tot_rad_power_tr[1:],intervals_tot_rad_power_tr[:-1])/2)[np.array(prob_tot_rad_power_tr).argmax()]
		temp = np.cumsum(prob_tot_rad_power_tr)
		ML_tot_rad_power_sigma = np.mean([ML_tot_rad_power-intervals_tot_rad_power_tr[np.abs(temp-0.159).argmin()+1],intervals_tot_rad_power_tr[np.abs(temp-1+0.159).argmin()]-ML_tot_rad_power])
		ML_power_rad_Hm = (np.add(intervals_power_rad_Hm_tr[1:],intervals_power_rad_Hm_tr[:-1])/2)[np.array(prob_power_rad_Hm_tr).argmax()]
		temp = np.cumsum(prob_power_rad_Hm_tr)
		ML_power_rad_Hm_sigma = np.mean([ML_power_rad_Hm-intervals_power_rad_Hm_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_Hm_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_Hm])
		ML_power_rad_H2 = (np.add(intervals_power_rad_H2_tr[1:],intervals_power_rad_H2_tr[:-1])/2)[np.array(prob_power_rad_H2_tr).argmax()]
		temp = np.cumsum(prob_power_rad_H2_tr)
		ML_power_rad_H2_sigma = np.mean([ML_power_rad_H2-intervals_power_rad_H2_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_H2_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_H2])
		ML_power_rad_H2p = (np.add(intervals_power_rad_H2p_tr[1:],intervals_power_rad_H2p_tr[:-1])/2)[np.array(prob_power_rad_H2p_tr).argmax()]
		temp = np.cumsum(prob_power_rad_H2p_tr)
		ML_power_rad_H2p_sigma = np.mean([ML_power_rad_H2p-intervals_power_rad_H2p_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rad_H2p_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rad_H2p])
		ML_power_heating_rec = (np.add(intervals_power_heating_rec_tr[1:],intervals_power_heating_rec_tr[:-1])/2)[np.array(prob_power_heating_rec_tr).argmax()]
		temp = np.cumsum(prob_power_heating_rec_tr)
		ML_power_heating_rec_sigma = np.mean([ML_power_heating_rec-intervals_power_heating_rec_tr[np.abs(temp-0.159).argmin()+1],intervals_power_heating_rec_tr[np.abs(temp-1+0.159).argmin()]-ML_power_heating_rec])
		ML_power_rec_neutral = (np.add(intervals_power_rec_neutral_tr[1:],intervals_power_rec_neutral_tr[:-1])/2)[np.array(prob_power_rec_neutral_tr).argmax()]
		temp = np.cumsum(prob_power_rec_neutral_tr)
		ML_power_rec_neutral_sigma = np.mean([ML_power_rec_neutral-intervals_power_rec_neutral_tr[np.abs(temp-0.159).argmin()+1],intervals_power_rec_neutral_tr[np.abs(temp-1+0.159).argmin()]-ML_power_rec_neutral])
		ML_power_via_brem = (np.add(intervals_power_via_brem_tr[1:],intervals_power_via_brem_tr[:-1])/2)[np.array(prob_power_via_brem_tr).argmax()]
		temp = np.cumsum(prob_power_via_brem_tr)
		ML_power_via_brem_sigma = np.mean([ML_power_via_brem-intervals_power_via_brem_tr[np.abs(temp-0.159).argmin()+1],intervals_power_via_brem_tr[np.abs(temp-1+0.159).argmin()]-ML_power_via_brem])
		ML_total_removed_power = (np.add(intervals_total_removed_power_tr[1:],intervals_total_removed_power_tr[:-1])/2)[np.array(prob_total_removed_power_tr).argmax()]
		temp = np.cumsum(prob_total_removed_power_tr)
		ML_total_removed_power_sigma = np.mean([ML_total_removed_power-intervals_total_removed_power_tr[np.abs(temp-0.159).argmin()+1],intervals_total_removed_power_tr[np.abs(temp-1+0.159).argmin()]-ML_total_removed_power])
		ML_local_CX = (np.add(intervals_local_CX_tr[1:],intervals_local_CX_tr[:-1])/2)[np.array(prob_local_CX_tr).argmax()]
		temp = np.cumsum(prob_local_CX_tr)
		ML_local_CX_sigma = np.mean([ML_local_CX-intervals_local_CX_tr[np.abs(temp-0.159).argmin()+1],intervals_local_CX_tr[np.abs(temp-1+0.159).argmin()]-ML_local_CX])
		ML_net_power_removed_plasma_column = (np.add(intervals_net_power_removed_plasma_column_tr[1:],intervals_net_power_removed_plasma_column_tr[:-1])/2)[np.array(prob_net_power_removed_plasma_column_tr).argmax()]
		temp = np.cumsum(prob_net_power_removed_plasma_column_tr)
		ML_net_power_removed_plasma_column_sigma = np.mean([ML_net_power_removed_plasma_column-intervals_net_power_removed_plasma_column_tr[np.abs(temp-0.159).argmin()+1],intervals_net_power_removed_plasma_column_tr[np.abs(temp-1+0.159).argmin()]-ML_net_power_removed_plasma_column])

		results_summary = pd.read_csv('/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv',index_col=0)
		results_summary.loc[merge_ID_target,['B','Seed','p_n [Pa]','CB energy [J]','Delivered energy [J]','T_axial','Target','power_rad_excit','power_rad_excit_sigma','power_rad_rec_bremm','power_rad_rec_bremm_sigma','power_rad_mol','power_rad_mol_sigma','power_via_ionisation','power_via_ionisation_sigma','power_via_recombination','power_via_recombination_sigma','tot_rad_power','tot_rad_power_sigma','power_rad_Hm','power_rad_Hm_sigma','power_rad_H2','power_rad_H2_sigma','power_rad_H2p','power_rad_H2p_sigma','power_heating_rec','power_heating_rec_sigma','power_rec_neutral','power_rec_neutral_sigma','power_via_brem','power_via_brem_sigma','total_removed_power','total_removed_power_sigma','local_CX','local_CX_sigma','max_CX_energy','max_CX_energy_sigma','net_power_removed_plasma_column','net_power_removed_plasma_column_sigma']]=magnetic_field,feed_rate_SLM,target_chamber_pressure,0.5*(capacitor_voltage**2)*150e-6,energy_delivered_good_pulses,target_OES_distance,target_material,ML_power_rad_excit,ML_power_rad_excit_sigma,ML_power_rad_rec_bremm,ML_power_rad_rec_bremm_sigma,ML_power_rad_mol,ML_power_rad_mol_sigma,ML_power_via_ionisation,ML_power_via_ionisation_sigma,ML_power_via_recombination,ML_power_via_recombination_sigma,ML_tot_rad_power,ML_tot_rad_power_sigma,ML_power_rad_Hm,ML_power_rad_Hm_sigma,ML_power_rad_H2,ML_power_rad_H2_sigma,ML_power_rad_H2p,ML_power_rad_H2p_sigma,ML_power_heating_rec,ML_power_heating_rec_sigma,ML_power_rec_neutral,ML_power_rec_neutral_sigma,ML_power_via_brem,ML_power_via_brem_sigma,ML_total_removed_power,ML_total_removed_power_sigma,ML_local_CX,ML_local_CX_sigma,ML_E_HCX,ML_E_HCX_sigma,ML_net_power_removed_plasma_column,ML_net_power_removed_plasma_column_sigma
		results_summary.to_csv(path_or_buf='/home/ffederic/work/Collaboratory/test/experimental_data/results_summary.csv')

		plt.figure(figsize=(20, 10));
		plt.plot(np.sort(intervals_power_rad_excit_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_excit_tr]*2).T.flatten(),label='power_rad_excit ML=%.3g+/-%.3gJ' %(ML_power_rad_excit,ML_power_rad_excit_sigma))
		plt.plot(np.sort(intervals_power_rad_rec_bremm_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_rec_bremm_tr]*2).T.flatten(),label='power_rad_rec_bremm ML=%.3g+/-%.3gJ' %(ML_power_rad_rec_bremm,ML_power_rad_rec_bremm_sigma))
		plt.plot(np.sort(intervals_power_rad_mol_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_mol_tr]*2).T.flatten(),label='power_rad_mol ML=%.3g+/-%.3gJ' %(ML_power_rad_mol,ML_power_rad_mol_sigma))
		plt.plot(np.sort(intervals_power_via_ionisation_tr.tolist()*2)[1:-1],100*np.array([prob_power_via_ionisation_tr]*2).T.flatten(),label='power_via_ionisation (only potential) ML=%.3g+/-%.3gJ' %(ML_power_via_ionisation,ML_power_via_ionisation_sigma))
		plt.plot(np.sort(intervals_power_via_recombination_tr.tolist()*2)[1:-1],100*np.array([prob_power_via_recombination_tr]*2).T.flatten(),label='power_via_recombination (only potential) ML=%.3g+/-%.3gJ' %(ML_power_via_recombination,ML_power_via_recombination_sigma))
		plt.plot(np.sort(intervals_tot_rad_power_tr.tolist()*2)[1:-1],100*np.array([prob_tot_rad_power_tr]*2).T.flatten(),label='tot_rad_power ML=%.3g+/-%.3gJ' %(ML_tot_rad_power,ML_tot_rad_power_sigma))
		plt.plot(np.sort(intervals_power_rad_Hm_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_Hm_tr]*2).T.flatten(),label='power_rad_Hm ML=%.3g+/-%.3gJ' %(ML_power_rad_Hm,ML_power_rad_Hm_sigma))
		plt.plot(np.sort(intervals_power_rad_H2_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_H2_tr]*2).T.flatten(),label='power_rad_H2 ML=%.3g+/-%.3gJ' %(ML_power_rad_H2,ML_power_rad_H2_sigma))
		plt.plot(np.sort(intervals_power_rad_H2p_tr.tolist()*2)[1:-1],100*np.array([prob_power_rad_H2p_tr]*2).T.flatten(),label='power_rad_H2p ML=%.3g+/-%.3gJ' %(ML_power_rad_H2p,ML_power_rad_H2p_sigma))
		plt.plot(np.sort(intervals_power_heating_rec_tr.tolist()*2)[1:-1],100*np.array([prob_power_heating_rec_tr]*2).T.flatten(),label='power_heating_rec ML=%.3g+/-%.3gJ' %(ML_power_heating_rec,ML_power_heating_rec_sigma))
		plt.plot(np.sort(intervals_power_rec_neutral_tr.tolist()*2)[1:-1],100*np.array([prob_power_rec_neutral_tr]*2).T.flatten(),label='power_rec_neutral ML=%.3g+/-%.3gJ' %(ML_power_rec_neutral,ML_power_rec_neutral_sigma))
		plt.plot(np.sort(intervals_power_via_brem_tr.tolist()*2)[1:-1],100*np.array([prob_power_via_brem_tr]*2).T.flatten(),label='power_via_brem ML=%.3g+/-%.3gJ' %(ML_power_via_brem,ML_power_via_brem_sigma))
		plt.plot(np.sort(intervals_E_HCX.tolist()*2)[1:-1],100*np.array([prob_E_HCX]*2).T.flatten(),label='max CX ML=%.3g+/-%.3gJ' %(ML_E_HCX,ML_E_HCX_sigma))
		plt.plot(np.sort(intervals_total_removed_power_tr.tolist()*2)[1:-1],100*np.array([prob_total_removed_power_tr]*2).T.flatten(),label='total_removed_power from plasma fliud\nionisation*pot + rad_mol + rad_excit + recombination*pot + brem + rec_neutral ML=%.3g+/-%.3gJ' %(ML_total_removed_power,ML_total_removed_power_sigma))
		plt.plot(np.sort(intervals_net_power_removed_plasma_column_tr.tolist()*2)[1:-1],100*np.array([prob_net_power_removed_plasma_column_tr]*2).T.flatten(),label='net power removed from plasma column\nradiated + recombination neutral ML=%.3g+/-%.3gJ' %(ML_net_power_removed_plasma_column,ML_net_power_removed_plasma_column_sigma))
		plt.plot([np.sum(heat_inflow_upstream_max[conventional_start_pulse:conventional_end_pulse]*dt/1000)]*2,[0,100],'k--', label='Min/max power inflow from upstream:\n'+label_ion_source_at_upstream);
		plt.plot([np.sum(heat_inflow_upstream_min[conventional_start_pulse:conventional_end_pulse]*dt/1000)]*2,[0,100],'k--');
		plt.plot([(np.sum(power_pulse_shape_crop[conventional_start_pulse:conventional_end_pulse])+np.sum(power_pulse_shape_std_crop[conventional_start_pulse:conventional_end_pulse]**2)**0.5)*dt/1000]*2,[0,100],'r--',label='Power from plasma source ML=%.3g+/-%.3gJ, SS%.3gJ, net%.3gJ' %(np.sum(power_pulse_shape_crop[conventional_start_pulse:conventional_end_pulse])*dt/1000,dt/1000*np.sum(power_pulse_shape_std_crop[conventional_start_pulse:conventional_end_pulse]**2)**0.5,steady_state_power*(conventional_end_pulse - conventional_start_pulse)*dt/1000,np.sum(power_pulse_shape_crop[conventional_start_pulse:conventional_end_pulse])*dt/1000-steady_state_power*(conventional_end_pulse - conventional_start_pulse)*dt/1000));
		plt.plot([(np.sum(power_pulse_shape_crop[conventional_start_pulse:conventional_end_pulse])-np.sum(power_pulse_shape_std_crop[conventional_start_pulse:conventional_end_pulse]**2)**0.5)*dt/1000]*2,[0,100],'r--');
		plt.semilogx()
		plt.grid()
		plt.legend(loc='best', fontsize='xx-small')
		# plt.xlim(left=np.min(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]),right=np.max(intervals_power_rad_excit_tr[1:][np.array(prob_power_rad_excit_tr)>1e-20]))
		plt.xlim(left=1e0)
		# plt.ylim(bottom=1e-1)
		plt.xlabel('Energy [J]')
		plt.ylabel('Likelyhood [au]')
		plt.title(pre_title+'Time and spatially integrated energy lost in the pulse')
		figure_index += 1
		plt.savefig(path_where_to_save_everything + mod4 + '/pass_'+str(global_pass)+'_merge'+str(merge_ID_target)+'_global_fit' + str(
			figure_index+1) + '.eps', bbox_inches='tight')
		plt.close('all')


		np.savez_compressed(path_where_to_save_everything + mod4 +'/bayesian_results'+str(global_pass),intervals_power_rad_excit=intervals_power_rad_excit,prob_power_rad_excit=prob_power_rad_excit,actual_values_power_rad_excit=actual_values_power_rad_excit,intervals_power_rad_excit_r=intervals_power_rad_excit_r,prob_power_rad_excit_r=prob_power_rad_excit_r,actual_values_power_rad_excit_r=actual_values_power_rad_excit_r,intervals_power_rad_excit_tr=intervals_power_rad_excit_tr,prob_power_rad_excit_tr=prob_power_rad_excit_tr,ML_power_rad_excit=ML_power_rad_excit, intervals_power_rad_rec_bremm=intervals_power_rad_rec_bremm,prob_power_rad_rec_bremm=prob_power_rad_rec_bremm,actual_values_power_rad_rec_bremm=actual_values_power_rad_rec_bremm,intervals_power_rad_rec_bremm_r=intervals_power_rad_rec_bremm_r,prob_power_rad_rec_bremm_r=prob_power_rad_rec_bremm_r,actual_values_power_rad_rec_bremm_r=actual_values_power_rad_rec_bremm_r,intervals_power_rad_rec_bremm_tr=intervals_power_rad_rec_bremm_tr,prob_power_rad_rec_bremm_tr=prob_power_rad_rec_bremm_tr,ML_power_rad_rec_bremm=ML_power_rad_rec_bremm, intervals_power_rad_mol=intervals_power_rad_mol,prob_power_rad_mol=prob_power_rad_mol,actual_values_power_rad_mol=actual_values_power_rad_mol,intervals_power_rad_mol_r=intervals_power_rad_mol_r,prob_power_rad_mol_r=prob_power_rad_mol_r,actual_values_power_rad_mol_r=actual_values_power_rad_mol_r,intervals_power_rad_mol_tr=intervals_power_rad_mol_tr,prob_power_rad_mol_tr=prob_power_rad_mol_tr,ML_power_rad_mol=ML_power_rad_mol, intervals_power_via_ionisation=intervals_power_via_ionisation,prob_power_via_ionisation=prob_power_via_ionisation,actual_values_power_via_ionisation=actual_values_power_via_ionisation,intervals_power_via_ionisation_r=intervals_power_via_ionisation_r,prob_power_via_ionisation_r=prob_power_via_ionisation_r,actual_values_power_via_ionisation_r=actual_values_power_via_ionisation_r,intervals_power_via_ionisation_tr=intervals_power_via_ionisation_tr,prob_power_via_ionisation_tr=prob_power_via_ionisation_tr,ML_power_via_ionisation=ML_power_via_ionisation, intervals_power_via_recombination=intervals_power_via_recombination,prob_power_via_recombination=prob_power_via_recombination,actual_values_power_via_recombination=actual_values_power_via_recombination,intervals_power_via_recombination_r=intervals_power_via_recombination_r,prob_power_via_recombination_r=prob_power_via_recombination_r,actual_values_power_via_recombination_r=actual_values_power_via_recombination_r,intervals_power_via_recombination_tr=intervals_power_via_recombination_tr,prob_power_via_recombination_tr=prob_power_via_recombination_tr,ML_power_via_recombination=ML_power_via_recombination, intervals_tot_rad_power=intervals_tot_rad_power,prob_tot_rad_power=prob_tot_rad_power,actual_values_tot_rad_power=actual_values_tot_rad_power,intervals_tot_rad_power_r=intervals_tot_rad_power_r,prob_tot_rad_power_r=prob_tot_rad_power_r,actual_values_tot_rad_power_r=actual_values_tot_rad_power_r,intervals_tot_rad_power_tr=intervals_tot_rad_power_tr,prob_tot_rad_power_tr=prob_tot_rad_power_tr,ML_tot_rad_power=ML_tot_rad_power, intervals_power_rad_Hm=intervals_power_rad_Hm,prob_power_rad_Hm=prob_power_rad_Hm,actual_values_power_rad_Hm=actual_values_power_rad_Hm,intervals_power_rad_Hm_r=intervals_power_rad_Hm_r,prob_power_rad_Hm_r=prob_power_rad_Hm_r,actual_values_power_rad_Hm_r=actual_values_power_rad_Hm_r,intervals_power_rad_Hm_tr=intervals_power_rad_Hm_tr,prob_power_rad_Hm_tr=prob_power_rad_Hm_tr,ML_power_rad_Hm=ML_power_rad_Hm, intervals_power_rad_H2=intervals_power_rad_H2,prob_power_rad_H2=prob_power_rad_H2,actual_values_power_rad_H2=actual_values_power_rad_H2,intervals_power_rad_H2_r=intervals_power_rad_H2_r,prob_power_rad_H2_r=prob_power_rad_H2_r,actual_values_power_rad_H2_r=actual_values_power_rad_H2_r,intervals_power_rad_H2_tr=intervals_power_rad_H2_tr,prob_power_rad_H2_tr=prob_power_rad_H2_tr,ML_power_rad_H2=ML_power_rad_H2, intervals_power_rad_H2p=intervals_power_rad_H2p,prob_power_rad_H2p=prob_power_rad_H2p,actual_values_power_rad_H2p=actual_values_power_rad_H2p,intervals_power_rad_H2p_r=intervals_power_rad_H2p_r,prob_power_rad_H2p_r=prob_power_rad_H2p_r,actual_values_power_rad_H2p_r=actual_values_power_rad_H2p_r,intervals_power_rad_H2p_tr=intervals_power_rad_H2p_tr,prob_power_rad_H2p_tr=prob_power_rad_H2p_tr,ML_power_rad_H2p=ML_power_rad_H2p, intervals_power_heating_rec=intervals_power_heating_rec,prob_power_heating_rec=prob_power_heating_rec,actual_values_power_heating_rec=actual_values_power_heating_rec,intervals_power_heating_rec_r=intervals_power_heating_rec_r,prob_power_heating_rec_r=prob_power_heating_rec_r,actual_values_power_heating_rec_r=actual_values_power_heating_rec_r,intervals_power_heating_rec_tr=intervals_power_heating_rec_tr,prob_power_heating_rec_tr=prob_power_heating_rec_tr,ML_power_heating_rec=ML_power_heating_rec, intervals_power_rec_neutral=intervals_power_rec_neutral,prob_power_rec_neutral=prob_power_rec_neutral,actual_values_power_rec_neutral=actual_values_power_rec_neutral,intervals_power_rec_neutral_r=intervals_power_rec_neutral_r,prob_power_rec_neutral_r=prob_power_rec_neutral_r,actual_values_power_rec_neutral_r=actual_values_power_rec_neutral_r,intervals_power_rec_neutral_tr=intervals_power_rec_neutral_tr,prob_power_rec_neutral_tr=prob_power_rec_neutral_tr,ML_power_rec_neutral=ML_power_rec_neutral, intervals_power_via_brem=intervals_power_via_brem,prob_power_via_brem=prob_power_via_brem,actual_values_power_via_brem=actual_values_power_via_brem,intervals_power_via_brem_r=intervals_power_via_brem_r,prob_power_via_brem_r=prob_power_via_brem_r,actual_values_power_via_brem_r=actual_values_power_via_brem_r,intervals_power_via_brem_tr=intervals_power_via_brem_tr,prob_power_via_brem_tr=prob_power_via_brem_tr,ML_power_via_brem=ML_power_via_brem, intervals_total_removed_power=intervals_total_removed_power,prob_total_removed_power=prob_total_removed_power,actual_values_total_removed_power=actual_values_total_removed_power,intervals_total_removed_power_r=intervals_total_removed_power_r,prob_total_removed_power_r=prob_total_removed_power_r,actual_values_total_removed_power_r=actual_values_total_removed_power_r,intervals_total_removed_power_tr=intervals_total_removed_power_tr,prob_total_removed_power_tr=prob_total_removed_power_tr,ML_total_removed_power=ML_total_removed_power, intervals_local_CX=intervals_local_CX,prob_local_CX=prob_local_CX,actual_values_local_CX=actual_values_local_CX,intervals_local_CX_r=intervals_local_CX_r,prob_local_CX_r=prob_local_CX_r,actual_values_local_CX_r=actual_values_local_CX_r,intervals_local_CX_tr=intervals_local_CX_tr,prob_local_CX_tr=prob_local_CX_tr,ML_local_CX=ML_power_rad_mol, intervals_net_power_removed_plasma_column_r=intervals_net_power_removed_plasma_column_r,prob_net_power_removed_plasma_column_r=prob_net_power_removed_plasma_column_r,actual_values_net_power_removed_plasma_column_r=actual_values_net_power_removed_plasma_column_r)

mkl.set_num_threads(1)
