import numpy as np
from scipy import interpolate
from scipy.io import loadmat
import os
import collect_and_eval as coleval
from adas import read_adf15,read_adf11
pecfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_balmer#h0.dat'
pecfile_2 = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/pec12#h_pju#h0.dat'
scdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/scd12_h.dat'
acdfile = '/home/ffederic/work/Collaboratory/test/experimental_data/functions/acd12_h.dat'
# exec(open("/home/ffederic/work/analysis_scripts/scripts/preamble_import_pc.py").read())

os.chdir("/home/ffederic/work/Collaboratory/test/experimental_data/functions/MolRad_Yacora/Yacora_FF")


path = os.getcwd()

f = []
for (dirpath, dirnames, filenames) in os.walk(path):
	f.extend(dirnames)

for folder in f:
	filenames = coleval.all_file_names(path + '/' + folder, 'dat')
	filenames = coleval.order_filenames(filenames)

	data=[]
	for fname in filenames:
		print(fname)
		print(np.shape(np.genfromtxt(path + '/' + folder+'/'+fname)))
		data.append(np.genfromtxt(path + '/' + folder+'/'+fname))
	data=np.array(data)
	print('creation of PEC_' + folder)
	exec('PEC_' + folder + "=data")


# Direct excitation
# H(q) + e- → H(p>q) + e-
# n = 2-13

if False:
	T_e_From_H = PEC_From_H[:,:,0].T.flatten()
	# Te_HmH2p_N = np.repeat(Te_HmH2p_N,len(excited_states)).reshape((len(Te_HmH2p_N),len(excited_states))).T.flatten()
	T_H_From_H = PEC_From_H[:,:,1].T.flatten()
	n_e_From_H = PEC_From_H[:,:,2].T.flatten()
	n_Hp_From_H = PEC_From_H[:,:,3].T.flatten()
	n_H_From_H = PEC_From_H[:,:,4].T.flatten()
	PEC_val_From_H = PEC_From_H[:,:,5].T.flatten()
	# n_H2p_HmH2p_N = 1.0000e+15	# #/m^3
	# T_H_HmH2p_N = 50000/11600	#K

	excited_states_From_H=len(PEC_From_H)
	excited_states_From_H = np.linspace(2,2+excited_states_From_H-1,excited_states_From_H).astype('int')
	excited_states_From_H = np.array(excited_states_From_H.tolist()*np.shape(PEC_From_H)[1])

	def From_H_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_H,excited_states=excited_states_From_H, coord_1=T_e_From_H, coord_2=T_H_From_H, coord_3=n_e_From_H, coord_4=n_Hp_From_H, coord_5=n_H_From_H):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_H = PEC_From_H[:, :, 0].T.flatten()
	T_H_From_H = PEC_From_H[:, :, 1].T.flatten()
	n_e_From_H = PEC_From_H[:, :, 2].T.flatten()
	n_Hp_From_H = PEC_From_H[:, :, 3].T.flatten()
	n_H_From_H = PEC_From_H[:, :, 4].T.flatten()
	PEC_val_From_H = PEC_From_H[:, :, 5].T.flatten()

	excited_states_From_H = len(PEC_From_H)
	excited_states_From_H = np.linspace(2, 2 + excited_states_From_H - 1, excited_states_From_H).astype('int')
	excited_states_From_H = np.array(excited_states_From_H.tolist() * np.shape(PEC_From_H)[1])

	T_H_From_H_selected = np.unique(T_H_From_H)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	selected = T_H_From_H==T_H_From_H_selected
	T_e_From_H = T_e_From_H[selected]
	n_e_From_H = n_e_From_H[selected]
	n_Hp_From_H = n_Hp_From_H[selected]
	n_H_From_H = n_H_From_H[selected]
	PEC_val_From_H = PEC_val_From_H[selected]
	excited_states_From_H = excited_states_From_H[selected]

	T_e_From_H_lowT = PEC_From_H_lowT[:, :, 0].T.flatten()
	T_H_From_H_lowT = 8000	# K
	n_e_From_H_lowT = PEC_From_H_lowT[:, :, 1].T.flatten()
	n_Hp_From_H_lowT = PEC_From_H_lowT[:, :, 2].T.flatten()
	n_H_From_H_lowT = PEC_From_H_lowT[:, :, 3].T.flatten()
	PEC_val_From_H_lowT = PEC_From_H_lowT[:, :, 4].T.flatten()

	excited_states_From_H_lowT = len(PEC_From_H_lowT)
	excited_states_From_H_lowT = np.linspace(2, 2 + excited_states_From_H_lowT - 1, excited_states_From_H_lowT).astype('int')
	excited_states_From_H_lowT = np.array(excited_states_From_H_lowT.tolist() * np.shape(PEC_From_H_lowT)[1])

	T_e_From_H = np.concatenate((T_e_From_H_lowT,T_e_From_H))
	n_e_From_H = np.concatenate((n_e_From_H_lowT,n_e_From_H))
	n_Hp_From_H = np.concatenate((n_Hp_From_H_lowT,n_Hp_From_H))
	n_H_From_H = np.concatenate((n_H_From_H_lowT,n_H_From_H))
	PEC_val_From_H = np.concatenate((PEC_val_From_H_lowT,PEC_val_From_H))
	excited_states_From_H = np.concatenate((excited_states_From_H_lowT,excited_states_From_H))

	def From_H_pop_coeff_full(x, requested_excited_states, parameters_orig=PEC_val_From_H,  excited_states=excited_states_From_H, coord_1=T_e_From_H, coord_2=n_e_From_H, coord_3=n_Hp_From_H, coord_4=n_H_From_H):
		import numpy as np
		x = np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines += excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1, coord_2, coord_3, coord_4])[:, selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2] > np.max(next_coord[0]):
					x[index][index_2] = np.max(next_coord[0])
				if x[index][index_2] < np.min(next_coord[0]):
					x[index][index_2] = np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0, temp >= 0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2] == up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('1')
				# print(up_bound)
				elif x[index][index_2] == low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
				# print('2')
				elif up_bound == low_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('3')
				else:
					alpha = np.log(x[index][index_2] / low_bound) / np.log(up_bound / low_bound)
					next_parameters = ((next_parameters[next_coord[0] == low_bound]) ** (1 - alpha)) * (
								(next_parameters[next_coord[0] == up_bound]) ** (alpha))
				# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_H_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_H,  excited_states=excited_states_From_H, coord_1=T_e_From_H, coord_2=n_e_From_H, coord_3=n_Hp_From_H, coord_4=n_H_From_H):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		coord_3 = np.unique(coord_3)
		coord_4 = np.unique(coord_4)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(coord_3),len(coord_4),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T

# color = ['b', 'r', 'm', 'y', 'g', 'c', 'k', 'slategrey', 'darkorange', 'lime', 'pink', 'gainsboro', 'paleturquoise']
# x1 = np.logspace(np.log10(0.1),np.log10(15),num=100)
# x1= np.array([x1,np.ones_like(x1)*np.unique(n_e_From_H)[int(len(np.unique(n_e_From_H))/2)],np.ones_like(x1)*np.unique(n_Hp_From_H)[int(len(np.unique(n_Hp_From_H))/2)],np.ones_like(x1)*np.unique(n_H_From_H)[int(len(np.unique(n_H_From_H))/2)]]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_H_pop_coeff_full(x1, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	select = np.logical_and(np.logical_and(np.logical_and(n_e_From_H==np.unique(n_e_From_H)[int(len(np.unique(n_e_From_H))/2)],n_Hp_From_H==np.unique(n_Hp_From_H)[int(len(np.unique(n_Hp_From_H))/2)]),n_H_From_H==np.unique(n_H_From_H)[int(len(np.unique(n_H_From_H))/2)]),excited_states_From_H==line)
# 	plt.plot(np.unique(T_e_From_H),PEC_val_From_H[select],color[i_line]+'+',label='line '+str(line))
# 	plt.plot(x1[:,0],coefficients[:,i_line],color[i_line])
# plt.semilogy()
# plt.xlabel('Te [eV]')
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)
#
# plt.figure()
# requested_excited_states = 2
# for i_line,line in enumerate(np.unique(n_e_From_H)[::3]):
# 	select = np.logical_and(np.logical_and(np.logical_and(n_e_From_H==line,n_Hp_From_H==np.unique(n_Hp_From_H)[int(len(np.unique(n_Hp_From_H))/2)]),n_H_From_H==np.unique(n_H_From_H)[int(len(np.unique(n_H_From_H))/2)]),excited_states_From_H==requested_excited_states)
# 	plt.plot(np.unique(T_e_From_H),PEC_val_From_H[select],color[i_line]+'+',label='ne '+str(line))
# 	x1 = np.logspace(np.log10(0.1),np.log10(15),num=100)
# 	x1= np.array([x1,np.ones_like(x1)*line,np.ones_like(x1)*np.unique(n_Hp_From_H)[int(len(np.unique(n_Hp_From_H))/2)],np.ones_like(x1)*np.unique(n_H_From_H)[int(len(np.unique(n_H_From_H))/2)]]).T
# 	coefficients = From_H_pop_coeff_full(x1, requested_excited_states)
# 	plt.plot(x1[:,0],coefficients[:],color[i_line])
# plt.semilogy()
# plt.xlabel('Te [eV]')
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)









# Three body recombination
# H+ + e- → H(p) + hν
# H+ + 2e- → H(p) + e-
# n = 2-13
if False:
	T_e_From_Hp = PEC_From_Hp[:,:,0].T.flatten()
	T_Hp_From_Hp = PEC_From_Hp[:,:,1].T.flatten()
	T_H_From_Hp = PEC_From_Hp[:,:,2].T.flatten()
	n_e_From_Hp = PEC_From_Hp[:,:,3].T.flatten()
	n_Hp_From_Hp = PEC_From_Hp[:,:,4].T.flatten()
	PEC_val_From_Hp = PEC_From_Hp[:,:,5].T.flatten()
	# n_Hp_HmHp_N = 1.0000e+15	# #/m^3
	# T_H_HmHp_N = 8000/11600	#K

	excited_states_From_Hp=len(PEC_From_Hp)
	excited_states_From_Hp = np.linspace(2,2+excited_states_From_Hp-1,excited_states_From_Hp).astype('int')
	excited_states_From_Hp = np.array(excited_states_From_Hp.tolist()*np.shape(PEC_From_Hp)[1])

	def From_Hp_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_Hp,excited_states=excited_states_From_Hp, coord_1=T_e_From_Hp, coord_2=T_Hp_From_Hp, coord_3=T_H_From_Hp, coord_4=n_e_From_Hp, coord_5=n_Hp_From_Hp):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_Hp = PEC_From_Hp[:, :, 0].T.flatten()
	T_Hp_From_Hp = PEC_From_Hp[:, :, 1].T.flatten()
	T_H_From_Hp = PEC_From_Hp[:, :, 2].T.flatten()
	n_e_From_Hp = PEC_From_Hp[:, :, 3].T.flatten()
	n_Hp_From_Hp = PEC_From_Hp[:, :, 4].T.flatten()
	PEC_val_From_Hp = PEC_From_Hp[:, :, 5].T.flatten()

	excited_states_From_Hp = len(PEC_From_Hp)
	excited_states_From_Hp = np.linspace(2, 2 + excited_states_From_Hp - 1, excited_states_From_Hp).astype('int')
	excited_states_From_Hp = np.array(excited_states_From_Hp.tolist() * np.shape(PEC_From_Hp)[1])

	T_Hp_From_Hp_selected = np.unique(T_Hp_From_Hp)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	T_H_From_Hp_selected = np.unique(T_H_From_Hp)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	n_Hp_From_Hp_selected = np.unique(n_Hp_From_Hp)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	selected = np.logical_and(np.logical_and(T_Hp_From_Hp==T_Hp_From_Hp_selected,T_H_From_Hp==T_H_From_Hp_selected),n_Hp_From_Hp==n_Hp_From_Hp_selected)
	T_e_From_Hp = T_e_From_Hp[selected]
	n_e_From_Hp = n_e_From_Hp[selected]
	PEC_val_From_Hp = PEC_val_From_Hp[selected]
	excited_states_From_Hp = excited_states_From_Hp[selected]

	T_e_From_Hp_lowT = PEC_From_Hp_lowT[:, :, 0].T.flatten()
	T_Hp_From_Hp_lowT = 8000	# K
	T_H_From_Hp_lowT = 8000	# K
	n_e_From_Hp_lowT = PEC_From_Hp_lowT[:, :, 1].T.flatten()
	n_Hp_From_Hp_lowT = 1e20	# # m^-3
	PEC_val_From_Hp_lowT = PEC_From_Hp_lowT[:, :, 2].T.flatten()

	excited_states_From_Hp_lowT = len(PEC_From_Hp_lowT)
	excited_states_From_Hp_lowT = np.linspace(2, 2 + excited_states_From_Hp_lowT - 1, excited_states_From_Hp_lowT).astype('int')
	excited_states_From_Hp_lowT = np.array(excited_states_From_Hp_lowT.tolist() * np.shape(PEC_From_Hp_lowT)[1])

	T_e_From_Hp = np.concatenate((T_e_From_Hp_lowT,T_e_From_Hp))
	n_e_From_Hp = np.concatenate((n_e_From_Hp_lowT,n_e_From_Hp))
	PEC_val_From_Hp = np.concatenate((PEC_val_From_Hp_lowT,PEC_val_From_Hp))
	excited_states_From_Hp = np.concatenate((excited_states_From_Hp_lowT,excited_states_From_Hp))

	def From_Hp_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_Hp,excited_states=excited_states_From_Hp, coord_1=T_e_From_Hp, coord_2=n_e_From_Hp):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_Hp_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_Hp,excited_states=excited_states_From_Hp, coord_1=T_e_From_Hp, coord_2=n_e_From_Hp):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T


# x = np.array([np.unique(T_e_From_Hp),np.ones_like(np.unique(T_e_From_Hp))*1e20]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_Hp_pop_coeff_full(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(T_e_From_Hp),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)





# Mutual neutralisation
# H+ + H- → H(p) + H
# n = 2-13
if False:
	T_e_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,0].T.flatten()
	T_Hp_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,1].T.flatten()
	T_Hm_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,2].T.flatten()
	T_H_From_Hn_with_Hp = 8000	# K
	n_e_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,3].T.flatten()
	n_Hp_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,4].T.flatten()
	n_Hm_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,5].T.flatten()
	PEC_val_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,6].T.flatten()
	# T_H_HmHp_N = 8000/11600	#K

	excited_states_From_Hn_with_Hp=len(PEC_From_Hn_with_Hp)
	excited_states_From_Hn_with_Hp = np.linspace(2,2+excited_states_From_Hn_with_Hp-1,excited_states_From_Hn_with_Hp).astype('int')
	excited_states_From_Hn_with_Hp = np.array(excited_states_From_Hn_with_Hp.tolist()*np.shape(PEC_From_Hn_with_Hp)[1])

	def From_Hn_with_Hp_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_Hn_with_Hp,excited_states=excited_states_From_Hn_with_Hp, coord_1=T_e_From_Hn_with_Hp, coord_2=T_Hp_From_Hn_with_Hp, coord_3=T_Hm_From_Hn_with_Hp, coord_4=n_e_From_Hn_with_Hp, coord_5=n_Hp_From_Hn_with_Hp, coord_6=n_Hm_From_Hn_with_Hp):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5,coord_6])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,0].T.flatten()
	T_Hp_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,1].T.flatten()
	T_Hm_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,2].T.flatten()
	T_H_From_Hn_with_Hp = 8000	# K
	n_e_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,3].T.flatten()
	n_Hp_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,4].T.flatten()
	n_Hm_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,5].T.flatten()
	PEC_val_From_Hn_with_Hp = PEC_From_Hn_with_Hp[:,:,6].T.flatten()
	# T_H_HmHp_N = 8000/11600	#K

	excited_states_From_Hn_with_Hp=len(PEC_From_Hn_with_Hp)
	excited_states_From_Hn_with_Hp = np.linspace(2,2+excited_states_From_Hn_with_Hp-1,excited_states_From_Hn_with_Hp).astype('int')
	excited_states_From_Hn_with_Hp = np.array(excited_states_From_Hn_with_Hp.tolist()*np.shape(PEC_From_Hn_with_Hp)[1])

	n_Hm_From_Hn_with_Hp_selected = np.unique(n_Hm_From_Hn_with_Hp)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	selected = n_Hm_From_Hn_with_Hp==n_Hm_From_Hn_with_Hp_selected
	T_e_From_Hn_with_Hp = T_e_From_Hn_with_Hp[selected]
	T_Hp_From_Hn_with_Hp = T_Hp_From_Hn_with_Hp[selected]
	T_Hm_From_Hn_with_Hp = T_Hm_From_Hn_with_Hp[selected]
	n_e_From_Hn_with_Hp = n_e_From_Hn_with_Hp[selected]
	n_Hp_From_Hn_with_Hp = n_Hp_From_Hn_with_Hp[selected]
	PEC_val_From_Hn_with_Hp = PEC_val_From_Hn_with_Hp[selected]
	excited_states_From_Hn_with_Hp = excited_states_From_Hn_with_Hp[selected]

	T_e_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:, :, 0].T.flatten()
	T_Hp_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:, :, 1].T.flatten()
	T_Hm_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:, :, 2].T.flatten()
	T_H_From_Hn_with_Hp_lowT = 8000	# K
	n_e_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:,:,3].T.flatten()
	n_Hp_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:,:,4].T.flatten()
	n_Hm_From_Hn_with_Hp_lowT = 1e15	# # m^-3
	PEC_val_From_Hn_with_Hp_lowT = PEC_From_Hn_with_Hp_lowT[:,:,5].T.flatten()

	excited_states_From_Hn_with_Hp_lowT = len(PEC_From_Hn_with_Hp_lowT)
	excited_states_From_Hn_with_Hp_lowT = np.linspace(2, 2 + excited_states_From_Hn_with_Hp_lowT - 1, excited_states_From_Hn_with_Hp_lowT).astype('int')
	excited_states_From_Hn_with_Hp_lowT = np.array(excited_states_From_Hn_with_Hp_lowT.tolist()*np.shape(PEC_From_Hn_with_Hp_lowT)[1])

	T_e_From_Hn_with_Hp = np.concatenate((T_e_From_Hn_with_Hp_lowT,T_e_From_Hn_with_Hp))
	T_Hp_From_Hn_with_Hp = np.concatenate((T_Hp_From_Hn_with_Hp_lowT,T_Hp_From_Hn_with_Hp))
	T_Hm_From_Hn_with_Hp = np.concatenate((T_Hm_From_Hn_with_Hp_lowT,T_Hm_From_Hn_with_Hp))
	n_e_From_Hn_with_Hp = np.concatenate((n_e_From_Hn_with_Hp_lowT,n_e_From_Hn_with_Hp))
	n_Hp_From_Hn_with_Hp = np.concatenate((n_Hp_From_Hn_with_Hp_lowT,n_Hp_From_Hn_with_Hp))
	PEC_val_From_Hn_with_Hp = np.concatenate((PEC_val_From_Hn_with_Hp_lowT,PEC_val_From_Hn_with_Hp))
	excited_states_From_Hn_with_Hp = np.concatenate((excited_states_From_Hn_with_Hp_lowT,excited_states_From_Hn_with_Hp))


	def From_Hn_with_Hp_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_Hn_with_Hp,excited_states=excited_states_From_Hn_with_Hp, coord_1=T_e_From_Hn_with_Hp, coord_2=T_Hp_From_Hn_with_Hp, coord_3=T_Hm_From_Hn_with_Hp, coord_4=n_e_From_Hn_with_Hp, coord_5=n_Hp_From_Hn_with_Hp):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_Hn_with_Hp_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_Hn_with_Hp,excited_states=excited_states_From_Hn_with_Hp, coord_1=T_e_From_Hn_with_Hp, coord_2=T_Hp_From_Hn_with_Hp, coord_3=T_Hm_From_Hn_with_Hp, coord_4=n_e_From_Hn_with_Hp, coord_5=n_Hp_From_Hn_with_Hp,T_mol_T_e_correlation=False):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		coord_3 = np.unique(coord_3)
		coord_4 = np.unique(coord_4)
		coord_5 = np.unique(coord_5)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(coord_3),len(coord_4),len(coord_5),len(excited_states)))

		number_of_points = len(x)
		if T_mol_T_e_correlation:
			print('gna')
		else:
			# x_full  = []
			# for excited_state in requested_excited_states:
			# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
			# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
			# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4),np.log(coord_5),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
			# return out.reshape((len(requested_excited_states),number_of_points)).T
			temp = []
			for excited_state in requested_excited_states:
				parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,:,:,:,(excited_states==excited_state).argmax()]
				temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4),np.log(coord_5)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
			return np.array(temp).T

# x = np.array([np.ones_like(np.unique(T_Hp_From_Hn_with_Hp))*0.2,np.unique(T_Hp_From_Hn_with_Hp),np.ones_like(np.unique(T_Hp_From_Hn_with_Hp)),np.ones_like(np.unique(T_Hp_From_Hn_with_Hp))*1e21,np.ones_like(np.unique(T_Hp_From_Hn_with_Hp))]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_Hn_with_Hp_pop_coeff_full_extra(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(T_Hp_From_Hn_with_Hp),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)






# H2+ mutual neutralisation
# H2+ + H- → H(p) + H2
# n = 2-13
if False:
	T_e_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,0].T.flatten()
	# Te_HmH2p_N = np.repeat(Te_HmH2p_N,len(excited_states)).reshape((len(Te_HmH2p_N),len(excited_states))).T.flatten()
	T_H2p_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,1].T.flatten()
	T_Hm_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,2].T.flatten()
	T_H_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,3].T.flatten()
	n_e_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,4].T.flatten()
	n_H2p_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,5].T.flatten()
	n_Hm_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,6].T.flatten()
	PEC_val_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:,:,7].T.flatten()
	# n_H2p_HmH2p_N = 1.0000e+15	# #/m^3
	# T_H_HmH2p_N = 50000/11600	#K

	excited_states_From_Hn_with_H2p=len(PEC_From_Hn_with_H2p)
	excited_states_From_Hn_with_H2p = np.linspace(2,2+excited_states_From_Hn_with_H2p-1,excited_states_From_Hn_with_H2p).astype('int')
	excited_states_From_Hn_with_H2p = np.array(excited_states_From_Hn_with_H2p.tolist()*np.shape(PEC_From_Hn_with_H2p)[1])

	def From_Hn_with_H2p_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_Hn_with_H2p,excited_states=excited_states_From_Hn_with_H2p, coord_1=T_e_From_Hn_with_H2p, coord_2=T_H2p_From_Hn_with_H2p, coord_3=T_Hm_From_Hn_with_H2p, coord_4=T_H_From_Hn_with_H2p, coord_5=n_e_From_Hn_with_H2p, coord_6=n_H2p_From_Hn_with_H2p, coord_7=n_Hm_From_Hn_with_H2p):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5,coord_6,coord_7])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 0].T.flatten()
	T_H2p_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 1].T.flatten()
	T_Hm_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 2].T.flatten()
	T_H_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 3].T.flatten()
	n_e_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 4].T.flatten()
	n_H2p_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 5].T.flatten()
	n_Hm_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 6].T.flatten()
	PEC_val_From_Hn_with_H2p = PEC_From_Hn_with_H2p[:, :, 7].T.flatten()

	excited_states_From_Hn_with_H2p = len(PEC_From_Hn_with_H2p)
	excited_states_From_Hn_with_H2p = np.linspace(2, 2 + excited_states_From_Hn_with_H2p - 1,excited_states_From_Hn_with_H2p).astype('int')
	excited_states_From_Hn_with_H2p = np.array(excited_states_From_Hn_with_H2p.tolist() * np.shape(PEC_From_Hn_with_H2p)[1])

	T_H_From_Hn_with_H2p_selected = np.unique(T_H_From_Hn_with_H2p)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	n_Hm_From_Hn_with_H2p_selected = np.unique(n_Hm_From_Hn_with_H2p)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	selected = np.logical_and(T_H_From_Hn_with_H2p==T_H_From_Hn_with_H2p_selected,n_Hm_From_Hn_with_H2p==n_Hm_From_Hn_with_H2p_selected)
	T_e_From_Hn_with_H2p = T_e_From_Hn_with_H2p[selected]
	T_H2p_From_Hn_with_H2p = T_H2p_From_Hn_with_H2p[selected]
	T_Hm_From_Hn_with_H2p = T_Hm_From_Hn_with_H2p[selected]
	n_e_From_Hn_with_H2p = n_e_From_Hn_with_H2p[selected]
	n_H2p_From_Hn_with_H2p = n_H2p_From_Hn_with_H2p[selected]
	PEC_val_From_Hn_with_H2p = PEC_val_From_Hn_with_H2p[selected]
	excited_states_From_Hn_with_H2p = excited_states_From_Hn_with_H2p[selected]

	T_e_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:, :, 0].T.flatten()
	T_H2p_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:, :, 1].T.flatten()
	T_Hm_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:, :, 2].T.flatten()
	T_H_From_Hn_with_H2p_lowT = 8000	# K
	n_e_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:,:,3].T.flatten()
	n_H2p_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:,:,4].T.flatten()
	n_Hm_From_Hn_with_H2p_lowT = 1e15	# # m^-3
	PEC_val_From_Hn_with_H2p_lowT = PEC_From_Hn_with_H2p_lowT[:,:,5].T.flatten()

	excited_states_From_Hn_with_H2p_lowT = len(PEC_From_Hn_with_H2p_lowT)
	excited_states_From_Hn_with_H2p_lowT = np.linspace(2, 2 + excited_states_From_Hn_with_H2p_lowT - 1, excited_states_From_Hn_with_H2p_lowT).astype('int')
	excited_states_From_Hn_with_H2p_lowT = np.array(excited_states_From_Hn_with_H2p_lowT.tolist()*np.shape(PEC_From_Hn_with_H2p_lowT)[1])

	T_e_From_Hn_with_H2p = np.concatenate((T_e_From_Hn_with_H2p_lowT,T_e_From_Hn_with_H2p))
	T_H2p_From_Hn_with_H2p = np.concatenate((T_H2p_From_Hn_with_H2p_lowT,T_H2p_From_Hn_with_H2p))
	T_Hm_From_Hn_with_H2p = np.concatenate((T_Hm_From_Hn_with_H2p_lowT,T_Hm_From_Hn_with_H2p))
	n_e_From_Hn_with_H2p = np.concatenate((n_e_From_Hn_with_H2p_lowT,n_e_From_Hn_with_H2p))
	n_H2p_From_Hn_with_H2p = np.concatenate((n_H2p_From_Hn_with_H2p_lowT,n_H2p_From_Hn_with_H2p))
	PEC_val_From_Hn_with_H2p = np.concatenate((PEC_val_From_Hn_with_H2p_lowT,PEC_val_From_Hn_with_H2p))
	excited_states_From_Hn_with_H2p = np.concatenate((excited_states_From_Hn_with_H2p_lowT,excited_states_From_Hn_with_H2p))


	def From_Hn_with_H2p_pop_coeff_full(x, requested_excited_states, parameters_orig=PEC_val_From_Hn_with_H2p,excited_states=excited_states_From_Hn_with_H2p, coord_1=T_e_From_Hn_with_H2p,coord_2=T_H2p_From_Hn_with_H2p, coord_3=T_Hm_From_Hn_with_H2p,coord_4=n_e_From_Hn_with_H2p, coord_5=n_H2p_From_Hn_with_H2p):
		import numpy as np
		x = np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines += excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1, coord_2, coord_3, coord_4, coord_5])[:, selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2] > np.max(next_coord[0]):
					x[index][index_2] = np.max(next_coord[0])
				if x[index][index_2] < np.min(next_coord[0]):
					x[index][index_2] = np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0, temp >= 0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2] == up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('1')
				# print(up_bound)
				elif x[index][index_2] == low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
				# print('2')
				elif up_bound == low_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('3')
				else:
					alpha = np.log(x[index][index_2] / low_bound) / np.log(up_bound / low_bound)
					next_parameters = ((next_parameters[next_coord[0] == low_bound]) ** (1 - alpha)) * (
								(next_parameters[next_coord[0] == up_bound]) ** (alpha))
				# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_Hn_with_H2p_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_Hn_with_H2p,excited_states=excited_states_From_Hn_with_H2p, coord_1=T_e_From_Hn_with_H2p,coord_2=T_H2p_From_Hn_with_H2p, coord_3=T_Hm_From_Hn_with_H2p,coord_4=n_e_From_Hn_with_H2p, coord_5=n_H2p_From_Hn_with_H2p):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		coord_3 = np.unique(coord_3)
		coord_4 = np.unique(coord_4)
		coord_5 = np.unique(coord_5)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(coord_3),len(coord_4),len(coord_5),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4),np.log(coord_5),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,:,:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(coord_3),np.log(coord_4),np.log(coord_5)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T

# x = np.array([np.unique(T_e_From_Hn_with_H2p),np.ones_like(np.unique(T_e_From_Hn_with_H2p)),np.ones_like(np.unique(T_e_From_Hn_with_H2p)),np.ones_like(np.unique(T_e_From_Hn_with_H2p))*1e21,np.ones_like(np.unique(T_e_From_Hn_with_H2p))]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_Hn_with_H2p_pop_coeff_full(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(T_e_From_Hn_with_H2p),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)





# H2 dissociation
# H2 + e- → H(p) + H(1) + e-
# n = 2-13
if False:
	T_e_From_H2 = PEC_From_H2[:,:,0].T.flatten()
	# Te_HmH2p_N = np.repeat(Te_HmH2p_N,len(excited_states)).reshape((len(Te_HmH2p_N),len(excited_states))).T.flatten()
	T_H_From_H2 = PEC_From_H2[:,:,1].T.flatten()
	n_e_From_H2 = PEC_From_H2[:,:,2].T.flatten()
	n_H2_From_H2 = PEC_From_H2[:,:,3].T.flatten()
	PEC_val_From_H2 = PEC_From_H2[:,:,4].T.flatten()
	# n_H2p_HmH2p_N = 1.0000e+15	# #/m^3
	# T_H_HmH2p_N = 50000/11600	#K

	excited_states_From_H2=len(PEC_From_H2)
	excited_states_From_H2 = np.linspace(2,2+excited_states_From_H2-1,excited_states_From_H2).astype('int')
	excited_states_From_H2 = np.array(excited_states_From_H2.tolist()*np.shape(PEC_From_H2)[1])

	def From_H2_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_H2,excited_states=excited_states_From_H2, coord_1=T_e_From_H2, coord_2=T_H_From_H2, coord_3=n_e_From_H2, coord_4=n_H2_From_H2):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_H2 = PEC_From_H2[:, :, 0].T.flatten()
	T_H_From_H2 = PEC_From_H2[:, :, 1].T.flatten()
	n_e_From_H2 = PEC_From_H2[:, :, 2].T.flatten()
	n_H2_From_H2 = PEC_From_H2[:, :, 3].T.flatten()
	PEC_val_From_H2 = PEC_From_H2[:, :, 4].T.flatten()

	excited_states_From_H2 = len(PEC_From_H2)
	excited_states_From_H2 = np.linspace(2, 2 + excited_states_From_H2 - 1, excited_states_From_H2).astype('int')
	excited_states_From_H2 = np.array(excited_states_From_H2.tolist() * np.shape(PEC_From_H2)[1])

	T_H_From_H2_selected = np.unique(T_H_From_H2)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	n_H2_From_H2_selected = np.unique(n_H2_From_H2)[0]	# pop coeff do not change with this, so it is removed from the interpolation
	selected = np.logical_and(T_H_From_H2==T_H_From_H2_selected,n_H2_From_H2==n_H2_From_H2_selected)
	T_e_From_H2 = T_e_From_H2[selected]
	n_e_From_H2 = n_e_From_H2[selected]
	PEC_val_From_H2 = PEC_val_From_H2[selected]
	excited_states_From_H2 = excited_states_From_H2[selected]

	T_e_From_H2_lowT = PEC_From_H2_lowT[:, :, 0].T.flatten()
	T_H_From_H2_lowT = 8000	# K
	n_e_From_H2_lowT = PEC_From_H2_lowT[:, :, 1].T.flatten()
	n_H2_From_H2_lowT = 1e19	# #/m^3
	PEC_val_From_H2_lowT = PEC_From_H2_lowT[:,:,2].T.flatten()

	excited_states_From_H2_lowT = len(PEC_From_H2_lowT)
	excited_states_From_H2_lowT = np.linspace(2, 2 + excited_states_From_H2_lowT - 1, excited_states_From_H2_lowT).astype('int')
	excited_states_From_H2_lowT = np.array(excited_states_From_H2_lowT.tolist()*np.shape(PEC_From_H2_lowT)[1])

	T_e_From_H2 = np.concatenate((T_e_From_H2_lowT,T_e_From_H2))
	n_e_From_H2 = np.concatenate((n_e_From_H2_lowT,n_e_From_H2))
	PEC_val_From_H2 = np.concatenate((PEC_val_From_H2_lowT,PEC_val_From_H2))
	excited_states_From_H2 = np.concatenate((excited_states_From_H2_lowT,excited_states_From_H2))

	def From_H2_pop_coeff_full(x, requested_excited_states, parameters_orig=PEC_val_From_H2,excited_states=excited_states_From_H2, coord_1=T_e_From_H2, coord_2=n_e_From_H2):
		import numpy as np
		x = np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines += excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1, coord_2])[:, selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2] > np.max(next_coord[0]):
					x[index][index_2] = np.max(next_coord[0])
				if x[index][index_2] < np.min(next_coord[0]):
					x[index][index_2] = np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0, temp >= 0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2] == up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('1')
				# print(up_bound)
				elif x[index][index_2] == low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
				# print('2')
				elif up_bound == low_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('3')
				else:
					alpha = np.log(x[index][index_2] / low_bound) / np.log(up_bound / low_bound)
					next_parameters = ((next_parameters[next_coord[0] == low_bound]) ** (1 - alpha)) * (
								(next_parameters[next_coord[0] == up_bound]) ** (alpha))
				# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_H2_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_H2,excited_states=excited_states_From_H2, coord_1=T_e_From_H2, coord_2=n_e_From_H2):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T

# x = np.array([np.ones_like(np.unique(n_e_From_H2))*0.2,np.unique(n_e_From_H2)]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_H2_pop_coeff_full(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(n_e_From_H2),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)



# H2+ recombination
# H2+ + e- → H(p) + H+ + e-
# n = 2-13
if False:
	T_e_From_H2p = PEC_From_H2p[:,:,0].T.flatten()
	# Te_HmH2p_N = np.repeat(Te_HmH2p_N,len(excited_states)).reshape((len(Te_HmH2p_N),len(excited_states))).T.flatten()
	T_H_From_H2p = PEC_From_H2p[:,:,1].T.flatten()
	n_e_From_H2p = PEC_From_H2p[:,:,2].T.flatten()
	n_H2p_From_H2p = PEC_From_H2p[:,:,3].T.flatten()
	PEC_val_From_H2p = PEC_From_H2p[:,:,4].T.flatten()
	# n_H2p_HmH2p_N = 1.0000e+15	# #/m^3
	# T_H_HmH2p_N = 50000/11600	#K

	excited_states_From_H2p=len(PEC_From_H2p)
	excited_states_From_H2p = np.linspace(2,2+excited_states_From_H2p-1,excited_states_From_H2p).astype('int')
	excited_states_From_H2p = np.array(excited_states_From_H2p.tolist()*np.shape(PEC_From_H2p)[1])

	def From_H2p_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_H2p,excited_states=excited_states_From_H2p, coord_1=T_e_From_H2p, coord_2=T_H_From_H2p, coord_3=n_e_From_H2p, coord_4=n_H2p_From_H2p):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_H2p = PEC_From_H2p[:,:,0].T.flatten()
	T_H_From_H2p = PEC_From_H2p[:,:,1].T.flatten()
	n_e_From_H2p = PEC_From_H2p[:,:,2].T.flatten()
	n_H2p_From_H2p = PEC_From_H2p[:,:,3].T.flatten()
	PEC_val_From_H2p = PEC_From_H2p[:,:,4].T.flatten()

	excited_states_From_H2p=len(PEC_From_H2p)
	excited_states_From_H2p = np.linspace(2,2+excited_states_From_H2p-1,excited_states_From_H2p).astype('int')
	excited_states_From_H2p = np.array(excited_states_From_H2p.tolist()*np.shape(PEC_From_H2p)[1])

	T_H_From_H2p_selected = np.unique(T_H_From_H2p)[0]  # pop coeff do not change with this, so it is removed from the interpolation
	n_H2p_From_H2p_selected = np.unique(n_H2p_From_H2p)[0]  # pop coeff do not change with this, so it is removed from the interpolation
	selected = np.logical_and(T_H_From_H2p == T_H_From_H2p_selected, n_H2p_From_H2p == n_H2p_From_H2p_selected)
	T_e_From_H2p = T_e_From_H2p[selected]
	n_e_From_H2p = n_e_From_H2p[selected]
	PEC_val_From_H2p = PEC_val_From_H2p[selected]
	excited_states_From_H2p = excited_states_From_H2p[selected]

	T_e_From_H2p_lowT = PEC_From_H2p_lowT[:, :, 0].T.flatten()
	T_H_From_H2p_lowT = 8000  # K
	n_e_From_H2p_lowT = PEC_From_H2p_lowT[:, :, 1].T.flatten()
	n_H2p_From_H2p_lowT = 1e19  # #/m^3
	PEC_val_From_H2p_lowT = PEC_From_H2p_lowT[:, :, 2].T.flatten()

	excited_states_From_H2p_lowT = len(PEC_From_H2p_lowT)
	excited_states_From_H2p_lowT = np.linspace(2, 2 + excited_states_From_H2p_lowT - 1, excited_states_From_H2p_lowT).astype('int')
	excited_states_From_H2p_lowT = np.array(excited_states_From_H2p_lowT.tolist() * np.shape(PEC_From_H2p_lowT)[1])

	T_e_From_H2p = np.concatenate((T_e_From_H2p_lowT, T_e_From_H2p))
	n_e_From_H2p = np.concatenate((n_e_From_H2p_lowT, n_e_From_H2p))
	PEC_val_From_H2p = np.concatenate((PEC_val_From_H2p_lowT, PEC_val_From_H2p))
	excited_states_From_H2p = np.concatenate((excited_states_From_H2p_lowT, excited_states_From_H2p))


	def From_H2p_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_H2p,excited_states=excited_states_From_H2p, coord_1=T_e_From_H2p, coord_2=n_e_From_H2p):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_H2p_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_H2p,excited_states=excited_states_From_H2p, coord_1=T_e_From_H2p, coord_2=n_e_From_H2p):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T


# x = np.array([np.ones_like(np.unique(n_e_From_H2p))*0.2,np.unique(n_e_From_H2p)]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_H2p_pop_coeff_full_extra(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(n_e_From_H2p),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)






# H3+ recombination
# H3+ + e- → H(p) + H2
# n = 2-13
if False:
	T_e_From_H3p = PEC_From_H3p[:,:,0].T.flatten()
	# Te_HmH3p_N = np.repeat(Te_HmH3p_N,len(excited_states)).reshape((len(Te_HmH3p_N),len(excited_states))).T.flatten()
	T_H3p_From_H3p = PEC_From_H3p[:,:,1].T.flatten()
	T_H_From_H3p = PEC_From_H3p[:,:,2].T.flatten()
	n_e_From_H3p = PEC_From_H3p[:,:,3].T.flatten()
	n_H3p_From_H3p = PEC_From_H3p[:,:,4].T.flatten()
	PEC_val_From_H3p = PEC_From_H3p[:,:,5].T.flatten()
	# n_H3p_HmH3p_N = 1.0000e+15	# #/m^3
	# T_H_HmH3p_N = 50000/11600	#K

	excited_states_From_H3p=len(PEC_From_H3p)
	excited_states_From_H3p = np.linspace(2,2+excited_states_From_H3p-1,excited_states_From_H3p).astype('int')
	excited_states_From_H3p = np.array(excited_states_From_H3p.tolist()*np.shape(PEC_From_H3p)[1])

	def From_H3p_pop_coeff_full(x,requested_excited_states,parameters_orig=PEC_val_From_H3p,excited_states=excited_states_From_H3p, coord_1=T_e_From_H3p, coord_2=T_H3p_From_H3p, coord_3=T_H_From_H3p, coord_4=n_e_From_H3p, coord_5=n_H3p_From_H3p):
		import numpy as np
		x=np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines+=excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1,coord_2,coord_3,coord_4,coord_5])[:,selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2]>np.max(next_coord[0]):
					x[index][index_2]=np.max(next_coord[0])
				if x[index][index_2]<np.min(next_coord[0]):
					x[index][index_2]=np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0,temp>=0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2]==up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
					# print('1')
					# print(up_bound)
				elif x[index][index_2]==low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
					# print('2')
				elif up_bound==low_bound:
					next_parameters = next_parameters[next_coord[0]==up_bound]
					# print('3')
				else:
					alpha = np.log(x[index][index_2]/low_bound)/np.log(up_bound/low_bound)
					next_parameters = ((next_parameters[next_coord[0]==low_bound])**(1-alpha)) * ((next_parameters[next_coord[0]==up_bound])**(alpha))
					# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)
else:
	T_e_From_H3p = PEC_From_H3p[:, :, 0].T.flatten()
	T_H3p_From_H3p = 2000	# K
	T_H_From_H3p = 8000	# K
	n_e_From_H3p = PEC_From_H3p[:, :, 1].T.flatten()
	n_H3p_From_H3p = 1e16	# # m^-3
	PEC_val_From_H3p = PEC_From_H3p[:, :, 2].T.flatten()

	excited_states_From_H3p = len(PEC_From_H3p)
	excited_states_From_H3p = np.linspace(2, 2 + excited_states_From_H3p - 1, excited_states_From_H3p).astype('int')
	excited_states_From_H3p = np.array(excited_states_From_H3p.tolist() * np.shape(PEC_From_H3p)[1])

	# T_H3p_From_H3p_selected = np.unique(T_H3p_From_H3p)[0]  # pop coeff do not change with this, so it is removed from the interpolation
	# T_H_From_H3p_selected = np.unique(T_H_From_H3p)[0]  # pop coeff do not change with this, so it is removed from the interpolation
	# n_H3p_From_H3p_selected = np.unique(n_H3p_From_H3p)[0]  # pop coeff do not change with this, so it is removed from the interpolation
	# selected = np.logical_and(np.logical_and(T_H3p_From_H3p == T_H3p_From_H3p_selected, T_H_From_H3p == T_H_From_H3p_selected),n_H3p_From_H3p==n_H3p_From_H3p_selected)
	# T_e_From_H3p = T_e_From_H3p[selected]
	# n_e_From_H3p = n_e_From_H3p[selected]
	# PEC_val_From_H3p = PEC_val_From_H3p[selected]
	# excited_states_From_H3p = excited_states_From_H3p[selected]

	T_e_From_H3p_lowT = PEC_From_H3p_lowT[:, :, 0].T.flatten()
	T_H3p_From_H3p_lowT = 2000  # K
	T_H_From_H3p_lowT = 8000  # K
	n_e_From_H3p_lowT = PEC_From_H3p_lowT[:, :, 1].T.flatten()
	n_H3p_From_H3p_lowT = 1e16  # #/m^3
	PEC_val_From_H3p_lowT = PEC_From_H3p_lowT[:, :, 2].T.flatten()

	excited_states_From_H3p_lowT = len(PEC_From_H3p_lowT)
	excited_states_From_H3p_lowT = np.linspace(2, 2 + excited_states_From_H3p_lowT - 1, excited_states_From_H3p_lowT).astype('int')
	excited_states_From_H3p_lowT = np.array(excited_states_From_H3p_lowT.tolist() * np.shape(PEC_From_H3p_lowT)[1])

	T_e_From_H3p = np.concatenate((T_e_From_H3p_lowT, T_e_From_H3p))
	n_e_From_H3p = np.concatenate((n_e_From_H3p_lowT, n_e_From_H3p))
	PEC_val_From_H3p = np.concatenate((PEC_val_From_H3p_lowT, PEC_val_From_H3p))
	excited_states_From_H3p = np.concatenate((excited_states_From_H3p_lowT, excited_states_From_H3p))


	def From_H3p_pop_coeff_full(x, requested_excited_states, parameters_orig=PEC_val_From_H3p,excited_states=excited_states_From_H3p, coord_1=T_e_From_H3p, coord_2=n_e_From_H3p):
		import numpy as np
		x = np.array(x)

		selected_lines = np.zeros_like(excited_states)
		for line in np.unique(requested_excited_states):
			selected_lines += excited_states == line
		selected_lines = selected_lines.astype('bool')

		coords = np.array([coord_1, coord_2])[:, selected_lines]
		parameters = parameters_orig[selected_lines]
		# if len(np.shape(x))>=2:
		interpolated_data = []
		for index in range(np.shape(x)[0]):
			next_coord = cp.deepcopy(coords)
			next_parameters = cp.deepcopy(parameters)
			for index_2 in range(np.shape(x)[1]):
				# print(index_2)
				# print(np.shape(next_coord))
				# print(np.shape(next_parameters))
				if x[index][index_2] > np.max(next_coord[0]):
					x[index][index_2] = np.max(next_coord[0])
				if x[index][index_2] < np.min(next_coord[0]):
					x[index][index_2] = np.min(next_coord[0])
				temp = next_coord[0] - x[index][index_2]
				up_bound = np.min(next_coord[0, temp >= 0])
				low_bound = np.max(next_coord[0, temp <= 0])
				if x[index][index_2] == up_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('1')
				# print(up_bound)
				elif x[index][index_2] == low_bound:
					next_parameters = next_parameters[next_coord[0] == low_bound]
				# print('2')
				elif up_bound == low_bound:
					next_parameters = next_parameters[next_coord[0] == up_bound]
				# print('3')
				else:
					alpha = np.log(x[index][index_2] / low_bound) / np.log(up_bound / low_bound)
					next_parameters = ((next_parameters[next_coord[0] == low_bound]) ** (1 - alpha)) * (
								(next_parameters[next_coord[0] == up_bound]) ** (alpha))
				# print('4')
				next_coord = next_coord[:, next_coord[0] == up_bound][1:]
			interpolated_data.append(next_parameters)
		return np.array(interpolated_data)

	def From_H3p_pop_coeff_full_extra(x, requested_excited_states, parameters_orig=PEC_val_From_H3p,excited_states=excited_states_From_H3p, coord_1=T_e_From_H3p, coord_2=n_e_From_H3p):
		coord_1 = np.unique(coord_1)
		coord_2 = np.unique(coord_2)
		excited_states = np.unique(excited_states)
		parameters_orig_simple = parameters_orig.reshape((len(coord_1),len(coord_2),len(excited_states)))

		number_of_points = len(x)
		# x_full  = []
		# for excited_state in requested_excited_states:
		# 	x_full.append(np.concatenate((x,np.array([excited_state*np.ones(len(x))]).T.tolist()),axis=1))
		# x_full = np.reshape(x_full,(len(x)*len(requested_excited_states),np.shape(x)[1]+1))
		# out = np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2),np.log(excited_states)),np.log(parameters_orig_simple),np.log(x_full), bounds_error=False,fill_value=None))
		# return out.reshape((len(requested_excited_states),number_of_points)).T
		temp = []
		for excited_state in requested_excited_states:
			parameters_orig_simple_single_exc_state = parameters_orig_simple[:,:,(excited_states==excited_state).argmax()]
			temp.append(np.exp(interpolate.interpn((np.log(coord_1),np.log(coord_2)),np.log(parameters_orig_simple_single_exc_state),np.log(x), bounds_error=False,fill_value=None)))
		return np.array(temp).T

# x = np.array([np.unique(T_e_From_H3p),np.ones_like(np.unique(T_e_From_H3p))*1e20]).T
# plt.figure()
# requested_excited_states = [2, 5, 8, 13]
# coefficients = From_H3p_pop_coeff_full(x, requested_excited_states)
# for i_line,line in enumerate(requested_excited_states):
# 	plt.plot(np.unique(T_e_From_H3p),coefficients[:,i_line],'+',label='line '+str(line))
# plt.semilogy()
# plt.semilogx()
# plt.legend(loc='best')
# plt.pause(0.01)




# I put here the reaction rates calculation
boltzmann_constant_J = 1.380649e-23	# J/K
eV_to_K = 8.617333262145e-5	# eV/K
avogadro_number = 6.02214076e23
hydrogen_mass = 1.008*1.660*1e-27	# kg
electron_mass = 9.10938356* 1e-31	# kg
gauss = lambda x, A, sig, x0: A * np.exp(-(((x - x0) / sig) ** 2)/2)
J_to_eV = 6.242e18


def RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	from scipy.special import hyp1f1
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
	# reaction rate		e +H(nl) → H- + hν, n ≥ 1
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.1.3
	# info only on e +H(1s) → H- + hν
	Eb = 0.754	# eV
	beta = np.float64(Eb/merge_Te_prof_multipulse_interp_crop_limited)	# converting to float64 I should have less issues with numerical cancellation
	# temp = 1.17*( 2* (np.pi**(1/2)) * (beta**(3/2)) * np.exp(beta)+1 -2*beta* hyp1f1(1,0.5,beta) )
	temp = np.float32(1.17*( 2*beta*( (np.pi**(1/2)) * (beta**(1/2)) * np.exp(beta) +1/(2*beta) -hyp1f1(1,0.5,beta)) ))
	if np.shape(temp)==():
		if np.isnan(temp):
			temp=0
		if temp<0:
			temp=0
	else:
		temp[temp<0] = 0
		temp[np.isnan(temp)] = 0
	# temp[np.logical_and(temp/(2*beta* hyp1f1(1,0.5,beta))<1e-5,beta>15)]=3/(2*beta[np.logical_and(temp/(2*beta* hyp1f1(1,0.5,beta))<1e-5,beta>15)])	# added to deal with numerical cancellation in subtraction
	# reaction_rate = 1.17*( 2* np.pi**(1/2) * beta**(3/2) * np.exp(beta)+1 -2*beta* hyp1f1(1,0.5,beta) )*(1e-10 * 1e-6* 1e20)		# m^3/s *1e20
	reaction_rate = temp*(1e-10 * 1e-6* 1e20)		# m^3/s *1e20
	# reaction_rate[np.isnan(reaction_rate)] = 0
	e_H__Hm = (merge_ne_prof_multipulse_interp_crop_limited**2) * reaction_rate
	return e_H__Hm	# m^-3/s *1e-20 / (nH(1)/ne)

def RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		# reaction rate		e +H- → e +H + e
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 3.1.1
		# as mentioned I neglect:
		# e +H- → e +H(n ≥ 2) + e
		# e +H- → e +H+ + 2e
		# because negligible at low temp
		# I'm not sure if only electron energy or also H-
		# I'll assume it is electron energy because usually it is the highest
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
	elif True:
		# data from https://www-amdis.iaea.org/ALADDIN/
		# I do this just because it's faster to calculate and it produces the same results, given I assume a maxwellian distribution
		interpolation_ev = np.array([0.126,0.16088,0.20542,0.26229,0.3349,0.42761,0.54599,0.69714,0.89014,1.1366,1.4512,1.853,2.3659,3.0209,3.8572,4.925,6.2885,8.0294,10.252,13.09,16.714,21.341,27.25,34.793,44.425,56.724,72.428,92.478,118.08,150.77,192.51,245.8,313.85,400.73,511.67,653.32,834.19,1065.1,1360,1736.5,2217.2,2831,3614.8,4615.5,5893.2,7524.6,9607.8,12268,15664,20000])
		interpolation_reaction_rate = np.array([8.553E-12,3.559E-11,1.1517E-10,3.1067E-10,7.3556E-10,1.5868E-09,3.2008E-09,6.1402E-09,1.1315E-08,2.0133E-08,3.464E-08,5.7584E-08,9.2294E-08,1.4225E-07,2.1026E-07,2.9746E-07,4.0221E-07,5.1956E-07,6.4157E-07,7.585E-07,8.6067E-07,9.4036E-07,9.931E-07,1.0181E-06,1.0178E-06,9.967E-07,9.6031E-07,9.1408E-07,8.6272E-07,8.0988E-07,7.5805E-07,7.0876E-07,6.6267E-07,6.1989E-07,5.801E-07,5.4278E-07,5.073E-07,4.7309E-07,4.397E-07,4.0685E-07,3.7451E-07,3.4285E-07,3.1221E-07,2.8305E-07,2.5587E-07,2.3113E-07,2.0915E-07,1.9015E-07,1.7413E-07,1.6093E-07]) * (1e-6 * 1e20)		# m^3/s * 1e20
		interpolator_reaction_rate = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_reaction_rate),fill_value='extrapolate')
		reaction_rate = np.exp(interpolator_reaction_rate(np.log(merge_Te_prof_multipulse_interp_crop_limited)))
		if not np.shape(np.array(merge_Te_prof_multipulse_interp_crop_limited))==():
			reaction_rate[reaction_rate<0] = 0
	e_Hm__e_H_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return e_Hm__e_H_e	# m^-3/s *1e-20 / (nHm/ne)


def RR_Hp_Hm__H_2_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp=np.array([T_Hp])
		T_Hm=np.array([T_Hm])
	# reaction rate		H+ +H- →H(2) +H(1s)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 3.2.1
	# I'm not sure if only H+ energy or also H-
	# I'll assume it is H+ energy because usually it is the highest
	def internal(*arg):
		T_Hp,T_Hm = arg[0]
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		# Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
		# Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		# Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
		# Hp_energy = np.array(([Hp_energy.tolist()])*9)
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
		cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* (1e-4 * 1e20)	# multiplied 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
		cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
		cross_section[baricenter_impact_energy<0.1]=cross_section_min
		cross_section[baricenter_impact_energy>20000]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s * 1e20
		# reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_Hp.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_Hp))[0]
	Hp_Hm__H_2_H = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return Hp_Hm__H_2_H	# m^-3/s *1e-20 / (nHm/ne) / (nHp/ne)

def RR_Hp_Hm__H_3_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp=np.array([T_Hp])
		T_Hm=np.array([T_Hm])
	# reaction rate		H+ +H- →H(2) +H(1s)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 3.2.1
	# I'm not sure if only H+ energy or also H-
	# I'll assume it is H+ energy because usually it is the highest
	def internal(*arg):
		T_Hp,T_Hm = arg[0]
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		# Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
		# Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		# Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
		# Hp_energy = np.array(([Hp_energy.tolist()])*9)
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
		cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* (1e-4 * 1e20)	# multiplied 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(Hp_energy[0]-0.1)).argmin()]
		cross_section_max = cross_section.flatten()[(np.abs(Hp_energy[0]-20000)).argmin()]
		cross_section[baricenter_impact_energy<0.1]=cross_section_min
		cross_section[baricenter_impact_energy>20000]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
		# reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s * 1e20
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_Hp.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_Hp))[0]
	Hp_Hm__H_3_H = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return Hp_Hm__H_3_H	# m^-3/s *1e-20 / (nHm/ne) / (nHp/ne)

def RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		nHp_ne_all = np.array([nHp_ne_all])
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])

	# H+ + H- → H(v) + H
	# Yacora, H+ Mutual neutralization
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if len(np.shape(merge_Te_prof_multipulse_interp_crop_limited))==4:	# H, Hm, H2, H2p, ne, Te
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,:,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,:,0]
		T_Hp_temp = T_Hp[0,:,0]
		T_Hm_temp = T_Hm[0,:,0]
		nHp_ne_all_temp = nHp_ne_all[0,:,0]
		population_coefficients = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te.flatten(),T_Hp_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nHp_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		reaction_rate = (ne.flatten()**2)*1e20*np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1)	# 	#/s * 1e-20
		reaction_rate = reaction_rate.reshape(np.shape(T_Hp[0,:,0]))
		reaction_rate = np.transpose([[reaction_rate.tolist()]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[2]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[0] ,(0,2,1,3))
	else:
		population_coefficients = From_Hn_with_Hp_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_Hp.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nHp_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited.flatten()**2)*1e20*(np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1).reshape(np.shape(merge_Te_prof_multipulse_interp_crop_limited)))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nHm/ne) / (nHp/ne)

def RR_Hp_Hm__Hn_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		nHp_ne_all = np.array([nHp_ne_all])
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])

	# H+ + H- → H(v) + H
	# Yacora, H+ Mutual neutralization
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if len(np.shape(merge_Te_prof_multipulse_interp_crop_limited))==4:	# H, Hm, H2, H2p, ne, Te
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,:,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,:,0]
		T_Hp_temp = T_Hp[0,:,0]
		T_Hm_temp = T_Hm[0,:,0]
		nHp_ne_all_temp = nHp_ne_all[0,:,0]
		population_coefficients = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te.flatten(),T_Hp_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nHp_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		reaction_rate = (ne.flatten()**2)*1e20*(population_coefficients*einstein_coeff_full_cumulative).T	# 	#/s * 1e-20
		reaction_rate = reaction_rate.reshape((len(einstein_coeff_full_cumulative),*np.shape(T_Hp[0,:,0])))
		reaction_rate = np.transpose([[reaction_rate.tolist()]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[2]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[0] ,(2,0,3,1,4))
	else:
		population_coefficients = From_Hn_with_Hp_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_Hp.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nHp_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited**2)*1e20*np.transpose((population_coefficients*einstein_coeff_full_cumulative).reshape((*np.shape(merge_Te_prof_multipulse_interp_crop_limited),len(einstein_coeff_full_cumulative))),(-1,*np.arange(len(merge_Te_prof_multipulse_interp_crop_limited.shape))))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nHm/ne) / (nHp/ne)

def RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		nH2p_ne_all = np.array([nHp_ne_all])
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		T_H2p = np.array([T_Hm])

	# H2+ + H- → H(v) + H2
	# Yacora, H2+ Mutual neutralization
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if len(np.shape(merge_Te_prof_multipulse_interp_crop_limited))==4:	# H, Hm, H2, H2p, ne, Te
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,0,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,0,0]
		T_Hp_temp = T_Hp[0,0,0]
		T_Hm_temp = T_Hm[0,0,0]
		T_H2p_temp = T_H2p[0,0,0]
		nH2p_ne_all_temp = nH2p_ne_all[0,0,0]
		population_coefficients = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te.flatten(),T_H2p_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nH2p_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		reaction_rate = (ne.flatten()**2)*1e20*np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1)	# 	#/s * 1e-20
		reaction_rate = np.array([[[reaction_rate.tolist()]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[2]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[1]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[0])
	else:
		population_coefficients = From_Hn_with_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_H2p.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited.flatten()**2)*1e20*(np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1).reshape(np.shape(merge_Te_prof_multipulse_interp_crop_limited)))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nHm/ne)

def RR_H2p_Hm__Hn_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		nH2p_ne_all = np.array([nHp_ne_all])
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		T_H2p = np.array([T_Hm])

	# H2+ + H- → H(v) + H2
	# Yacora, H2+ Mutual neutralization
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if len(np.shape(merge_Te_prof_multipulse_interp_crop_limited))==4:	# H, Hm, H2, H2p, ne, Te
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,0,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,0,0]
		T_Hp_temp = T_Hp[0,0,0]
		T_Hm_temp = T_Hm[0,0,0]
		T_H2p_temp = T_H2p[0,0,0]
		nH2p_ne_all_temp = nH2p_ne_all[0,0,0]
		population_coefficients = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te.flatten(),T_H2p_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nH2p_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		reaction_rate = (ne.flatten()**2)*1e20*(population_coefficients*einstein_coeff_full_cumulative).T	# 	#/s * 1e-20
		reaction_rate = np.transpose([[[reaction_rate.tolist()]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[2]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[1]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[0], (3,0,1,2,4))
	else:
		population_coefficients = From_Hn_with_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_H2p.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited.flatten()**2)*1e20*(np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1).reshape(np.shape(merge_Te_prof_multipulse_interp_crop_limited)))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nHm/ne)

def RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp=np.array([T_Hp])
		T_Hm=np.array([T_Hm])
	# reaction rate		H+ +H- → H2+(v) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 3.2.2
	# valid from at least 1e-3eV
	def internal(*arg):
		T_Hp,T_Hm = arg[0]
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
		Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
		Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
		Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
		cross_section = (1.38 / ((baricenter_impact_energy**0.85) * (1+0.065 * baricenter_impact_energy**2.7))) * (1e-16 *1e-4* 1e20)	# multiplied 1e20
		cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy-0.001)).argmin()]
		cross_section[baricenter_impact_energy<0.001]=cross_section_min
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*Hp_velocity*Hp_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(Hp_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s * 1e20
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_Hp.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_Hp))[0]
	Hp_Hm__H2p_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return Hp_Hm__H2p_e	# m^-3/s *1e-20 / (nHm/ne) / (nHp/ne)

def RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H=np.array([T_H])
		T_Hm=np.array([T_Hm])
	# reaction rate		H- +H(1s) →H2-(B2Σ+g ) →H(1s) +H(1s) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 3.3.2
	def internal(*arg):
		T_H,T_Hm = arg[0]
		T_H = np.array([T_H])
		T_Hm = np.array([T_Hm])
		H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/hydrogen_mass)**0.5).T
		H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
		Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
		Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hm),200))+H_energy).T + Hm_energy).T
		# baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
		baricenter_impact_energy[baricenter_impact_energy<0.1]=0.1
		baricenter_impact_energy[baricenter_impact_energy>20000]=20000
		coefficients = np.array([-3.61799082*1e+1, 1.16615172, -1.41928602*1e-1, -1.11195959*1e-2, -1.72505995*1e-3, 1.59040356*1e-3, -2.53196144*1e-4, 1.66978235*1e-5, -4.09725797*1e-7])
		# cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
		cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* (1e-4* 1e20)	# multiplied 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		# cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-0.1)).argmin()]	# I dont think that this is necessary because the polinomial fit seems behaving well (going to ~0 quite rapidly) for decreasing energy
		cross_section_max = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-20000)).argmin()]	# it becomes flattish for high energies, so this is not a big mistake
		# cross_section[baricenter_impact_energy<0.1]=cross_section_min
		cross_section[baricenter_impact_energy>20000]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s * 1e20
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H))[0]
	Hm_H1s__H1s_H1s_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return Hm_H1s__H1s_H1s_e	# m^-3/s *1e-20 / (nHm/ne) / (nH(1)/ne)


def RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H=np.array([T_H])
		T_Hm=np.array([T_Hm])
	# H- +H(1s) →H2-(X2Σ+u ; B2Σ+g ) →H2(X1Σ+ g ; v) + e
	# reaction rate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 3.3.2
	def internal(*arg):
		T_H,T_Hm = arg[0]
		T_H = np.array([T_H])
		T_Hm = np.array([T_Hm])
		H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/hydrogen_mass)**0.5).T
		H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
		Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
		Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hm),200))+H_energy).T + Hm_energy).T
		# baricenter_impact_energy = np.array(([baricenter_impact_energy.tolist()])*9)
		baricenter_impact_energy[baricenter_impact_energy<0.1]=0.1
		baricenter_impact_energy[baricenter_impact_energy>20000]=20000
		coefficients = np.array([-3.44152907*1e+1, -3.39348209*1e-1, 5.66591705*1e-2, -9.05150459*1e-3, 7.66060418*1e-4, -4.27126462*1e-5, -1.57273749*1e-7, 2.57607677*1e-7, -1.20071919*1e-8])
		# cross_section = (np.exp(np.sum(((np.log(baricenter_impact_energy.T))**(np.arange(9)))*coefficients ,axis=-1)) * 1e-4).T
		cross_section = np.exp(np.polyval(np.flip(coefficients,axis=0),np.log(baricenter_impact_energy)))* (1e-4* 1e20)	# multiplied 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-0.01)).argmin()]	# instead of 0.1 as limit I use 0.01, because I need to look at low energies, but still the fit starts to misbehave for too low energies
		cross_section_max = cross_section.flatten()[(np.abs(baricenter_impact_energy[0]-20000)).argmin()]	# this can be used because the cross section is quite flat
		cross_section[baricenter_impact_energy<0.01]=cross_section_min
		cross_section[baricenter_impact_energy>20000]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s * 1e20
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H))[0]
	Hm_H1s__H2_v_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return Hm_H1s__H2_v_e	# m^-3/s *1e-20 / (nHm/ne) / (nH(1)/ne)


def RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H=np.array([T_H])
		T_Hp=np.array([T_Hp])
	# reaction rate		H+ +H(1s) +H(1s) →H+ +H2(v)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.2.4, equation 46a
	# valid for temperature up to 30000K (~2.587eV)
	# I'm not sure which temperature to use here. I take a mean of T_H+, T_H, T_H
	temperature = np.mean(np.array([T_Hp,T_H,T_H]),axis=0)
	reaction_rate = 1.145/(( temperature )**1.12) *1e-29 * (1e-6 * 1e-6* 1e40)	# m^6/s * 1e20
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))] = 0
	reaction_rate_max = reaction_rate.flatten()[(np.abs(temperature-30000)).argmin()]
	reaction_rate[temperature>30000]=reaction_rate_max
	Hp_H_H__Hp_H2 = (merge_ne_prof_multipulse_interp_crop_limited**3)*reaction_rate
	return Hp_H_H__Hp_H2	# m^-3/s *1e-20 / (nHp/ne) / (nH(1)/ne) / (nH(1)/ne)


def RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H=np.array([T_H])
	# reaction rate		H +H +H →H +H2(v)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.3.4
	# valid for temperature up to 20000-30000K (~2eV)
	reaction_rate = 1.65/T_H *(1e-30 * 1e-6 * 1e-6*1e40)		# m^6/s * 1e40
	reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H-20000)).argmin()]
	reaction_rate[T_H>20000]=reaction_rate_max
	H_H_H__H_H2 = ((merge_ne_prof_multipulse_interp_crop_limited)**3)*reaction_rate
	return H_H_H__H_H2	# m^-3/s *1e-20 / (nH/ne)


def RR_e_H2v__e_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	# NOTE: I stop using this one and I use the one from Yacora instead
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		# reaction rate		e+H2(X1Σg;v) → e+H2∗(N1,3Λσ;eps) → e+H(1s)+H(nl)
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# valid for energy up to ~1000eV
		# Vibrational partition function
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
			cross_section = 5.984/e_energy*((1-1/x)**coefficients[1]) * coefficients[2] * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section = cross_section * coefficients[2] /100
			cross_section[cross_section<0] = 0
			reaction_rate += np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s * 1e20
		e_H2_X1Σg_x__e_H_H = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
		e_H2_X1Σg_x__e_H_H[0] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
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
				shape_function = (coefficients1[0]/x) * ( (1-1/x)**coefficients1[1] ) * ( coefficients1[2] + coefficients1[3]/x + np.log(x) ) * (1e-16 * 1e4 * 1e20)	# m^2 * 1e20
				cross_section = shape_function * ((13.6/coefficients2)**3)	# m^2
				cross_section[np.logical_not(np.isfinite(cross_section))]=0
				cross_section = cross_section * coefficients3 /100
				cross_section[cross_section<0] = 0
				reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^6 / s
				e_H2_X1Σg_x__e_H_H[v_index+1] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index+1]
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
				shape_function = (coefficients1[0]/x) * ( (1-1/x)**coefficients1[1] ) * ( coefficients1[2] + coefficients1[3]/x + np.log(x) ) * (1e-16 * 1e4 * 1e20)	# m^2 * 1e20
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
					e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
				else:
					e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]

		# DISSOCIATION VIA TRIPLETS

		# VIBRATIONAL STATE = 0
		# chapter 4.2.2a
		# through b3Σ+ and e3Σ+
		# coefficients: A,β,γ,∆E(eV ), % dissociation vs excitation
		coefficients_all = np.array([[11.16,2.33,3.78,7.93,100],[0.190,4.5,1.60,13.0,20]])
		for coefficients in coefficients_all:
			x = e_energy/coefficients[3]
			cross_section = coefficients[0]/(x**3) * ((1-1/(x**coefficients[1]))**coefficients[2]) * (1e-16 * 1e-4*1e20)	# m^2 * 1e20
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section = cross_section * coefficients[4] /100
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s * 1e20
			e_H2_X1Σg_x__e_H_H[0] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
		# VIBRATIONAL STATE 0 - 10
		# chapter 4.2.2b
		# through b3Σu, a3Σg, c3Πu only to dissociation
		# coefficients: b1,...,b6
		# valid 1000 to 200000K
		coefficients_all = np.array([[-11.565,-7.6012e-2,-78.433,0.7496,-2.2126,0.22006],[-12.035,-6.6082e-2,-67.806,0.72403,-1.5419,1.5195],[-13.566,-4.3737e-2,-55.933,0.72286,-2.3103,1.5844],[-46.664,0.74122,-15.297,-2.2384e-2,-1.3674,1.3621],[-37.463,0.81763,-0.40373,-0.45851,-18.093,1.1460e-2],[-28.283,0.99053,-10.377,-8.5590e-2,-11.053,6.7271e-2],[-23.724,1.0112,-2.9905,-0.24701,-17.931,3.4376e-2],[-19.547,1.0224,-1.7489,-0.31413,-19.408,2.8643e-2],[-15.937,1.0213,-10175,-0.3804,-20.24,2.4170e-2],[-12.712,1.0212,-0.604,-0.44572,-20.766,2.1159e-2],[-0.40557,-0.49721,-9.9025,1.0212,-21.031,1.9383e-2]])
		for v_index,coefficients in enumerate(coefficients_all):
			T_scaled = T_e_temp/1000
			reaction_rate = np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**coefficients[5]) ) * (1e-6 * 1e20)		# m^3 / s * 1e20
			reaction_rate_min = reaction_rate.flatten()[(np.abs(T_e_temp-1000)).argmin()]
			reaction_rate[T_e_temp<1000]=reaction_rate_min
			if v_index==0:
				e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
			else:
				e_H2_X1Σg_x__e_H_H[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		e_H2_X1Σg_x__e_H_H = np.sum(e_H2_X1Σg_x__e_H_H,axis=0)
		return e_H2_X1Σg_x__e_H_H
	elif True:
		# from AMJUEL
		# 4.10 ReactionReaction 2.2.5g e +H2 → e +H +H
		# this RR is computed for TH2=0.1eV, THp=Te, H2(v) in vibrational equilibrioum with Te
		coefficients_all = [[-2.702372540584e+01,-3.152103191633e-03,5.990692171729e-03,-3.151252835426e-03,7.457309144890e-04,-9.238664007853e-05,6.222557542845e-06,-2.160024578659e-07,3.028755759836e-09],[1.081756417479e+01,-1.487216964825e-02,1.417396532101e-02,-4.689911797083e-03,7.180338663163e-04,-5.502798587526e-05,1.983066081752e-06,-2.207639762507e-08,-2.116339335271e-10],[-5.368872027676e+00,5.419787589654e-03,-1.747268613395e-02,9.532963297450e-03,-2.196705622859e-03,2.611447288152e-04,-1.695536960581e-05,5.737375510694e-07,-7.940900078995e-09],[1.340684229143e+00,1.058157580038e-02,-3.446019122786e-03,-7.032769815599e-04,4.427959286553e-04,-7.370484189164e-05,5.746786010618e-06,-2.182085196303e-07,3.264045809897e-09],[-1.561644923145e-01,-3.847438570333e-03,3.571477356851e-03,-1.103305795473e-03,1.476712517858e-04,-8.461162952132e-06,9.757111870171e-08,8.130014050833e-09,-2.234996157750e-10],[-1.444731533894e-04,-3.194532513126e-04,-2.987368098475e-04,2.092094838648e-04,-4.339352509941e-05,4.009328699469e-06,-1.762651912129e-07,3.357860444624e-09,-1.857322587267e-11],[2.117693926546e-03,2.679309814780e-04,-1.037559373832e-04,7.297053580368e-06,1.454171585421e-06,-2.251616910293e-07,9.191700327811e-09,-2.052366968228e-11,-3.567738654108e-12],[-2.143738340207e-04,-3.539232757385e-05,1.909399233821e-05,-3.819368125069e-06,3.754063159414e-07,-2.441872829462e-08,1.437490161488e-09,-6.172308568891e-11,1.104905484620e-12],[6.979740947331e-06,1.462031952352e-06,-8.858634506391e-07,2.099830142707e-07,-2.606862169776e-08,2.039813579349e-09,-1.113483084607e-10,3.859777100010e-12,-5.909099891913e-14]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6*1e20)	# m^3/s * 1e20
		if not np.shape(np.array(merge_Te_prof_multipulse_interp_crop_limited))==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
			# I have to add external bouds to avoid the extrapolation to stop working and give me crazy values
			reaction_rate[merge_Te_prof_multipulse_interp_crop_limited<0.1]=0
		else:
			if merge_Te_prof_multipulse_interp_crop_limited<0.1:
				reaction_rate = 0
		e_H2v__e_H_H = reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2)
		return e_H2v__e_H_H	# m^-3/s *1e-20 / (nH2/ne)

def RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	# using this plus the one from Janev that has only the ground state gives me the whole picture
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])

	# H2 + e- → H(v) + H(g) + e-
	# Yacora, Dissociation of H2
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		population_coefficients = From_H2_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2))
		reaction_rate = (ne**2)*1e20*np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1)	# 	#/s * 1e-20
	else:
		population_coefficients = From_H2_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited.flatten()**2)*1e20*(np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1).reshape(np.shape(merge_Te_prof_multipulse_interp_crop_limited)))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nH2/ne)

def RR_e_H2__e_Hn_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	# using this plus the one from Janev that has only the ground state gives me the whole picture
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])

	# H2 + e- → H(v) + H(g) + e-
	# Yacora, Dissociation of H2
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		population_coefficients = From_H2_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2))
		reaction_rate = (ne**2)*1e20*(population_coefficients*einstein_coeff_full_cumulative)	# 	#/s * 1e-20
	else:
		population_coefficients = From_H2_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited**2)*1e20*np.transpose((population_coefficients*einstein_coeff_full_cumulative).reshape((*np.shape(merge_Te_prof_multipulse_interp_crop_limited),len(einstein_coeff_full_cumulative))),(-1,*np.arange(len(merge_Te_prof_multipulse_interp_crop_limited.shape))))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nH2/ne)


def RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		# reaction rate dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; v’) +e
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 4.3.1
		# I neglect what happens for electronic excited states because they will anyway be less populated
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
			cross_section = ((coefficients/coefficients_all[0])**1.15) * 1.828/x * ((1-1/(x**0.92))**2.19) * np.log(2.05*e_energy) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
			if v_index==0:
				e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
			else:
				e_H2N13Λσ__e_H2pX2Σg_e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		e_H2N13Λσ__e_H2pX2Σg_e = np.sum(e_H2N13Λσ__e_H2pX2Σg_e,axis=0)
	elif True:
		# e +H2 → 2e +H2+
		# from AMJUEL
		# 4.11 Reaction 2.2.9
		# I use this because the to calculate the cross section i use standard H2 vibrational states already in AMJUL, but this method is much faster
		coefficients_all = [[-35.74773783577,0.3470247049909,-0.09683166540937,0.00195957627625,0.00247936111919,-0.0001196632952666,-0.00001862956119592,0.000001669867158509,-0.000000036737362782],[17.69208985507,-1.311169841222,0.4700486215943,-0.05521175478827,-0.002689651616933,0.0007308915874002,-0.00002920560755694,-0.0000003148831240316,0.00000002514856386324],[-8.291764008409,1.591701525694,-0.5814996025336,0.09160898084105,-0.004770789631868,0.00001994775632224,-0.000007511552245648,0.000001089689676313,-0.00000002920863498031],[2.55571234724,-0.8625268584825,0.2612076696684,-0.03686525285376,0.001945480608139,-0.00003690918356665,0.000004836340453567,-0.0000004165748666929,0.000000009265898224345],[-0.5370404654062,0.2375816996323,-0.0416590877817,0.001732469114063,0.0003693513203529,-0.00004931268184607,0.000002727501534044,-0.0000001081027384449,0.000000002420509440644],[0.07443307905391,-0.03322214182214,-0.002351235556666,0.001723053881691,-0.0002096625925098,0.00001358575558294,-0.000001041586202167,0.00000006928574330531,-0.000000001746656185835],[-0.006391785721973,0.00186255427819,0.001540632467396,-0.0003547150770477,0.00001392157055273,0.000001047463944093,0.00000001513510667993,-0.000000009915499708242,0.0000000003298173891188],[0.0003001729098239,0.00003497202259366,-0.0001742029226138,0.00002296551698214,0.000002357520372192,-0.000000530608551395,0.00000002223137028418,0.00000000033401693098,-0.00000000002560542889504],[-0.000005607182991432,-0.000005779550092391,0.000006495742927455,-0.0000003040011333889,-0.0000002361542565281,0.00000003655056080262,-0.000000001771478792301,0.00000000001334615260635,0.0000000000006831564719957]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6*1e20)	# m^3/s * 1e20
		if not np.shape(np.array(merge_Te_prof_multipulse_interp_crop_limited))==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
			# I have to add external bouds to avoid the extrapolation to stop working and give me crazy values
			reaction_rate[merge_Te_prof_multipulse_interp_crop_limited<0.5]=0
		else:
			if merge_Te_prof_multipulse_interp_crop_limited<0.5:
				reaction_rate = 0
		e_H2__e_H2p_e = reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2)

	return e_H2__e_H2p_e	# m^-3/s *1e-20 / (nH2/ne)
	# checked up to here


def RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		# reaction rate
		# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(X2Σg+; eps’) +e → H+ +H(1s) + 2e
		# dissociative: e +H2 (N1,3Λσ; v) → e +H2+(B2Σu+; eps’) +e → H+ +H(1s) + 2e
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 4.3.1
		# I neglect what happens for electronic excited states because they will anyway be less populated
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
			reaction_rate = np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2] + coefficients[3]*np.exp(-coefficients[4]*T_scaled)) * (1e-6 * 1e20) 		# m^3 / s * 1e20
			reaction_rate_min = reaction_rate.flatten()[(np.abs(T_e_temp-3000)).argmin()]
			reaction_rate_max = reaction_rate.flatten()[(np.abs(T_e_temp-200000)).argmin()]
			reaction_rate[T_e_temp<3000]=reaction_rate_min
			reaction_rate[T_e_temp>200000]=reaction_rate_max
			if v_index==0:
				e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
			else:
				e_H2N13Λσ__Hp_H_2e[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		e_H2N13Λσ__Hp_H_2e = np.sum(e_H2N13Λσ__Hp_H_2e,axis=0)
	elif True:
		# e +H2 → 2e +H +H+
		# from AMJUEL
		# 4.12 Reaction 2.2.10
		# I use this because the to calculate the cross section i use standard H2 vibrational states already in AMJUL, but this method is much faster
		coefficients_all = [[-37.93749300315,-0.3333162972531,0.1849601203843,-0.08803945197107,0.02205180180735,-0.002852568161901,0.0001942314738448,-0.000006597388255594,0.00000008798544848606],[12.80249398154,1.028969438485,-0.3271855492638,0.1305597441611,-0.0340843982191,0.004591924060066,-0.0003167471002157,0.00001070920193931,-0.0000001408139742113],[-3.77814855314,-1.415561059533,0.2928509524911,-0.07425165688158,0.02028424685287,-0.003042376564749,0.0002279124955373,-0.000008197224564797,0.0000001130682076163],[0.2499987501522,1.032922656537,-0.1580288004759,0.009934702707539,-0.002450845732158,0.0005716646876513,-0.00005339115778704,0.000002135848413694,-0.00000003072223247387],[0.2480574522949,-0.4372934216955,0.06448433196301,0.00122922293263,-0.0009281410519553,0.00005946235618034,-0.00000008758032156912,-0.00000007270955072707,0.000000001100087131523],[-0.09960628182831,0.1092652428162,-0.01782307798975,0.0001192181214757,0.0002310636556641,-0.00002492990725967,0.000001217600444191,-0.00000003624263301602,0.0000000006139167092128],[0.01709129400742,-0.01574889001363,0.002865310743302,-0.0001700396064727,-0.000001502644504654,0.0000003297869416435,0.0000000006572135289627,0.0000000004269190108005,-0.00000000003666090917669],[-0.001435304503973,0.001203823111704,-0.0002350465388313,0.00002507288189894,-0.000003077975735212,0.0000003748299687254,-0.00000002613600078122,0.0000000008263175463927,-0.000000000008509179497022],[0.00004808639828229,-0.00003761591649539,0.000007490531472388,-0.000001077314971617,0.0000001950247963978,-0.00000002569729600929,0.000000001804377780165,-0.00000000006031847199601,0.0000000000007416020205748]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6*1e20)	# m^3/s * 1e20
		if not np.shape(np.array(merge_Te_prof_multipulse_interp_crop_limited))==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
			# I have to add external bouds to avoid the extrapolation to stop working and give me crazy values
			reaction_rate[merge_Te_prof_multipulse_interp_crop_limited<1]=0
		else:
			if merge_Te_prof_multipulse_interp_crop_limited<1:
				reaction_rate = 0
		e_H2__Hp_H_2e = reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2)
	return e_H2__Hp_H_2e	# m^-3/s *1e-20 / (nH2/ne)


def RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
	if False:
		# reaction rate:
		# e +H2(X1Σ+g ; v) →H-2 (X2Σ+u ) → H- +H(1s)
		# e +H2(X1Σ+g ; v) →H−2 (B2Σ+g ) → H− +H(1s)
		if True:
			# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
			# chapter 4.4.1
			# I neglect what happens for different intermediate excited states
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
				cross_section =  coefficients[1]*np.exp(-(e_energy-np.abs(coefficients[0]))/0.45) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
				cross_section[np.logical_not(np.isfinite(cross_section))]=0
				cross_section[cross_section<0] = 0
				reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s * 1e20
				if v_index==0:
					e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[0]
				else:
					e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[v_index]
		elif False:
			# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
			# chapter 4.4.1
			# I neglect what happens for different intermediate excited states
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
			interpolation_reaction_rate = np.array([1.5800E-08,1.8056E-08,2.0202E-08,2.2169E-08,2.3891E-08,2.5299E-08,2.6336E-08,2.6954E-08,2.7125E-08,2.6839E-08,2.6113E-08,2.4986E-08,2.3516E-08,2.1779E-08,1.9857E-08,1.7834E-08,1.5789E-08,1.3790E-08,1.1893E-08,1.0137E-08,8.5487E-09,7.1395E-09,5.9110E-09,4.8562E-09,3.9627E-09,3.2146E-09,2.5945E-09,2.0848E-09,1.6691E-09,1.3321E-09,1.0602E-09,8.4191E-10,6.6722E-10,5.2786E-10,4.1696E-10,3.2890E-10,2.5912E-10,2.0391E-10,1.6031E-10,1.2592E-10,9.8839E-11,7.7541E-11,6.0811E-11,4.7682E-11,3.7388E-11,2.9318E-11,2.2991E-11,1.8026E-11,1.4124E-11,1.1050E-11]) * (1e-6*1e20)		# m^3/s * 1e20
			interpolator_reaction_rate = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_reaction_rate),fill_value='extrapolate')
			for v_index in range(5):
				if v_index<=3:
					cross_section =  coefficients_all[v_index][1]*np.exp(-(e_energy-np.abs(coefficients_all[v_index][0]))/0.45) * (1e-16 * 1e-4 * 1e20)	# m^2*1e20
					cross_section[np.logical_not(np.isfinite(cross_section))]=0
					cross_section[cross_section<0] = 0
					reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
				else:
					reaction_rate = np.zeros_like(merge_Te_prof_multipulse_interp_crop_limited)
					reaction_rate[merge_Te_prof_multipulse_interp_crop_limited>0] = np.exp(interpolator_reaction_rate(np.log(merge_Te_prof_multipulse_interp_crop_limited[merge_Te_prof_multipulse_interp_crop_limited>0])))
					reaction_rate[reaction_rate<0] = 0
				if v_index==0:
					e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
				else:
					e_H2X1Σg__Hm_H1s[v_index] += reaction_rate * merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		e_H2X1Σg__Hm_H1s = np.sum(e_H2X1Σg__Hm_H1s,axis=0)
	else:
		# reaction rate:
		# e +H2 → e +H2(v) →H +H−
		# from AMJUEL
		# 2.23 Reaction 2.2.17
		# Effective dissociative attachment rate.
		# Vibrational distribution pH2(v, Te) (vs. Te) taken into account. Only coupling to H2(v) electronic ground state.
		# No population of H2(v) from electronically excited H∗2 , no radiative transitions between vibrational levels.
		# Assume: incident H2 particle with 0.1 eV (for the rate taken to be for H2 at rest) and Ti = Te, hence: density independent vibrational distribution and effective rate, as well as neutral molecule energy independent rate.
		coefficients_all = np.array([-2.278396332892e+01,8.634828071751e-01,-1.686619409809e+00,4.392288378207e-01,-4.393128035945e-01,2.640299048385e-01,-6.748601049114e-02,7.753368735736e-03,-3.328288267126e-04])
		reaction_rate = np.exp(np.polyval(np.flip(coefficients_all,axis=0),np.log(merge_Te_prof_multipulse_interp_crop_limited))) * (1e-6*1e20)	# m^3/s * 1e20
		e_H2__Hm_H = (merge_ne_prof_multipulse_interp_crop_limited**2) * reaction_rate
	return e_H2__Hm_H	# m^-3/s *1e-20 / (nH2/ne)




def RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	# using this together with the rate from Yacora, that does not include H* as product, gives me the whole picture
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
	# reaction rate:
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 4.5.1
	# e +H2(X1Σ+g;v) →H2-(X2Σ+u) → e +H2(X1Σ+;eps)→ e +H(1s) +H(1s)
	# e +H2(X1Σ+g;v) →H2-(B2Σ+g) → e +H2(b3Σ+;eps)→ e +H(1s) +H(1s)
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
		reaction_rate = np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) * (1e-6 * 1e20)		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[0]
		else:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[v_index]
	# via H2- excited state
	# coefficients: a1,..,a6
	coefficients_all = np.array([[-15.76,-0.052659,-84.679,1.0414,-8.2933,0.18756],[-16.162,-0.0498200333333333,-74.3906666666667,1.01586,-6.15236666666667,0.227996666666667],[-16.564,-0.0469810666666667,-64.1023333333333,0.99032,-4.01143333333333,0.268433333333333],[-16.966,-0.0441421,-53.814,0.96478,-1.8705,0.30887],[-16.1206666666667,-0.0490894,-47.1276666666667,0.94422,-1.72766666666667,0.208215033333333],[-15.2753333333333,-0.0540367,-40.4413333333333,0.92366,-1.58483333333333,0.107560066666667],[-14.43,-0.058984,-33.755,0.9031,-1.442,0.0069051],[-14.4276666666667,-0.0575976666666667,-28.0646666666667,0.897233333333333,-1.5259,0.00691206666666667],[-14.4253333333333,-0.0562113333333333,-22.3743333333333,0.891366666666667,-1.6098,0.00691903333333333],[-14.423,-0.054825,-16.684,0.8855,-1.6937,0.006926],[-16.2556666666667,-0.0396174,-26.4876666666667,0.799833333333333,13.6192,0.0993073333333334],[-18.0883333333333,-0.0244098,-36.2913333333333,0.714166666666667,28.9321,0.191688666666667],[-19.921,-0.0092022,-46.095,0.6285,44.245,0.28407]])
	for v_index,coefficients in enumerate(coefficients_all):
		T_scaled = T_e_temp/1000
		reaction_rate = np.exp(coefficients[0]/(T_scaled**coefficients[1]) + coefficients[2]/(T_scaled**coefficients[3]) + coefficients[4]/(T_scaled**(2*coefficients[5])) ) * (1e-6 * 1e20) 		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[0]
		else:
			e_H2X1Σg__e_H1s_H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[v_index]
	e_H2X1Σg__e_H1s_H1s = np.sum(e_H2X1Σg__e_H1s_H1s,axis=0)
	return e_H2X1Σg__e_H1s_H1s	# m^-3/s *1e-20 / (nH2/ne)


def RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp = np.array([T_Hp])
		T_H2 = np.array([T_H2])
	if False:
		# H+ +H2(X1Σ+g ; v) →H(1s) +H2+(X2Σ+ g ; v)
		# reaction rate
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 5.2.1a
		Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
		Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
		Hp_H2X1Σg__H1s_H2pX2Σg = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
		# v=0-8
		# coefficients: E 0v(eV),a 1,...,a 6,b 1,...,b 11
		coefficients_all = np.array([[2.67,18.6,-1.66,2.53,1.93,0,0,17.3,105,2,1.0e4,-1.12,3.64e-4,0.9,5.03e-19,4,5.87e-28,5.5],[1.74,2.51,-0.56,4.21,4.07,1.0e-5,1,58,11.28,0.246,1,0,3.92e-5,1.11,4.95e-17,3.65,3.88e-26,5.2],[1.17,3.01,-0.63,7.04,10.74,1.0e-5,1,26.53,25.2,0.65,1,0,1.56e-6,1.45,5.50e-19,4,8.50e-27,5.3],[0.48,4.5,-0.57,5,14.62,1.0e-5,1,39.5,9.35,0.51,1,0,5.32e-7,1.6,3.52e-20,4.25,3.50e-27,5.4],[0,24,0.32,0,0,0.145,1.84,10.8,0,0,1,-0.297,2.92e-4,0.76,4.93e-11,2.35,2.62e-27,5.5],[0,11.75,0.092,0,0,3.86e-3,2.86,20,0,0,1,-0.193,1.36e-5,1.15,4.46e-12,2.61,4.31e-27,5.5],[0,11.58,0.091,0,0,3.84e-3,2.87,20.04,0,0,1,-0.192,1.34e-5,1.15,4.46e-12,2.61,4.31e-27,5.5],[0,0,0,0,0,0,0,33,0,0,1,-0.022,1.22e-2,0.36,6.51e-8,1.78,3.25e-23,4.86],[0,0,0,0,0,0,0,30,0,0,1,-0.017,1.87e-2,0.375,9.0e-10,2.18,1.85e-25,5.25]])
		for v_index,coefficients in enumerate(coefficients_all):
			cross_section1 =  coefficients[1]*(Hp_energy*coefficients[2])*((1-((coefficients[0]/Hp_energy)**coefficients[3]))**coefficients[4])*np.exp(-coefficients[5]*(Hp_energy**coefficients[6])) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
			cross_section1[cross_section1<0] = 0
			cross_section2 =  coefficients[7]*np.exp(-coefficients[8]/(Hp_energy**coefficients[9])) / (coefficients[10]*(Hp_energy**coefficients[11]) + coefficients[12]*(Hp_energy**coefficients[13]) + coefficients[14]*(Hp_energy**coefficients[15]) + coefficients[16]*(Hp_energy**coefficients[17]) ) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
			cross_section2[cross_section2<0] = 0
			cross_section=cross_section1+cross_section2
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
			if v_index==0:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
			else:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		# v=9-14
		for v_index in range(9,15):
			if v_index==9:
				f_v =1
			else:
				f_v = 1.97/(v_index - 8)**1.23
			cross_section =  27*f_v/((Hp_energy**0.033) + 9.85e-10 * (Hp_energy**2.16) + 1.66*f_v*1e-25 * (Hp_energy**5.25)) * (1e-16 * 1e-4 * 1e20)	# m^2*1e20
			cross_section[np.logical_not(np.isfinite(cross_section))]=0
			cross_section[cross_section<0] = 0
			reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s*1e20
			if v_index==0:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
			else:
				Hp_H2X1Σg__H1s_H2pX2Σg[v_index] += reaction_rate *nHp_ne_all* merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[v_index]
		Hp_H2X1Σg__H1s_H2pX2Σg = np.sum(Hp_H2X1Σg__H1s_H2pX2Σg,axis=0)
	elif True:	# 27/05/2020
		# from AMJUEL
		# 3.28 Reaction 3.2.3 p +H2(v) →H +H+
		coefficients_all = [[-2.133104980000e+1,2.961905900000e-1,-2.876892150000e-2,-3.323271590000e-2,7.234558340000e-3,2.940230100000e-4,-8.005031610000e-5,0.000000000000e+0,0.000000000000e+0],[2.308461720000e+0,-1.064800460000e+0,2.310120950000e-1,6.809382980000e-2,-4.241210420000e-2,8.271152020000e-3,-6.275988100000e-4,0.000000000000e+0,0.000000000000e+0],[-2.026151710000e+0,1.142806740000e+0,-2.621943460000e-1,-6.877694430000e-2,4.012716970000e-2,-6.143307540000e-3,3.233852920000e-4,0.000000000000e+0,0.000000000000e+0],[1.648000330000e-1,-4.675786500000e-1,1.242261910000e-1,1.774294860000e-2,-1.157658350000e-2,1.311061300000e-3,-1.125957730000e-5,0.000000000000e+0,0.000000000000e+0],[1.651993580000e-1,5.766584690000e-2,-3.659922760000e-2,7.083346120000e-3,3.403537010000e-4,-2.752152790000e-4,2.225165850000e-5,0.000000000000e+0,0.000000000000e+0],[-2.598458070000e-2,1.349144350000e-2,8.871659800000e-3,-5.231162040000e-3,3.324241650000e-4,1.985585660000e-4,-2.813630850000e-5,0.000000000000e+0,0.000000000000e+0],[-4.330453510000e-3,-5.246404340000e-3,-1.636107180000e-3,1.242023150000e-3,-4.524774630000e-5,-6.369415730000e-5,8.679231940000e-6,0.000000000000e+0,0.000000000000e+0],[1.187405610000e-3,6.281964210000e-4,1.740000360000e-4,-1.337853740000e-4,6.784609160000e-7,8.284840740000e-6,-1.075372230000e-6,0.000000000000e+0,0.000000000000e+0],[-6.897815380000e-5,-2.667160440000e-5,-7.528040300000e-6,5.516687380000e-6,1.140207820000e-7,-3.837975410000e-7,4.793672020000e-8,0.000000000000e+0,0.000000000000e+0]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(3/2*T_H2*eV_to_K),np.log(T_Hp*eV_to_K),coefficients_all)) * (1e-6 * 1e20)	# m^2 *1e20, T_H2 to en cin H2 must be multiplied by 3/2
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		Hp_H2v__H_H2p = reaction_rate*(merge_ne_prof_multipulse_interp_crop_limited**2)
		reaction_rate[reaction_rate<0]=0
	return Hp_H2v__H_H2p	# m^-3/s *1e-20 / (nH2/ne) / (nHp/ne)

def RR_Hp_H1s__H1s_Hp__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp = np.array([T_Hp])
		T_H = np.array([T_H])
	# from AMJUEL
	# 3.19 Reaction 3.1.8 p +H(1s) →H(1s) + p
	coefficients_all = [[-18.31670498376,0.165023933207,0.05025740610454,0.005288358515136,-0.002437122342843,-0.000446189121472,0.000173163154811,-0.00001588434781959,0.0000004482291414386],[0.2143624996483,-0.1067658289373,-0.005304993033743,0.008289383645942,-0.00009698773663345,-0.0004470180279338,0.00007944326905066,-0.000005303688417551,0.0000001235167254501],[0.05139117192662,0.009536923957409,-0.01306075129405,-0.001033166370333,0.001280464204775,-0.00008453294908907,-0.00003040874906105,0.000004747888095498,-0.0000001923953750574],[-0.0009896180369559,0.006315097684976,0.002655464630308,-0.001365781346175,-0.0001859939123743,0.0001237942304972,-0.00001588253432932,0.00000066035603458,-0.000000001970606344918],[-0.00249532754608,-0.001265503371044,0.0007569269700468,0.0002756946036257,-0.0001107375149384,-0.000007217379426085,0.000005769971321188,-0.0000006717311113584,0.00000002440961351104],[-0.00002417046684097,-0.00006945512319613,-0.0002956984088728,0.00002318277483195,0.0000370449439714,-0.00000606655869248,-0.0000004951573401626,0.0000001437520597154,-0.000000006998724470004],[0.0001177406072793,0.00003698501620365,0.00003424317896619,-0.000009815693511794,-0.000004285719813022,0.000001169257650609,-0.0000000004968953461875,-0.00000001618948982477,0.0000000009440094842562],[-0.00001483036457978,-0.000003348172574417,-0.000001527018819072,0.0000008362050692462,0.0000002058392726953,-0.00000007463594884928,0.0000000005924370389093,0.000000001078208689229,-0.00000000006619767848464],[0.0000005351909441226,0.00000009728230870242,0.00000001676354786072,-0.00000002237567830699,-0.00000000308168580382,0.000000001450862501121,0.00000000004434231893204,-0.00000000003324377862622,0.000000000001935019679501]]
	reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(3/2*T_H*eV_to_K),np.log(T_Hp*eV_to_K),coefficients_all)) * (1e-6 * 1e20)	# m^2 *1e20, T_H to en cin H must be multiplied by 3/2
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
	reaction_rate[reaction_rate<0]=0
	Hp_H1s__H1s_Hp = reaction_rate*(merge_ne_prof_multipulse_interp_crop_limited**2)
	return Hp_H1s__H1s_Hp	# m^-3/s *1e-20 / (nHp/ne) / (nH(1)/ne)


def RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp = np.array([T_Hp])
	# H+ +H2(X1Σ+ g ; v) → H+ +H(1s) +H(1s)
	# reaction rate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 5.3
	Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp),200))).T*(2*boltzmann_constant_J*T_Hp.T/hydrogen_mass)**0.5).T
	Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hp.T))**(3/2) , (T_Hp.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
	Hp_energy = 0.5 * Hp_velocity**2 * hydrogen_mass * J_to_eV
	Hp_H2X1Σg__Hp_H1s_H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	# v=0-8
	# coefficients: E 0v (eV ),a 1,...,a 4
	# valid up to 20eV
	coefficients_all = np.array([[6.717,7.52e3,4.64,5.37,2.18],[5.943,1.56e3,3.91,3.42,1.55],[5.313,3.83e2,3.22,2.71,1.5],[4.526,72.5,2.4,2.32,1.34],[3.881,23.8,1.64,1.86,1.5],[3.278,8.68,0.94,1.39,1.04],[2.716,6.85,0.58,1.2,1.14],[2.2215,8.695,0.47,1.115,1.195],[1.727,10.54,0.36,1.03,1.25],[1.3225,16.7,0.32,0.88,1.515],[0.918,22.86,0.28,0.73,1.78],[0.627,26.485,0.24,0.69,1.71],[0.336,30.11,0.2,0.65,1.64],[0.18085,32.015,0.175,0.615,1.86],[0.0257,33.92,0.15,0.58,2.08]])
	for v_index,coefficients in enumerate(coefficients_all):
		cross_section =  (coefficients[1]/(Hp_energy**coefficients[2]))*((1-(1.5*coefficients[0]/Hp_energy)**coefficients[3])**coefficients[4]) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*Hp_velocity*Hp_velocity_PDF,axis=-1)* np.mean(np.diff(Hp_velocity))		# m^3 / s
		reaction_rate_max = reaction_rate.flatten()[(np.abs(T_Hp*eV_to_K-20)).argmin()]
		reaction_rate[T_Hp*eV_to_K>20]=reaction_rate_max
		if v_index==0:
			Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *(merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[0]
		else:
			Hp_H2X1Σg__Hp_H1s_H1s[v_index] += reaction_rate *(merge_ne_prof_multipulse_interp_crop_limited**2) * fractional_population_states_H2[v_index]
	Hp_H2X1Σg__Hp_H1s_H1s = np.sum(Hp_H2X1Σg__Hp_H1s_H1s,axis=0)
	return Hp_H2X1Σg__Hp_H1s_H1s	# m^-3/s *1e-20 / (nHp/ne) / (nH2/ne)


def RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hm = np.array([T_Hm])
	# reaction rate H- +H2(v) →H +H2(v") + e
	Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm),200))).T*(2*boltzmann_constant_J*T_Hm.T/hydrogen_mass)**0.5).T
	Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm.T))**(3/2) , (T_Hm.T*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
	Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
	if False:	# NOTE: I'm not using this because the this formulas return something (cros section >0) only for high energies, something I don't have in my cases
		# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
		# chapter 6.1.1
		# v=0, only available data
		Hm_energy_scaled = Hm_energy/1000
		cross_section1 =  81.1/((Hm_energy_scaled**0.63)*(1+2.03e-6 * Hm_energy_scaled**1.95)) * ( 1-np.exp(-0.057*(np.nanmax([(Hm_energy_scaled/2.18 -1)**0.94,np.zeros_like(Hm_energy_scaled)],axis=0)) )) * (1e-16 * 1e-4 * 1e20)	# m^2
		cross_section1[cross_section1<0] = 0
		cross_section2 =  1.22e3 /((Hm_energy_scaled**0.5)*(1+6.91e-4 * Hm_energy_scaled**0.4)) * np.exp(-125/(Hm_energy_scaled**0.663)) * (1e-16 * 1e-4 * 1e20)	# m^2 * 1e20
		cross_section2[cross_section2<0] = 0
		cross_section=cross_section1+cross_section2
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		reaction_rate = np.sum(cross_section*Hm_velocity*Hm_velocity_PDF,axis=-1)* np.mean(np.diff(Hm_velocity))		# m^3 / s
		v_index=0
		Hm_H2v__H_H2v_e = reaction_rate *nHm_ne_all* merge_ne_prof_multipulse_interp_crop_limited * population_states_H2[0]
	else:
		# data from https://www-amdis.iaea.org/ALADDIN/
		# no info on the vibrational state, I assume it is valid for all
		interpolation_ev = np.array([2.3183E+00,3.2014E+00,4.4210E+00,6.1052E+00,8.4309E+00,1.1643E+01,1.6078E+01,2.2203E+01,3.0661E+01,4.2341E+01,5.8471E+01,8.0746E+01,1.1151E+02,1.5398E+02,2.1264E+02,2.9365E+02,4.0552E+02,5.6000E+02,7.7333E+02,1.0679E+03,1.4748E+03,2.0366E+03,2.8124E+03,3.8838E+03,5.3633E+03,7.4065E+03,1.0228E+04,1.4124E+04,1.9505E+04,2.6935E+04,3.7197E+04,5.1367E+04,7.0935E+04,9.7957E+04,1.3527E+05,1.8681E+05,2.5797E+05,3.5624E+05,4.9196E+05,6.7937E+05,9.3817E+05,1.2956E+06,1.7891E+06,2.4707E+06,3.4119E+06,4.7116E+06,6.5065E+06,8.9852E+06,1.2408E+07,1.7135E+07])	# eV
		interpolation_cross_section = np.array([9.5063E-17,1.5011E-16,2.1010E-16,2.6792E-16,3.1859E-16,3.6021E-16,3.9345E-16,4.2053E-16,4.4425E-16,4.6737E-16,4.9230E-16,5.2099E-16,5.5498E-16,5.9540E-16,6.4289E-16,6.9763E-16,7.5912E-16,8.2608E-16,8.9629E-16,9.6648E-16,1.0324E-15,1.0892E-15,1.1314E-15,1.1542E-15,1.1537E-15,1.1280E-15,1.0773E-15,1.0041E-15,9.1306E-16,8.1021E-16,7.0205E-16,5.9470E-16,4.9323E-16,4.0127E-16,3.2091E-16,2.5285E-16,1.9672E-16,1.5146E-16,1.1562E-16,8.7634E-17,6.6012E-17,4.9419E-17,3.6733E-17,2.7048E-17,1.9657E-17,1.4021E-17,9.7429E-18,6.5296E-18,4.1680E-18,2.4949E-18]) * (1e-4)		# m^2
		interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
		cross_section=np.exp(interpolator_cross_section(np.log(Hm_energy)))
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		cross_section[Hm_energy<1] = 0	# added to avoid to use a cross section of which I have no idea on the reliability below 2.3eV
		reaction_rate = np.sum(cross_section*Hm_velocity*Hm_velocity_PDF,axis=-1)* np.mean(np.diff(Hm_velocity))		# m^3 / s
		if not np.shape(np.array(merge_Te_prof_multipulse_interp_crop_limited))==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		Hm_H2v__H_H2v_e = reaction_rate*(merge_ne_prof_multipulse_interp_crop_limited**2) * 1e20	# m^-3/s *1e-20
	return Hm_H2v__H_H2v_e	# m^-3/s *1e-20 / (nHm/ne) / (nH2/ne)

def RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
	# H(1s) +H2(v) → H(1s) + 2H(1s)
	# reaction rate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 6.2.2
	H1s_H2v__H1s_2H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	# coefficients: a1,..,a5
	coefficients_all = np.array([[2.06964e1,7.32149e7,1.7466,4.75874e3,-9.42775e-1],[2.05788e1,4.32679e7,1.6852,1.91812e3,-8.16838e-1],[2.05183e1,5.15169e7,1.73345,3.09006e3,-8.88414e-1],[2.04460e1,1.87116e8,1.87951,9.04442e3,-9.78327e-1],[2.03608e1,4.93688e8,1.99499,2.32656e4,-1.06294],[2.02426e1,1.80194e8,1.92249,1.28777e4,-1.02713],[2.00161e1,2.96945e5,1.31044,9.55214e2,-1.07546],[1.98954e1,4.53104e5,1.37055,3.88065e2,-8.71521e-1],[1.97543e1,5.13174e5,1.39819,3.54272e2,-8.07563e-1],[1.97464e1,9.47230e4,1.24048,2.28283e2,-8.51591e-1],[1.95900e1,6.43990e4,1.22211,1.16196e2,-7.35645e-1],[1.94937e1,3.49017e4,1.20883,1.26329e2,-8.15130e-1],[1.90708e1,1.05971e5,9.91646e-1,1.05518e2,-1.93837e-1],[1.89718e1,7.76046e5,7.84577e-1,1.31409e3,-1.00479e-2],[1.87530e1,5.81508e5,7.35904e-1,1.69328e3,4.47757e-3]])
	for v_index,coefficients in enumerate(coefficients_all):
		reaction_rate = np.exp(-coefficients[0] -coefficients[1]/(T_H**coefficients[2]*(1+coefficients[3]*(T_H**coefficients[4])) )) *(1e-6 * 1e20) 		# m^3 / s * 1e20
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		if v_index==0:
			H1s_H2v__H1s_2H1s[v_index] += reaction_rate *(merge_ne_prof_multipulse_interp_crop_limited**2)* fractional_population_states_H2[0]
		else:
			H1s_H2v__H1s_2H1s[v_index] += reaction_rate *(merge_ne_prof_multipulse_interp_crop_limited**2)* fractional_population_states_H2[v_index]
	H1s_H2v__H1s_2H1s = np.sum(H1s_H2v__H1s_2H1s,axis=0)
	return H1s_H2v__H1s_2H1s	# m^-3/s *1e-20 / (nH2/ne) / (nH(1)/ne)

def RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2 = np.array([T_H2])
	# H2(v1 = 0) +H2(v) →H2(v1 = 0) + 2H(1s)
	# reaction rate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 6.3.2
	# valid up to 20000K
	# I'm not sure which temperature to use here. I will use T_H2
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	H2v0_H2v__H2v0_2H1s = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	for v_index in range(len(fractional_population_states_H2)):
		reaction_rate_0 = 1.3*(1 +0.023*v_index + 1.93e-5 * (v_index**4.75) + 2.85e-24 *(v_index**21.6)  )
		T_0 = (7.47 - 0.322 *v_index) * 1e3
		reaction_rate = reaction_rate_0 * np.exp(-T_0/T_H2) * (1e-10 *1e-6 * 1e20) 		# m^3 / s * 1e20
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		reaction_rate_max = reaction_rate.flatten()[(np.abs(T_H2-20000)).argmin()]
		reaction_rate[T_H2>20000]=reaction_rate_max
		if v_index==0:
			H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) / q_vibr * fractional_population_states_H2[0]
		else:
			H2v0_H2v__H2v0_2H1s[v_index] += reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2) / q_vibr * fractional_population_states_H2[v_index]
	H2v0_H2v__H2v0_2H1s = np.sum(H2v0_H2v__H2v0_2H1s,axis=0)
	return H2v0_H2v__H2v0_2H1s	# m^-3/s *1e-20 / (nH2/ne) / (nH2/ne)

def RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
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
		cross_section = 13.2*np.log(np.exp(1) + 2.55e-4 * e_energy)/( (e_energy**0.31)*(1+0.017*(e_energy**0.76)))* (1e-16 * 1e-4 * 1e20)
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-0.01)).argmin()]
		cross_section[e_energy<0.01]=cross_section_min
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__XXX__e_Hp_H1s = merge_ne_prof_multipulse_interp_crop_limited *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*reaction_rate
	elif True:
		# from AMJUEL
		# 4.14 Reaction 2.2.12 e +H+2 → e +H +H+
		# I use this because it includes ne in the calculus
		coefficients_all = [[-17.934432746,-0.04932783688604,0.1039088280849,-0.04375935166008,0.009196691651936,-0.001043378648769,0.00006600342421838,-0.000002198466460165,0.00000003004145701249],[2.236108757681,-0.02545406018621,-0.1160421006835,0.04407846563362,-0.008192521304984,0.0008200277386433,-0.00004508284363534,0.000001282824614809,-0.00000001474719350236],[-0.3620018994703,0.0672152768015,0.01564387124002,-0.004939045440424,0.0004263195867947,0.00001034216805418,-0.0000039750286019,0.0000002322116289258,-0.00000000438121715447],[-0.4353922258965,-0.03051033606589,0.03512861172521,-0.01179504564265,0.002091772760029,-0.0001991100044575,0.00001018080238045,-0.0000002597941866088,0.000000002524118386011],[0.1580381801957,0.002493654957203,-0.01601970998119,0.005346709597939,-0.0008711870134835,0.00007542066727545,-0.000003410778344979,0.00000007120460603822,-0.0000000004412295474522],[0.01697880687685,0.0021066759639,0.000452198335817,-0.0003017151690655,0.00006209239389357,-0.000007598119096817,0.0000005523273241689,-0.00000002130508249251,0.0000000003319099650589],[-0.01521914651109,-0.0007527862162788,0.0009095551479381,-0.0002372576223034,0.00003018561480848,-0.000001365255868731,-0.00000004604769733903,0.00000000586791027043,-0.0000000001357779142836],[0.00240627636807,0.00009971361856278,-0.0001760978402353,0.00004877659148871,-0.000006477358351729,0.0000003541106430252,0.00000000130977289967,-0.000000000807290733423,0.00000000002074669430611],[-0.0001219469579955,-0.000004785505675232,0.000009858840337511,-0.000002779210878533,0.0000003720379996058,-0.00000002110289928486,0.00000000003753875073646,0.00000000004024906665497,-0.000000000001075990572574]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6 * 1e20)	# m^3/s * 1e20
		if not np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		e_H2p__XXX__e_Hp_H = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return e_H2p__XXX__e_Hp_H	# m^-3/s *1e-20 / (nH2p/ne)

def RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
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
		cross_section = 17.3* ( 1/((e_energy**0.665)*(1+1.1*(e_energy**0.512) + 0.011*(e_energy**3.10)) ) + 0.133* np.exp(-0.35*((e_energy - 6.05)**2)) )* (1e-16 * 1e-4 * 1e20)		# m^2 * 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section_min = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-0.01)).argmin()]
		cross_section_max = cross_section.flatten()[(np.abs(merge_Te_prof_multipulse_interp_crop_limited-10)).argmin()]
		cross_section[e_energy<0.01]=cross_section_min
		cross_section[e_energy>10]=cross_section_max
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__H1s_Hn2 = (merge_ne_prof_multipulse_interp_crop_limited**2) *nH2p_ne_all*reaction_rate
	elif True:
		# from AMJUEL
		# 4.15 Reaction 2.2.14 e +H+2 →H +H
		# I use this because it includes ne in the calculus
		coefficients_all = [[-16.64335253647,0.08953780953631,-0.1056411030518,0.0447700080869,-0.009729945434357,0.001174456882002,-0.00007987743820637,0.000002842957892768,-0.00000004104508608435],[-0.6005444031657,0.04063933992726,-0.04753947846841,0.02188304031377,-0.005201085606791,0.0006866340394051,-0.00005059940013116,0.000001930213882205,-0.00000002963966822809],[0.0004494812032769,0.00007884508616595,0.0003688007562485,-0.0004659255785539,0.00019071159804,-0.00003434324710145,0.000003067651560323,-0.000000132568946559,0.00000000221249307362],[0.0001632894866655,0.0003108116177617,-0.0003521552580917,-0.0002233169775063,0.0001869415236037,-0.00004329991211511,0.000004465256901322,-0.0000002136296167564,0.000000003873085368404],[-0.00007234142549752,-0.001316311320262,0.001643509328764,-0.0006412764282779,0.0001048891053765,-0.000007018555173322,0.00000004776213235854,0.00000001380537343974,-0.0000000004199397846492],[-0.00001504085050039,0.0001315865970237,-0.0001025653773999,0.00005310324781249,-0.00001831888048039,0.000003423755373077,-0.0000003303384352061,0.000000015516270977,-0.0000000002809391819541],[0.00001113923667684,0.00002711411525392,-0.00008495922363727,0.00004026487801017,-0.00000628932447424,0.0000001911447036702,0.00000003638198230235,-0.000000003235540606394,0.00000000007605442050634],[-0.00000184392616225,-0.000001663674537499,0.00001308069926896,-0.000007324021449032,0.000001431739868187,-0.0000001085644779665,0.000000001143164983367,0.0000000002151595003971,-0.000000000007052562220005],[0.00000009864173150662,-0.0000002212261708468,-0.0000004431749501051,0.0000003270530731011,-0.00000007282085521177,0.000000006578253567957,-0.0000000001925258267827,-0.000000000004217474167519,0.0000000000002364754029318]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6 * 1e20)	# m^3/s * 1e20
		if not np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		e_H2p__H_H = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return e_H2p__H_H	# m^-3/s *1e-20 / (nH2p/ne)

def RR_e_H2p__Hex_H__or__Hex_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	# I can use this ONLY for H2+
	# I CANNOT use this for H(p), H, e
	# H2p and e can do:
	# H2+ + e- → H(p) + H+ + e-
	# H2+ + e- → H(p) + H(1)
	# that have different products based on the path
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])

	# H2+ + e- → H(v) + H+ + e-
	# Yacora, Dissociation of H2+
	# H2+ + e- → H(v) + H(g)
	# Yacora, Dissociative recombination of H2+
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		population_coefficients = From_H2p_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2p))
		reaction_rate = (ne**2)*1e20*np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1)	# 	#/s * 1e-20
	else:
		population_coefficients = From_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2p))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited.flatten()**2)*1e20*(np.sum(population_coefficients*einstein_coeff_full_cumulative,axis=-1).reshape(np.shape(merge_Te_prof_multipulse_interp_crop_limited)))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nH2p/ne)

def RR_e_H2p__Hex_H__or__Hn_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	# used ONLY to calculate the controbution to the potential energy differenciated by hydrogen excider state
	# H2p and e can do:
	# H2+ + e- → H(p) + H+ + e-
	# H2+ + e- → H(p) + H(1)
	# that have different products based on the path
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])

	# H2+ + e- → H(v) + H+ + e-
	# Yacora, Dissociation of H2+
	# H2+ + e- → H(v) + H(g)
	# Yacora, Dissociative recombination of H2+
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	# einstein_coeff_Lyman = np.array([4.6986,5.5751e-01,1.2785e-01,4.1250e-02,1.6440e-02,7.5684e-03,3.8694e-03,2.1425e-03,1.2631e-03,7.8340e-04,5.0659e-04,3.3927e-04]) * 1e8  # 1/s

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		population_coefficients = From_H2p_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2p))
		reaction_rate = (ne**2)*1e20*(population_coefficients*einstein_coeff_full_cumulative)	# 	#/s * 1e-20
	else:
		population_coefficients = From_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2p))
		reaction_rate = (merge_ne_prof_multipulse_interp_crop_limited**2)*1e20*np.transpose((population_coefficients*einstein_coeff_full_cumulative).reshape((*np.shape(merge_Te_prof_multipulse_interp_crop_limited),len(einstein_coeff_full_cumulative))),(-1,*np.arange(len(merge_Te_prof_multipulse_interp_crop_limited.shape))))	# 	#/s * 1e-20
	return reaction_rate	# m^-3/s *1e-20 / (nH2p/ne)


def RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if False:
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
			print('error, Te and ne are different shapes')
			exit()
		if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
			merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
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
		cross_section = 7.39/e_energy * np.log(0.18*e_energy) * (1-np.exp(-0.105*((e_energy/15.2 -1)**1.55))) * (1e-16 * 1e-4 * 1e20)		# m^2 * 1e20
		cross_section[np.logical_not(np.isfinite(cross_section))]=0
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum(cross_section*e_velocity*e_velocity_PDF,axis=-1)* np.mean(np.diff(e_velocity))		# m^3 / s
		e_H2p__e_Hp_Hp_e = merge_ne_prof_multipulse_interp_crop_limited *nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited*reaction_rate
	elif True:
		# from AMJUEL
		# 4.13 Reaction 2.2.11 e +H2+ → 2e +H+ +H+
		# I use this because it included also ne other than Te
		coefficients_all = [[-37.08803769397,0.09784233987341,-0.00720036127213,0.006496843022778,-0.00142059081876,0.0001703620321164,-0.000011607389464,0.0000004148222302162,-0.000000006007853385325],[15.61780529774,-0.01673256230592,0.02743322772895,-0.01026956102747,0.001999561527383,-0.0002043607814503,0.00001084177127603,-0.0000002671800995803,0.000000002093182411476],[-6.874406034117,-0.007782929961315,-0.006888773684846,0.002306107197863,-0.0004029222834436,0.00003932152471491,-0.00000209490736415,0.0000000568290706001,-0.000000000632075254561],[2.010540060675,-0.003226785148562,-0.006181192193854,0.002388146990238,-0.0005018901320009,0.00005520233512352,-0.000003080798536641,0.00000007864770315002,-0.0000000006357395371638],[-0.361476890612,0.003710098881765,0.002045814599796,-0.0008523935993991,0.0001751295192861,-0.00001944203941844,0.000001138888354831,-0.00000003256303793266,0.0000000003501794038444],[0.02956861321735,-0.0005524443504504,-0.00002457951062112,0.00003433179945503,-0.000001450208898992,-0.0000002447566480782,0.00000001375679100044,0.0000000004863880510459,-0.00000000003004374374556],[0.0009662490252868,-0.0001548556801431,0.00001417215042439,-0.000006444863591678,-0.000001566028729499,0.0000004152486680818,-0.00000002855068942744,0.0000000006081804811,0.0000000000009512865901179],[-0.0003543571865464,0.00004662969089421,-0.00001471117766355,0.000005235585096328,-0.0000005779667826854,0.00000002139729421817,-0.000000000365604842523,0.00000000003759866326965,-0.000000000001486151370215],[0.00001827109843671,-0.000003179895716088,0.000001432429412413,-0.0000005141065080107,0.00000007734387173369,-0.000000006163336831045,0.0000000003128313515842,-0.00000000001061842444216,0.000000000000177109976964]]
		reaction_rate = np.exp(np.polynomial.polynomial.polyval2d(np.log(merge_Te_prof_multipulse_interp_crop_limited),np.log(merge_ne_prof_multipulse_interp_crop_limited*1e6),coefficients_all)) * (1e-6*1e20)	# m^3/s * 1e20
		if not np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
			reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
			# I have to add external bouds to avoid the extrapolation to stop working and give me crazy values
			reaction_rate[merge_Te_prof_multipulse_interp_crop_limited<0.5]=0
		else:
			if merge_Te_prof_multipulse_interp_crop_limited<0.5:
				reaction_rate = 0
		e_H2p__e_Hp_Hp_e = reaction_rate * (merge_ne_prof_multipulse_interp_crop_limited**2)
	return e_H2p__e_Hp_Hp_e	# m^-3/s *1e-20 / (nH2p/ne)


# reaction rates H(1s) +H+2 (v) → H+ +H2(v?)
# PROBLEM I can't find the energy levels of H2+, together with multeplicity.
# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
# chapter 7.2.2a
# Therefore I cannot get the population density, and as a consequence the reaction rate.

# I can get the vibrationally resolved cross section from
# https://www-amdis.iaea.org/ALADDIN/
# but only for very high energyes, (>200eV), so I cannot use them.

def RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2p = np.array([T_H2p])
		T_H = np.array([T_H])
	# reaction rate
	# H(1s) +H2+(v) → (H3+)∗ →H +H+ +H
	# H(1s) +H2+(v) → [H2+(2pσu/2pπu···)] → H +H+ +H
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 7.2.3
	# NO: I take it from https://www-amdis.iaea.org/ALADDIN/
	# H2+ [1sσg;v=0-9], H [G] → H+, H [G], H [G]
	interpolation_ev1 = np.array([1.0000E-01,1.3216E-01,1.7467E-01,2.3085E-01,3.0509E-01,4.0321E-01,5.3289E-01,7.0428E-01,9.3080E-01,1.2302E+00,1.6258E+00,2.1487E+00,2.8398E+00,3.7531E+00,4.9602E+00,6.5555E+00,8.6638E+00,1.1450E+01,1.5133E+01,2.0000E+01])
	interpolation_ev2 = np.array([0.1,0.4,1,1.5,5,20])
	interpolation_reaction_rate = np.array([[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,7.8250E+00,6.3511E+00,3.6876E+00,1.4091E+00,1.6587E+00,1.5533E-78,1.7749E-61,2.4111E-48,2.5606E-38,9.9662E-31],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,2.8825E+00,1.0492E+00,1.2838E+00,3.2525E+00,5.6641E+00,1.3117E-93,4.8863E-75,2.3583E-60,8.3672E-49,9.1706E-40,1.0101E-32,2.9288E-27,4.8779E-23],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,1.1780E+00,2.5938E+00,2.3298E+00,3.5205E+00,2.0763E+00,6.7252E+00,1.0927E-84,5.6088E-70,4.1677E-58,1.5679E-48,8.3129E-41,1.4278E-34,1.5571E-29,1.8523E-25,3.7074E-22],[0.0000E+00,0.0000E+00,0.0000E+00,0.0000E+00,3.2127E+00,9.4328E+00,2.8409E+00,3.1842E+00,2.7473E+00,2.3172E+00,1.5918E-88,5.1672E-74,3.3761E-62,1.4551E-52,1.0872E-44,3.0771E-38,6.1862E-33,1.4606E-28,6.0452E-25,6.0274E-22],[1.0630E+00,5.8307E+00,2.4311E+00,3.0605E+00,2.6228E+00,2.1022E+00,1.4070E+00,4.8355E-86,3.8061E-74,2.3267E-64,2.9679E-56,1.7448E-49,8.8682E-44,6.3858E-39,9.5609E-35,3.9963E-31,5.8340E-28,3.5211E-25,9.9705E-23,1.4563E-20],[1.9214E+00,1.0314E-86,3.5242E-74,3.3516E-64,3.0146E-56,7.0056E-50,9.5280E-45,1.4641E-40,4.2859E-37,3.5980E-34,1.1866E-31,1.9489E-29,1.8981E-27,1.2415E-25,5.9392E-24,2.1959E-22,6.4862E-21,1.5570E-19,3.0578E-18,4.9127E-17]])
	interpolator_reaction_rate = interpolate.interp2d(np.log(interpolation_ev1),np.log(interpolation_ev2), np.log(interpolation_reaction_rate),copy=False)
	selected = np.logical_and(T_H>0,T_H2p>0)
	reaction_rate = np.zeros_like(merge_Te_prof_multipulse_interp_crop_limited)
	temp = interpolator_reaction_rate(np.log(T_H2p[selected]*eV_to_K),np.log(T_H[selected]*eV_to_K))
	temp[np.logical_not(np.isfinite(temp))]=-np.inf
	if len(temp)==1:
		reaction_rate[selected] = np.exp(temp) * (1e-6 * 1e20)		# m^3/s * 1e20
	else:
		reaction_rate[selected] = np.exp(np.diag(temp)) * (1e-6 * 1e20)		# m^3/s * 1e20
	H1s_H2pv__Hp_H_H = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H1s_H2pv__Hp_H_H	# m^-3/s *1e-20 / (nH2p/ne) / (nH(1)/ne)

def RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2p = np.array([T_H2p])
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
	cross_section2 = 0.139*(H2p_energy**0.318) * np.exp(-680/(H2p_energy**2.1))/(1+2.75e-12*(H2p_energy**2.65)+9.04e-23*(H2p_energy*4.65)) * (1e-16 * 1e-4 * 1e20)		# m^2 * 1e20
	cross_section2[cross_section2<0] = 0
	cross_section = cross_section1 + cross_section2
	cross_section[np.logical_not(np.isfinite(cross_section))]=0
	cross_section_max = cross_section.flatten()[(np.abs(H2p_energy-30000)).argmin()]
	cross_section[H2p_energy>30000]=cross_section_max
	reaction_rate = np.sum(cross_section*H2p_velocity*H2p_velocity_PDF,axis=-1)* np.mean(np.diff(H2p_velocity))		# m^3 / s
	H2pvi_H2v0__Hp_H_H2v1 = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H2pvi_H2v0__Hp_H_H2v1	# m^-3/s *1e-20 / (nH2p/ne) / (nH2/ne)

def RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2p = np.array([T_H2p])
		T_H2 = np.array([T_H2])
	# H2+(vi) +H2(v0) →H3+(v3) +H(1s)
	# reaction rate
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 7.3.3
	H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
	H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
	H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
	H2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2.T),200))).T*(2*boltzmann_constant_J*T_H2/(2*hydrogen_mass))**0.5).T
	H2_velocity_PDF = (4*np.pi*(H2_velocity.T)**2 * gauss( H2_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2))**(3/2) , (T_H2*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
	H2_energy = 0.5 * H2_velocity**2 * (2*hydrogen_mass) * J_to_eV
	baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + H2_energy).T
	# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
	form_factor = 17.76/( (baricenter_impact_energy**0.477)*(1+0.0291*(baricenter_impact_energy**3.61)+1.53e-5 * (baricenter_impact_energy**6.55) )  ) * (1e-16 *1e-4 * 1e20)	# m^2 * 1e20
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
			if vi_index==0:
				H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * fractional_population_states_H2[0]
			else:
				H2pvi_H2v0__H3p_H1s[v0_index] += reaction_rate * fractional_population_states_H2[v0_index]
	H2pvi_H2v0__H3p_H1s = (merge_ne_prof_multipulse_interp_crop_limited**2)*np.sum(H2pvi_H2v0__H3p_H1s,axis=0)
	return H2pvi_H2v0__H3p_H1s	# m^-3/s *1e-20 / (nH2p/ne) / (nH2/ne)


def RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2p = np.array([T_H2p])
		T_Hm = np.array([T_Hm])
	# reaction rate :
	# H2+(vi) +H- → (H3∗) →H2(N1,3Λσ;v0) +H(1s),N ≤ 4
	# I only know about N=2,3
	# I should, but NOT include
	# H2+(vi) +H- →H2(X1Σ+g;v0) +H(n≥2)
	# because I don't have any info on that one reaction
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 7.4.1
	# I don't have any info on the functions I should be using to calculate the energy resolved cross section, so I use the flat estimate they give
	def internal(*arg):
		T_H2p,T_Hm = arg[0]
		T_H2p = np.array([T_H2p])
		T_Hm = np.array([T_Hm])
		H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
		H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
		H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
		Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/hydrogen_mass)**0.5).T
		Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)).T
		Hm_energy = 0.5 * Hm_velocity**2 * hydrogen_mass * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + Hm_energy).T
		# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
		cross_section1 = 7.5e-15 * (1e-4 *1e20)		# m^2	# channel N=2 * 1e20
		cross_section2 = 5e-14 * (1e-4 * 1e20)		# m^2	# channel N=2 * 1e20
		# cross_section = cross_section1 + cross_section2
		cross_section = np.ones_like(baricenter_impact_energy) * (cross_section1 + cross_section2)
		reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H2p.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H2p))[0]
	H2p_Hm__H2N13Λσ_H1s = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H2p_Hm__H2N13Λσ_H1s	# m^-3/s *1e-20 / (nH2p/ne) / (nHm/ne)


def RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H2p = np.array([T_H2p])
		T_Hm = np.array([T_Hm])
	# reaction rate H+2 (vi) +H- → (H∗ 3 ) →H+ 3 (v3) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 7.4.2
	def internal(*arg):
		T_H2p,T_Hm = arg[0]
		T_H2p = np.array([T_H2p])
		T_Hm = np.array([T_Hm])
		H2p_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H2p),200))).T*(2*boltzmann_constant_J*T_H2p.T/(2*hydrogen_mass))**0.5).T
		H2p_velocity_PDF = (4*np.pi*(H2p_velocity.T)**2 * gauss( H2p_velocity.T, ((2*hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H2p.T))**(3/2) , (T_H2p.T*boltzmann_constant_J/(2*hydrogen_mass))**0.5 ,0)).T
		H2p_energy = 0.5 * H2p_velocity**2 * (2*hydrogen_mass) * J_to_eV
		Hm_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hm.T),200))).T*(2*boltzmann_constant_J*T_Hm/(hydrogen_mass))**0.5).T
		Hm_velocity_PDF = (4*np.pi*(Hm_velocity.T)**2 * gauss( Hm_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_Hm))**(3/2) , (T_Hm*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		Hm_energy = 0.5 * Hm_velocity**2 * (hydrogen_mass) * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H2p),200))+H2p_energy).T + Hm_energy).T
		# baricenter_impact_energy = ((np.zeros((200,*np.shape(T_Hp),200))+Hp_energy).T + Hm_energy).T
		cross_section = 0.38/((baricenter_impact_energy**0.782)*(1+0.039*(baricenter_impact_energy**2.62))) * (1e-16 *1e-4 * 1e20)	# m^2 * 1e20
		cross_section[cross_section<0] = 0
		reaction_rate = np.nansum((cross_section*H2p_velocity*H2p_velocity_PDF).T * Hm_velocity*Hm_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H2p_velocity))*np.mean(np.diff(Hm_velocity))		# m^3 / s
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H2p.flatten(),T_Hm.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H2p))[0]
	H2p_Hm__H3p_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H2p_Hm__H3p_e	# m^-3/s *1e-20 / (nH2p/ne) / (nHm/ne)


def RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_Hp = np.array([T_Hp])
		T_H = np.array([T_H])
	# reaction rate		H+ +H(1s) +H(1s) →H(1s) +H2+ (ν)
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.2.4, equation 46b
	# valid for temperature up to 30000K (~2.587eV)
	temperature = np.mean(np.array([T_Hp,T_H,T_H]),axis=0)
	reaction_rate = 1.238/(( temperature )**1.046) *(1e-29 * 1e-6 * 1e-6 * 1e40)		# m^6/s
	reaction_rate[np.logical_not(np.isfinite(reaction_rate))] = 0
	reaction_rate_max = reaction_rate.flatten()[(np.abs(temperature-30000)).argmin()]
	reaction_rate[temperature>30000]=reaction_rate_max
	Hp_H_H__H_H2p = (merge_ne_prof_multipulse_interp_crop_limited**3)*reaction_rate
	return Hp_H_H__H_H2p	# m^-3/s *1e-20 / (nHp/ne) / (nH(1)/ne) / (nH(1)/ne)


def RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
	# reaction rate		H(1s) +H(2) → H2+(v) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# and from Associative Ionisation in Low Energy Collisions, Brouillard, F., Urbain, X., 2002
	# chapter 2.3.3
	# based on baricenter impact energy: I assume 1eV, maxwellian distribution
	def internal(T_H):
		T_H = np.array([T_H])
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
		cross_section = gauss(baricenter_impact_energy,2.5 * (1e-17 * 1e-4 * 1e20),0.9,3.25) # multiplied 1e20
		reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s * 1e20
		# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H))[0]
	H1s_H_2__H2p_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H1s_H_2__H2p_e	# m^-3/s *1e-20 / (nH(2)/ne) / (nH(1)/ne)

def RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
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
		cross_section[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] += 2.96*(3**4)*1e-19*1e-4/(baricenter_impact_energy[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)]) * 1e20 # multiplied 1e20
		cross_section[np.logical_and(baricenter_impact_energy<=1,baricenter_impact_energy>0.1)] += 2.96*(3**4)*1e-18*1e-4 * 1e20 # multiplied 1e20
		cross_section[baricenter_impact_energy>1] += 2.96*(3**4)*1e-18*1e-4/(baricenter_impact_energy[baricenter_impact_energy>1]**(0.4*3)) * 1e20 # multiplied 1e20
		cross_section[cross_section<0] = 0
		reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))**2		# m^3 / s
		H1s_H_3__H2p_e = population_states[1]*(merge_ne_prof_multipulse_interp_crop_limited *nH_ne_all - np.sum(population_states,axis=0))*reaction_rate
	else:
		# data from https://www-amdis.iaea.org/ALADDIN/
		def internal(T_H):
			T_H = np.array([T_H])
			H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
			H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
			H_1_energy = 0.5 * H_1_velocity**2 * (hydrogen_mass) * J_to_eV
			H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(hydrogen_mass))**0.5).T
			H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
			H_2_energy = 0.5 * H_2_velocity**2 * (hydrogen_mass) * J_to_eV
			baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
			interpolation_ev = np.array([5.5235E-02,7.1967E-02,9.0916E-02,1.1491E-01,1.4010E-01,1.7236E-01,2.0159E-01,2.3787E-01,2.7819E-01,3.2052E-01,3.6487E-01,4.1326E-01,4.5760E-01,5.2211E-01,5.7453E-01,6.2694E-01,6.9145E-01,7.5797E-01,8.3659E-01,9.0110E-01,9.7972E-01,1.0583E+00,1.1491E+00,1.2297E+00,1.3103E+00,1.4010E+00,1.5018E+00,1.5925E+00,1.6933E+00,1.7941E+00,1.8949E+00,1.9957E+00,2.1167E+00,2.2376E+00,2.3586E+00,2.4594E+00,2.5803E+00,2.7214E+00,2.8424E+00,2.9835E+00,3.1045E+00,3.2456E+00,3.3867E+00,3.5278E+00,3.6689E+00,3.8302E+00,3.9713E+00,4.1326E+00,4.2938E+00,4.4551E+00,4.6164E+00,4.7776E+00,4.9591E+00,5.1203E+00,5.3018E+00,5.4630E+00,5.6445E+00,5.8259E+00,5.9872E+00])
			interpolation_cross_section = np.array([7.3500E-16,5.9000E-16,4.2400E-16,3.1400E-16,2.3800E-16,2.0200E-16,3.1400E-16,2.6600E-16,2.1400E-16,2.1900E-16,1.9100E-16,2.0200E-16,2.6300E-16,2.6200E-16,2.3800E-16,2.6200E-16,2.1600E-16,2.3400E-16,2.5400E-16,2.8400E-16,2.7500E-16,2.8900E-16,3.0800E-16,3.0500E-16,3.3000E-16,2.8000E-16,3.1200E-16,2.3200E-16,3.0500E-16,2.7100E-16,2.5000E-16,2.2800E-16,2.4900E-16,2.3100E-16,2.0500E-16,1.9200E-16,2.1400E-16,2.0200E-16,2.1000E-16,2.0100E-16,1.8700E-16,1.8600E-16,2.0800E-16,1.7100E-16,1.6100E-16,1.5200E-16,1.6100E-16,1.2900E-16,1.2500E-16,1.2500E-16,1.2600E-16,1.1000E-16,1.0300E-16,1.1200E-16,1.0600E-16,9.3700E-17,7.7400E-17,8.0400E-17,6.9900E-17]) * (1e-4 * 1e20)		# m^2 * 1e20
			interpolator_cross_section = interpolate.interp1d(np.log(interpolation_ev), np.log(interpolation_cross_section),fill_value='extrapolate')
			cross_section = np.zeros_like(baricenter_impact_energy)
			cross_section[baricenter_impact_energy>0] = np.exp(interpolator_cross_section(np.log(baricenter_impact_energy[baricenter_impact_energy>0])))
			cross_section[cross_section<0] = 0
			reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s * 1e20
			return reaction_rate
		reaction_rate = list(map(internal,(np.array([T_H.flatten()]).T)))
		reaction_rate = np.reshape(reaction_rate,np.shape(T_H))[0]
		H1s_H_3__H2p_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H1s_H_3__H2p_e	# m^-3/s *1e-20 / (nH(3)/ne) / (nH(1)/ne)

def RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
	# reaction rate		H(1s) +H(4) → H2+(v) + e
	# from Collision Processes in Low-Temperature Hydrogen Plasmas, Janev, R K et al., 2003
	# chapter 2.3.3
	# based on baricenter impact energy: I assume 1eV, maxwellian distribution
	# T_H_single = 1/eV_to_K	# K
	# H_velocity = np.logspace(np.log10(0.001),np.log10(10),200)*(2*boltzmann_constant_J*T_H_single/hydrogen_mass)**0.5
	# H_velocity_PDF = 4*np.pi*H_velocity**2 * gauss( H_velocity, (hydrogen_mass/(2*np.pi*boltzmann_constant_J*T_H_single))**(3/2) , (T_H_single*boltzmann_constant_J/hydrogen_mass)**0.5 ,0)
	# H_energy = 0.5 * H_velocity**2 * hydrogen_mass * J_to_eV
	# baricenter_impact_energy = (np.zeros((len(H_energy),len(H_energy)))+H_energy).T + H_energy
	def internal(T_H):
		T_H = np.array([T_H])
		H_1_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
		H_1_velocity_PDF = (4*np.pi*(H_1_velocity.T)**2 * gauss( H_1_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_1_energy = 0.5 * H_1_velocity**2 * (hydrogen_mass) * J_to_eV
		H_2_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H.T),200))).T*(2*boltzmann_constant_J*T_H/(hydrogen_mass))**0.5).T
		H_2_velocity_PDF = (4*np.pi*(H_2_velocity.T)**2 * gauss( H_2_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H))**(3/2) , (T_H*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_2_energy = 0.5 * H_2_velocity**2 * (hydrogen_mass) * J_to_eV
		baricenter_impact_energy = ((np.zeros((200,*np.shape(T_H),200))+H_1_energy).T + H_2_energy).T
		cross_section = np.zeros_like(baricenter_impact_energy)
		cross_section[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] += 2.96*(4**4)*1e-19*1e-4/baricenter_impact_energy[np.logical_and(baricenter_impact_energy<=0.1,baricenter_impact_energy>0)] * 1e20
		cross_section[np.logical_and(baricenter_impact_energy<=1,baricenter_impact_energy>0.1)] += 2.96*(4**4)*1e-18*1e-4 * 1e20
		cross_section[baricenter_impact_energy>1] += 2.96*(4**4)*1e-18*1e-4/(baricenter_impact_energy[baricenter_impact_energy>1]**(0.4*4)) * 1e20 # multiplied 1e20
		cross_section[cross_section<0] = 0
		reaction_rate = np.nansum((cross_section*H_1_velocity*H_1_velocity_PDF).T * H_2_velocity*H_2_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_1_velocity))*np.mean(np.diff(H_2_velocity))		# m^3 / s * 1e20
		# reaction_rate = np.sum((cross_section*H_velocity*H_velocity_PDF).T * H_velocity*H_velocity_PDF )* np.mean(np.diff(H_velocity))		# m^3 / s * 1e20
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H))[0]
	H1s_H_4__H2p_e = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate
	return H1s_H_4__H2p_e	# m^-3/s *1e-20 / (nH(4)/ne) / (nH(1)/ne)



def RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
	temp = read_adf11(scdfile, 'scd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_ionisation_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited))) * (1e-6 * 1e20)  # in ionisations m^-3 s-1 / (# / m^3)**2 * 1e20
	effective_ionisation_rates = (effective_ionisation_rates * (merge_ne_prof_multipulse_interp_crop_limited**2)).astype('float')
	if flag_single_point:
		return effective_ionisation_rates[0]	# m^-3/s *1e-20 / (nH/ne)
	else:
		return effective_ionisation_rates	# m^-3/s *1e-20 / (nH/ne)

def RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited):	# [eV, 10e20 #/m3]
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
	temp = read_adf11(acdfile, 'acd', 1, 1, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())
	temp[np.isnan(temp)] = 0
	effective_recombination_rates = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited))) * (1e-6 * 1e20)  # in recombinations m^-3 s-1 / (# / m^3)**2 * 1e20
	effective_recombination_rates = (effective_recombination_rates*(merge_ne_prof_multipulse_interp_crop_limited**2)).astype('float')
	if flag_single_point:
		return effective_recombination_rates[0]	# m^-3/s *1e-20 / (nHp/ne)
	else:
		return effective_recombination_rates	# m^-3/s *1e-20 / (nHp/ne)

class MyException(Exception):
    pass

def all_RR_and_power_balance(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision,require_strongest=False,only_Yacora_as_molecule=False):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		e_H__Hm = nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0]	# m^-3/s *1e-20
		e_H2__Hm_H = nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0]
		e_Hm__e_H_e = nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0]
		Hp_Hm__Hex_H = nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		Hp_Hm__Hn_H = nHm_ne_all*RR_Hp_Hm__Hn_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		Hp_Hm__H2p_e = nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0]
		Hm_H1s__H1s_H1s_e = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0]
		Hm_H1s__H2_v_e = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0]
		Hm_H2v__H_H2v_e = nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0]
		H2p_Hm__Hex_H2 = nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		H2p_Hm__Hn_H2 = nHm_ne_all*RR_H2p_Hm__Hn_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		e_H2__e_H2p_e = nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0]
		Hp_H2v__H_H2p = nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0]
		Hp_H_H__H_H2p = nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0]
		H1s_H_2__H2p_e = nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne)
		H1s_H_3__H2p_e = nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne)
		H1s_H_4__H2p_e = nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne)
		e_H2p__XXX__e_Hp_H = nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0]
		e_H2p__H_H = nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne)[0]
		e_H2p__Hex_H__or__Hex_Hp_e = nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(Te,ne)[0]
		e_H2p__Hex_H__or__Hn_Hp_e = np.transpose(np.transpose(np.array([nH2p_ne_all.tolist()]*len(np.unique(excited_states_From_H2p))), (*np.arange(1,1+len(nH2p_ne_all.shape)),0))*RR_e_H2p__Hex_H__or__Hn_Hp_e__r(Te,ne)[0], (-1,*np.arange(len(nH2p_ne_all.shape))))	# split among hydrogen excited states
		e_H2p__e_Hp_Hp_e = nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0]
		H1s_H2pv__Hp_H_H = nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0]
		H2pvi_H2v0__Hp_H_H2v1 = nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0]
		H2pvi_H2v0__H3p_H1s = nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0]
		H2p_Hm__H2N13Λσ_H1s = nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0]
		H2p_Hm__H3p_e = nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0]
		Hp_H_H__Hp_H2 = nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(Te,np.unique(T_H),np.unique(T_Hp),ne)[0]
		H_H_H__H_H2 = (nH_ne_all**3)*RR_H_H_H__H_H2__r(Te,np.unique(T_H),ne)[0]
		e_H2__e_Hex_H = nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0]
		e_H2__e_Hn_H = np.transpose(np.transpose(np.array([nH2_ne_all.tolist()]*len(np.unique(excited_states_From_H2p))), (*np.arange(1,1+len(nH2_ne_all.shape)),0))*RR_e_H2__e_Hn_H__r(Te,ne)[0], (-1,*np.arange(len(nH2_ne_all.shape))))	# split among hydrogen excited states
		e_H2__Hp_H_2e = nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0]
		e_H2X1Σg__e_H1s_H1s = nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0]
		Hp_H2X1Σg__Hp_H1s_H1s = nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0]
		H1s_H2v__H1s_2H1s = nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0]
		H2v0_H2v__H2v0_2H1s = (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0]
		recombination = nHp_ne_all*RR_recombination__r(Te,ne)[0]
		ionisation = nH_ne_all*RR_ionisation__r(Te,ne)[0]
	else:
		e_H__Hm = nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2__Hm_H = nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		e_Hm__e_H_e = nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		Hp_Hm__Hex_H = nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		Hp_Hm__Hn_H = nHm_ne_all*RR_Hp_Hm__Hn_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		Hp_Hm__H2p_e = nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited)
		Hm_H1s__H1s_H1s_e = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited)
		Hm_H1s__H2_v_e = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited)
		Hm_H2v__H_H2v_e = nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited)
		H2p_Hm__Hex_H2 = nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		H2p_Hm__Hn_H2 = nHm_ne_all*RR_H2p_Hm__Hn_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		e_H2__e_H2p_e = nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		Hp_H2v__H_H2p = nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited)
		Hp_H_H__H_H2p = nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		H1s_H_2__H2p_e = nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		H1s_H_3__H2p_e = nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		H1s_H_4__H2p_e = nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2p__XXX__e_Hp_H = nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2p__H_H = nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2p__Hex_H__or__Hex_Hp_e = nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2p__Hex_H__or__Hn_Hp_e = nH2p_ne_all*RR_e_H2p__Hex_H__or__Hn_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)	# split among hydrogen excited states
		e_H2p__e_Hp_Hp_e = nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		H1s_H2pv__Hp_H_H = nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited)
		H2pvi_H2v0__Hp_H_H2v1 = nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited)
		H2pvi_H2v0__H3p_H1s = nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		H2p_Hm__H2N13Λσ_H1s = nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited)
		H2p_Hm__H3p_e = nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited)
		Hp_H_H__Hp_H2 = nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited)
		H_H_H__H_H2 = (nH_ne_all**3)*RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2__e_Hex_H = nH2_ne_all*RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2__e_Hn_H = nH2_ne_all*RR_e_H2__e_Hn_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)	# split among hydrogen excited states
		e_H2__Hp_H_2e = nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		e_H2X1Σg__e_H1s_H1s = nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		Hp_H2X1Σg__Hp_H1s_H1s = nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		H1s_H2v__H1s_2H1s = nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		H2v0_H2v__H2v0_2H1s = (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		recombination = nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		ionisation = nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)

		# I get e_H2p__H_H from e_H2p__Hex_H__or__Hex_Hp_e - e_H2p__XXX__e_Hp_H
		# the e_H2p__H_H rate is already calculate in AMJUEL subtracting e_H2p__H+H*__H_H1s from mthe total e_H2p__H+H
		# NO
		# in e_H2p__Hex_H__or__Hex_Hp_e I neglect the higher excited states of H, because I obtained the Yacora coefficients up to n=12 (I think).
		# Therefore I think it's best to use AMJUEL coefficients, that consider all excitation states
		# e_H2p__Hex_H__or__Hex_Hp_e it's important for the power balance, AMJUEL stuff it is for the particle one.

	# who knows how good or applicable the molecular rates are, beside Yacora ones, so I add the possibility to neglect all but those
	if only_Yacora_as_molecule:
		e_H__Hm[:] = 0	# m^-3/s *1e-20
		e_H2__Hm_H[:] = 0
		e_Hm__e_H_e[:] = 0
		# Hp_Hm__Hex_H = nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		# Hp_Hm__Hn_H = nHm_ne_all*RR_Hp_Hm__Hn_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all)
		Hp_Hm__H2p_e[:] = 0
		Hm_H1s__H1s_H1s_e[:] = 0
		Hm_H1s__H2_v_e[:] = 0
		Hm_H2v__H_H2v_e[:] = 0
		# H2p_Hm__Hex_H2 = nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		# H2p_Hm__Hn_H2 = nHm_ne_all*RR_H2p_Hm__Hn_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		e_H2__e_H2p_e[:] = 0
		Hp_H2v__H_H2p[:] = 0
		Hp_H_H__H_H2p[:] = 0
		H1s_H_2__H2p_e[:] = 0
		H1s_H_3__H2p_e[:] = 0
		H1s_H_4__H2p_e[:] = 0
		# I cannot erase the next 2 here because I use them to calculate in a fine way the potential energy contribution of e_H2p__Hex_H__or__Hn_Hp_e
		# also because they contribute to  different parts of the particle balance
		# e_H2p__XXX__e_Hp_H = nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0]
		# e_H2p__H_H = nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne)[0]
		# e_H2p__Hex_H__or__Hex_Hp_e = nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(Te,ne)[0]
		# e_H2p__Hex_H__or__Hn_Hp_e = np.transpose(np.transpose(np.array([nH2p_ne_all.tolist()]*len(np.unique(excited_states_From_H2p))), (*np.arange(1,1+len(nH2p_ne_all.shape)),0))*RR_e_H2p__Hex_H__or__Hn_Hp_e__r(Te,ne)[0], (-1,*np.arange(len(nH2p_ne_all.shape))))	# split among hydrogen excited states
		e_H2p__e_Hp_Hp_e[:] = 0
		H1s_H2pv__Hp_H_H[:] = 0
		H2pvi_H2v0__Hp_H_H2v1[:] = 0
		H2pvi_H2v0__H3p_H1s[:] = 0
		H2p_Hm__H2N13Λσ_H1s[:] = 0
		H2p_Hm__H3p_e[:] = 0
		Hp_H_H__Hp_H2[:] = 0
		H_H_H__H_H2[:] = 0
		# e_H2__e_Hex_H = nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0]
		# e_H2__e_Hn_H = nH2_ne_all*RR_e_H2__e_Hn_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)	# split among hydrogen excited states
		e_H2__Hp_H_2e[:] = 0
		e_H2X1Σg__e_H1s_H1s[:] = 0
		Hp_H2X1Σg__Hp_H1s_H1s[:] = 0
		H1s_H2v__H1s_2H1s[:] = 0
		H2v0_H2v__H2v0_2H1s[:] = 0

	e_H__Hm_sigma = molecular_precision*e_H__Hm*1e-10	# *1e-10 is only to avoid numerical overflow
	e_H2__Hm_H_sigma = molecular_precision*e_H2__Hm_H*1e-10
	e_Hm__e_H_e_sigma = molecular_precision*e_Hm__e_H_e*1e-10
	Hp_Hm__Hex_H_sigma = molecular_precision*Hp_Hm__Hex_H*1e-10
	Hp_Hm__H2p_e_sigma = molecular_precision*Hp_Hm__H2p_e*1e-10
	Hm_H1s__H1s_H1s_e_sigma = molecular_precision*Hm_H1s__H1s_H1s_e*1e-10
	Hm_H1s__H2_v_e_sigma = molecular_precision*Hm_H1s__H2_v_e*1e-10
	Hm_H2v__H_H2v_e_sigma = molecular_precision*Hm_H2v__H_H2v_e*1e-10
	H2p_Hm__Hex_H2_sigma = molecular_precision*H2p_Hm__Hex_H2*1e-10
	e_H2__e_H2p_e_sigma = molecular_precision*e_H2__e_H2p_e*1e-10
	Hp_H2v__H_H2p_sigma = molecular_precision*Hp_H2v__H_H2p*1e-10
	Hp_H_H__H_H2p_sigma = molecular_precision*Hp_H_H__H_H2p*1e-10
	H1s_H_2__H2p_e_sigma = molecular_precision*H1s_H_2__H2p_e*1e-10
	H1s_H_3__H2p_e_sigma = molecular_precision*H1s_H_3__H2p_e*1e-10
	H1s_H_4__H2p_e_sigma = molecular_precision*H1s_H_4__H2p_e*1e-10
	e_H2p__XXX__e_Hp_H_sigma = molecular_precision*e_H2p__XXX__e_Hp_H*1e-10
	e_H2p__Hex_H__or__Hex_Hp_e_sigma = molecular_precision*e_H2p__Hex_H__or__Hex_Hp_e
	e_H2p__e_Hp_Hp_e_sigma = molecular_precision*e_H2p__e_Hp_Hp_e*1e-10
	H1s_H2pv__Hp_H_H_sigma = molecular_precision*H1s_H2pv__Hp_H_H*1e-10
	H2pvi_H2v0__Hp_H_H2v1_sigma = molecular_precision*H2pvi_H2v0__Hp_H_H2v1*1e-10
	H2pvi_H2v0__H3p_H1s_sigma = molecular_precision*H2pvi_H2v0__H3p_H1s*1e-10
	H2p_Hm__H2N13Λσ_H1s_sigma = molecular_precision*H2p_Hm__H2N13Λσ_H1s*1e-10
	H2p_Hm__H3p_e_sigma = molecular_precision*H2p_Hm__H3p_e*1e-10
	Hp_H_H__Hp_H2_sigma = molecular_precision*Hp_H_H__Hp_H2*1e-10
	H_H_H__H_H2_sigma = molecular_precision*H_H_H__H_H2*1e-10
	e_H2__e_Hex_H_sigma = molecular_precision*e_H2__e_Hex_H*1e-10
	e_H2__Hp_H_2e_sigma = molecular_precision*e_H2__Hp_H_2e*1e-10
	e_H2X1Σg__e_H1s_H1s_sigma = molecular_precision*e_H2X1Σg__e_H1s_H1s*1e-10
	Hp_H2X1Σg__Hp_H1s_H1s_sigma = molecular_precision*Hp_H2X1Σg__Hp_H1s_H1s*1e-10
	H1s_H2v__H1s_2H1s_sigma = molecular_precision*H1s_H2v__H1s_2H1s*1e-10
	H2v0_H2v__H2v0_2H1s_sigma = molecular_precision*H2v0_H2v__H2v0_2H1s*1e-10
	e_H2p__H_H_sigma = molecular_precision*e_H2p__H_H*1e-10
	recombination_sigma = atomic_precision*recombination*1e-10
	ionisation_sigma = atomic_precision*ionisation*1e-10

	rate_creation_Hm = e_H__Hm + e_H2__Hm_H	# m^-3/s *1e-20
	rate_creation_Hm_sigma = (e_H__Hm_sigma**2 + e_H2__Hm_H_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_Hm = e_Hm__e_H_e + Hp_Hm__Hex_H + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + Hm_H2v__H_H2v_e + H2p_Hm__Hex_H2	# m^-3/s *1e-20
	rate_destruction_Hm_sigma = (e_Hm__e_H_e_sigma**2 + Hp_Hm__Hex_H_sigma**2 + Hp_Hm__H2p_e_sigma**2 + Hm_H1s__H1s_H1s_e_sigma**2 + Hm_H1s__H2_v_e_sigma**2 + Hm_H2v__H_H2v_e_sigma**2 + H2p_Hm__Hex_H2_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_creation_H2p = Hp_Hm__H2p_e + e_H2__e_H2p_e + Hp_H2v__H_H2p + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e	# m^-3/s *1e-20
	rate_creation_H2p_sigma = (Hp_Hm__H2p_e_sigma**2 + e_H2__e_H2p_e_sigma**2 + Hp_H2v__H_H2p_sigma**2 + Hp_H_H__H_H2p_sigma**2 + H1s_H_2__H2p_e_sigma**2 + H1s_H_3__H2p_e_sigma**2 + H1s_H_4__H2p_e_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_H2p = np.max([e_H2p__Hex_H__or__Hex_Hp_e,e_H2p__XXX__e_Hp_H + e_H2p__H_H],axis=0) + e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s + H2p_Hm__H3p_e + H2p_Hm__Hex_H2	# m^-3/s *1e-20
	rate_destruction_H2p_sigma = (np.max([e_H2p__Hex_H__or__Hex_Hp_e_sigma**2,e_H2p__XXX__e_Hp_H_sigma**2 + e_H2p__H_H_sigma**2],axis=0) + e_H2p__e_Hp_Hp_e_sigma**2 + H1s_H2pv__Hp_H_H_sigma**2 + H2pvi_H2v0__Hp_H_H2v1_sigma**2 + H2pvi_H2v0__H3p_H1s_sigma**2 + H2p_Hm__H2N13Λσ_H1s_sigma**2 + H2p_Hm__H3p_e_sigma**2 + H2p_Hm__Hex_H2_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_creation_H2 = Hm_H1s__H2_v_e + Hp_H_H__Hp_H2 + H_H_H__H_H2 + H2p_Hm__H2N13Λσ_H1s + H2p_Hm__Hex_H2	# m^-3/s *1e-20
	rate_creation_H2_sigma = (Hm_H1s__H2_v_e_sigma**2 + Hp_H_H__Hp_H2_sigma**2 + H_H_H__H_H2_sigma**2 + H2p_Hm__H2N13Λσ_H1s_sigma**2 + H2p_Hm__Hex_H2_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_H2 = e_H2__e_Hex_H + e_H2__e_H2p_e + e_H2__Hp_H_2e + e_H2__Hm_H + e_H2X1Σg__e_H1s_H1s + Hp_H2v__H_H2p + Hp_H2X1Σg__Hp_H1s_H1s + H1s_H2v__H1s_2H1s + H2v0_H2v__H2v0_2H1s + H2pvi_H2v0__H3p_H1s	# m^-3/s *1e-20
	rate_destruction_H2_sigma = (e_H2__e_Hex_H_sigma**2 + e_H2__e_H2p_e_sigma**2 + e_H2__Hp_H_2e_sigma**2 + e_H2__Hm_H_sigma**2 + e_H2X1Σg__e_H1s_H1s_sigma**2 + Hp_H2v__H_H2p_sigma**2 + Hp_H2X1Σg__Hp_H1s_H1s_sigma**2 + H1s_H2v__H1s_2H1s_sigma**2 + H2v0_H2v__H2v0_2H1s_sigma**2 + H2pvi_H2v0__H3p_H1s_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_creation_H = e_Hm__e_H_e + 2*Hp_Hm__Hex_H + Hm_H1s__H1s_H1s_e + 2*e_H2__e_Hex_H + e_H2__Hp_H_2e + e_H2__Hm_H + 2*e_H2X1Σg__e_H1s_H1s + Hp_H2v__H_H2p + Hp_H2X1Σg__Hp_H1s_H1s + Hm_H2v__H_H2v_e + 2*H1s_H2v__H1s_2H1s + 2*H2v0_H2v__H2v0_2H1s + e_H2p__XXX__e_Hp_H + H1s_H2pv__Hp_H_H + 2*e_H2p__H_H + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + H2pvi_H2v0__H3p_H1s + H2p_Hm__H2N13Λσ_H1s + H2p_Hm__Hex_H2 + recombination	# m^-3/s *1e-20
	rate_creation_H_sigma = (e_Hm__e_H_e_sigma**2 + (2*Hp_Hm__Hex_H_sigma)**2 + Hm_H1s__H1s_H1s_e_sigma**2 + (2*e_H2__e_Hex_H_sigma)**2 + e_H2__Hp_H_2e_sigma**2 + e_H2__Hm_H_sigma**2 + (2*e_H2X1Σg__e_H1s_H1s_sigma)**2 + Hp_H2v__H_H2p_sigma**2 + Hp_H2X1Σg__Hp_H1s_H1s_sigma**2 + Hm_H2v__H_H2v_e_sigma**2 + (2*H1s_H2v__H1s_2H1s_sigma)**2 + (2*H2v0_H2v__H2v0_2H1s_sigma)**2 + e_H2p__XXX__e_Hp_H_sigma**2 + H1s_H2pv__Hp_H_H_sigma**2 + (2*e_H2p__H_H_sigma)**2 + H1s_H2pv__Hp_H_H_sigma**2 + H2pvi_H2v0__Hp_H_H2v1_sigma**2 + H2pvi_H2v0__H3p_H1s_sigma**2 + H2p_Hm__H2N13Λσ_H1s_sigma**2 + H2p_Hm__Hex_H2_sigma**2 + recombination_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_H = Hm_H1s__H2_v_e + 2*Hp_H_H__Hp_H2 + 2*H_H_H__H_H2 + Hp_H_H__H_H2p + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e + e_H__Hm + ionisation	# m^-3/s *1e-20
	rate_destruction_H_sigma = (Hm_H1s__H2_v_e_sigma**2 + (2*Hp_H_H__Hp_H2_sigma)**2 + (2*H_H_H__H_H2_sigma)**2 + Hp_H_H__H_H2p_sigma**2 + H1s_H_2__H2p_e_sigma**2 + H1s_H_3__H2p_e_sigma**2 + H1s_H_4__H2p_e_sigma**2 + e_H__Hm_sigma**2 + ionisation_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_creation_Hp = e_H2__Hp_H_2e + e_H2p__XXX__e_Hp_H + 2*e_H2p__e_Hp_Hp_e + H1s_H2pv__Hp_H_H + H2pvi_H2v0__Hp_H_H2v1 + ionisation	# m^-3/s *1e-20
	rate_creation_Hp_sigma = (e_H2__Hp_H_2e_sigma**2 + e_H2p__XXX__e_Hp_H_sigma**2 + (2*e_H2p__e_Hp_Hp_e_sigma)**2 + H1s_H2pv__Hp_H_H_sigma**2 + H2pvi_H2v0__Hp_H_H2v1_sigma**2 + ionisation_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_Hp = Hp_Hm__Hex_H + Hp_Hm__H2p_e + Hp_H2v__H_H2p + Hp_H_H__H_H2p + recombination	# m^-3/s *1e-20
	rate_destruction_Hp_sigma = (Hp_Hm__Hex_H_sigma**2 + Hp_Hm__H2p_e_sigma**2 + Hp_H2v__H_H2p_sigma**2 + Hp_H_H__H_H2p_sigma**2 + recombination_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_creation_e = e_Hm__e_H_e + Hp_Hm__H2p_e + Hm_H1s__H1s_H1s_e + Hm_H1s__H2_v_e + e_H2__e_H2p_e + e_H2__Hp_H_2e + Hm_H2v__H_H2v_e + e_H2p__e_Hp_Hp_e + H2p_Hm__H3p_e + H1s_H_2__H2p_e + H1s_H_3__H2p_e + H1s_H_4__H2p_e + ionisation	# m^-3/s *1e-20
	rate_creation_e_sigma = (e_Hm__e_H_e_sigma**2 + Hp_Hm__H2p_e_sigma**2 + Hm_H1s__H1s_H1s_e_sigma**2 + Hm_H1s__H2_v_e_sigma**2 + e_H2__e_H2p_e_sigma**2 + e_H2__Hp_H_2e_sigma**2 + Hm_H2v__H_H2v_e_sigma**2 + e_H2p__e_Hp_Hp_e_sigma**2 + H2p_Hm__H3p_e_sigma**2 + H1s_H_2__H2p_e_sigma**2 + H1s_H_3__H2p_e_sigma**2 + H1s_H_4__H2p_e_sigma**2 + ionisation_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	rate_destruction_e = e_H__Hm + e_H2__Hm_H + e_H2p__H_H + recombination	# m^-3/s *1e-20
	rate_destruction_e_sigma = (e_H__Hm_sigma**2 + e_H2__Hm_H_sigma**2 + e_H2p__H_H_sigma**2 + recombination_sigma**2)**0.5 *1e10	# m^-3/s *1e-20

	# from Potential-energy curves for molecular hydrogen and its ions, Sharp, T. E., 1970
	# positive energy is energy freed by the reaction of making the element.
	H_potential_n1 = 13.6	# eV
	H_potential_n2_x = H_potential_n1/(np.unique(excited_states_From_H2p)**2)	# eV
	H_potential_n2 = H_potential_n2_x[0]	# eV
	H_potential_n3 = H_potential_n2_x[1]	# eV
	H_potential_n4 = H_potential_n2_x[2]	# eV
	Hm_potential = H_potential_n1 + 0.754	# eV	 # H- is at a LOWER energetic level than H
	H2_potential = 31.673	# eV
	H2p_potential = H2_potential - 15.425	# eV
	H3p_potential = H2_potential - 15.425 + H_potential_n1	# eV	# only guessed, I couldn't find a source
	e_potential = 0
	Hp_potential = 0
	J_to_eV = 6.242e18


	# this is energy PAID by the plasma
	# plasma heating
	e_H__Hm_energy = e_H__Hm * ((H_potential_n1 + e_potential) - (Hm_potential))	# eV m^-3/s *1e-20		# 1
	# Hp_Hm__Hex_H_energy = Hp_Hm__Hex_H * ((Hm_potential + Hp_potential) - (2*H_potential_n1))		# 2
	Hp_Hm__Hex_H_energy = np.sum(Hp_Hm__Hn_H.T * ((Hm_potential + Hp_potential) - (H_potential_n1 + H_potential_n2_x)),axis=-1).T		# 2
	Hp_Hm__H2p_e_energy = Hp_Hm__H2p_e * ((Hm_potential + Hp_potential) - (H2p_potential + e_potential))		# 3
	Hm_H1s__H2_v_e_energy = Hm_H1s__H2_v_e * ((H_potential_n1 + Hm_potential) - (H2_potential + e_potential))		# 4
	# H2p_Hm__Hex_H2_energy = H2p_Hm__Hex_H2 * ((Hm_potential + H2p_potential) - (H_potential_n1 + H2_potential))		# 5
	H2p_Hm__Hex_H2_energy = np.sum(H2p_Hm__Hn_H2.T * ((Hm_potential + H2p_potential) - (H_potential_n2_x + H2_potential)),axis=-1).T		# 5
	Hp_H_H__H_H2p_energy = Hp_H_H__H_H2p * ((2*H_potential_n1 + Hp_potential) - (H_potential_n1 + H2p_potential))		# 6
	H1s_H_3__H2p_e_energy = H1s_H_3__H2p_e * ((H_potential_n1 + H_potential_n3) - (H2p_potential + e_potential))		# 7
	H1s_H_4__H2p_e_energy = H1s_H_4__H2p_e * ((H_potential_n1 + H_potential_n4) - (H2p_potential + e_potential))		# 8
	H2pvi_H2v0__Hp_H_H2v1_energy = H2pvi_H2v0__Hp_H_H2v1 * ((H2_potential + H2p_potential) - (H_potential_n1 + H2_potential + H2p_potential))		# 9
	H2p_Hm__H2N13Λσ_H1s_energy = H2p_Hm__H2N13Λσ_H1s * ((Hm_potential + H2p_potential) - (H_potential_n1 + H2_potential))		# 10
	Hp_H_H__Hp_H2_energy = Hp_H_H__Hp_H2 * ((2*H_potential_n1 + Hp_potential) - (H2_potential + Hp_potential))		# 11
	H_H_H__H_H2_energy = H_H_H__H_H2 * ((3*H_potential_n1) - (H_potential_n1 + H2_potential))		# 12
	# e_H2p__H_H_energy = e_H2p__H_H * ((H2p_potential + e_potential) - (2*H_potential_n1))
	recombination_energy = recombination * ((e_potential + Hp_potential) - (H_potential_n1))		# 13
	# plasma cooling
	e_H2__Hm_H_energy = e_H2__Hm_H * ((H2_potential + e_potential) - (H_potential_n1 + Hm_potential))		# 21
	e_Hm__e_H_e_energy = e_Hm__e_H_e * ((Hm_potential + e_potential) - (H_potential_n1 + 2*e_potential))		# 22
	Hm_H1s__H1s_H1s_e_energy = Hm_H1s__H1s_H1s_e * ((H_potential_n1 + Hm_potential) - (2*H_potential_n1 + e_potential))		# 23
	Hm_H2v__H_H2v_e_energy = Hm_H2v__H_H2v_e * ((Hm_potential + H2_potential) - (H_potential_n1 + H2_potential + e_potential))		# 24
	e_H2__e_H2p_e_energy = e_H2__e_H2p_e * ((H2_potential + e_potential) - (H2p_potential + 2*e_potential))		# 25
	Hp_H2v__H_H2p_energy = Hp_H2v__H_H2p * ((H2_potential + Hp_potential) - (H_potential_n1 + H2p_potential))		# 26
	H1s_H_2__H2p_e_energy = H1s_H_2__H2p_e * ((H_potential_n1 + H_potential_n2) - (H2p_potential + e_potential))		# 27
	# e_H2p__XXX__e_Hp_H_energy = e_H2p__XXX__e_Hp_H * ((H2p_potential + e_potential) - (H_potential_n1 + e_potential + Hp_potential))
	e_H2p__e_Hp_Hp_e_energy = e_H2p__e_Hp_Hp_e * ((H2p_potential + e_potential) - (2*e_potential + 2*Hp_potential))		# 28
	H1s_H2pv__Hp_H_H_energy = H1s_H2pv__Hp_H_H * ((H_potential_n1 + H2p_potential) - (2*H_potential_n1 + Hp_potential))		# 29
	H2pvi_H2v0__H3p_H1s_energy = H2pvi_H2v0__H3p_H1s * ((H2_potential + H2p_potential) - (H_potential_n1 + H3p_potential))		# 30
	H2p_Hm__H3p_e_energy = H2p_Hm__H3p_e * ((Hm_potential + H2p_potential) - (H3p_potential + e_potential))		# 31
	# e_H2__e_Hex_H_energy = e_H2__e_Hex_H * ((H2_potential + e_potential) - (2*H_potential_n1 + e_potential))		# 32
	e_H2__e_Hex_H_energy = np.sum(e_H2__e_Hn_H.T * ((H2_potential + e_potential) - (H_potential_n1 + H_potential_n2_x + e_potential)),axis=-1).T		# 32
	e_H2__Hp_H_2e_energy = e_H2__Hp_H_2e * ((H2_potential + e_potential) - (H_potential_n1 + 2*e_potential + Hp_potential))		# 33
	e_H2X1Σg__e_H1s_H1s_energy = e_H2X1Σg__e_H1s_H1s * ((H2_potential + e_potential) - (2*H_potential_n1 + e_potential))		# 34
	Hp_H2X1Σg__Hp_H1s_H1s_energy = Hp_H2X1Σg__Hp_H1s_H1s * ((H2_potential + Hp_potential) - (2*H_potential_n1 + Hp_potential))		# 35
	H1s_H2v__H1s_2H1s_energy = H1s_H2v__H1s_2H1s * ((H_potential_n1 + H2_potential) - (3*H_potential_n1))		# 36
	H2v0_H2v__H2v0_2H1s_energy = H2v0_H2v__H2v0_2H1s * ((2*H2_potential) - (2*H_potential_n1 + H2_potential))		# 37
	ionisation_energy = ionisation * ((e_potential + H_potential_n1) - (2*e_potential + Hp_potential))		# 38

	# part added to evaluate the potential energy difference from e_H2p__Hex_H__or__Hn_Hp_e divided by excitation state
	e_H2__potential_consumed = np.min([e_H2p__XXX__e_Hp_H + e_H2p__H_H , np.sum(e_H2p__Hex_H__or__Hn_Hp_e,axis=0)],axis=0) * (H2p_potential + e_potential)
	e__potential_created = e_H2p__XXX__e_Hp_H * (e_potential)
	Hp__potential_created = e_H2p__XXX__e_Hp_H * (Hp_potential)
	H_1__potential_created = e_H2p__H_H + np.max([e_H2p__XXX__e_Hp_H + e_H2p__H_H - np.sum(e_H2p__Hex_H__or__Hn_Hp_e,axis=0) , np.zeros_like(e_H2p__XXX__e_Hp_H)],axis=0) * (H_potential_n1)
	H_2_n__potential_created = np.sum((e_H2p__Hex_H__or__Hn_Hp_e.T * H_potential_n2_x).T , axis=0)
	e_H2p__Hex_H__or__Hex_Hp_e_energy = e_H2__potential_consumed - (e__potential_created + Hp__potential_created + H_1__potential_created + H_2_n__potential_created)
	e_H2p__Hex_H__or__Hex_Hp_e_energy_heating = np.zeros_like(e_H2p__Hex_H__or__Hex_Hp_e_energy)
	e_H2p__Hex_H__or__Hex_Hp_e_energy_heating[e_H2p__Hex_H__or__Hex_Hp_e_energy<0] = e_H2p__Hex_H__or__Hex_Hp_e_energy[e_H2p__Hex_H__or__Hex_Hp_e_energy<0]		# 14
	e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling = np.zeros_like(e_H2p__Hex_H__or__Hex_Hp_e_energy)
	e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling[e_H2p__Hex_H__or__Hex_Hp_e_energy>0] = e_H2p__Hex_H__or__Hex_Hp_e_energy[e_H2p__Hex_H__or__Hex_Hp_e_energy>0]		# 39

	e_H__Hm_energy_sigma = molecular_precision*e_H__Hm_energy*1e-10	# *1e-10 is only to avoid numerical overflow
	e_H2__Hm_H_energy_sigma = molecular_precision*e_H2__Hm_H_energy*1e-10
	e_Hm__e_H_e_energy_sigma = molecular_precision*e_Hm__e_H_e_energy*1e-10
	Hp_Hm__Hex_H_energy_sigma = molecular_precision*Hp_Hm__Hex_H_energy*1e-10
	Hp_Hm__H2p_e_energy_sigma = molecular_precision*Hp_Hm__H2p_e_energy*1e-10
	Hm_H1s__H1s_H1s_e_energy_sigma = molecular_precision*Hm_H1s__H1s_H1s_e_energy*1e-10
	Hm_H1s__H2_v_e_energy_sigma = molecular_precision*Hm_H1s__H2_v_e_energy*1e-10
	Hm_H2v__H_H2v_e_energy_sigma = molecular_precision*Hm_H2v__H_H2v_e_energy*1e-10
	H2p_Hm__Hex_H2_energy_sigma = molecular_precision*H2p_Hm__Hex_H2_energy*1e-10
	e_H2__e_H2p_e_energy_sigma = molecular_precision*e_H2__e_H2p_e_energy*1e-10
	Hp_H2v__H_H2p_energy_sigma = molecular_precision*Hp_H2v__H_H2p_energy*1e-10
	Hp_H_H__H_H2p_energy_sigma = molecular_precision*Hp_H_H__H_H2p_energy*1e-10
	H1s_H_2__H2p_e_energy_sigma = molecular_precision*H1s_H_2__H2p_e_energy*1e-10
	H1s_H_3__H2p_e_energy_sigma = molecular_precision*H1s_H_3__H2p_e_energy*1e-10
	H1s_H_4__H2p_e_energy_sigma = molecular_precision*H1s_H_4__H2p_e_energy*1e-10
	# e_H2p__XXX__e_Hp_H_energy_sigma = molecular_precision*e_H2p__XXX__e_Hp_H_energy*1e-10
	e_H2p__e_Hp_Hp_e_energy_sigma = molecular_precision*e_H2p__e_Hp_Hp_e_energy*1e-10
	H1s_H2pv__Hp_H_H_energy_sigma = molecular_precision*H1s_H2pv__Hp_H_H_energy*1e-10
	H2pvi_H2v0__Hp_H_H2v1_energy_sigma = molecular_precision*H2pvi_H2v0__Hp_H_H2v1_energy*1e-10
	H2pvi_H2v0__H3p_H1s_energy_sigma = molecular_precision*H2pvi_H2v0__H3p_H1s_energy*1e-10
	H2p_Hm__H2N13Λσ_H1s_energy_sigma = molecular_precision*H2p_Hm__H2N13Λσ_H1s_energy*1e-10
	H2p_Hm__H3p_e_energy_sigma = molecular_precision*H2p_Hm__H3p_e_energy*1e-10
	Hp_H_H__Hp_H2_energy_sigma = molecular_precision*Hp_H_H__Hp_H2_energy*1e-10
	H_H_H__H_H2_energy_sigma = molecular_precision*H_H_H__H_H2_energy*1e-10
	e_H2__e_Hex_H_energy_sigma = molecular_precision*e_H2__e_Hex_H_energy*1e-10
	e_H2__Hp_H_2e_energy_sigma = molecular_precision*e_H2__Hp_H_2e_energy*1e-10
	e_H2X1Σg__e_H1s_H1s_energy_sigma = molecular_precision*e_H2X1Σg__e_H1s_H1s_energy*1e-10
	Hp_H2X1Σg__Hp_H1s_H1s_energy_sigma = molecular_precision*Hp_H2X1Σg__Hp_H1s_H1s_energy*1e-10
	H1s_H2v__H1s_2H1s_energy_sigma = molecular_precision*H1s_H2v__H1s_2H1s_energy*1e-10
	H2v0_H2v__H2v0_2H1s_energy_sigma = molecular_precision*H2v0_H2v__H2v0_2H1s_energy*1e-10
	# e_H2p__H_H_energy_sigma = molecular_precision*e_H2p__H_H_energy*1e-10
	# recombination_energy_sigma = atomic_precision*recombination_energy
	# ionisation_energy_sigma = atomic_precision*ionisation_energy
	e_H2p__Hex_H__or__Hex_Hp_e_energy_heating_sigma = molecular_precision*e_H2p__Hex_H__or__Hex_Hp_e_energy_heating*1e-10
	e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling_sigma = molecular_precision*e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling*1e-10

	# power_potential_mol = (e_H__Hm_energy + e_H2__Hm_H_energy + e_Hm__e_H_e_energy + Hp_Hm__Hex_H_energy + Hp_Hm__H2p_e_energy + Hm_H1s__H1s_H1s_e_energy + Hm_H1s__H2_v_e_energy + Hm_H2v__H_H2v_e_energy + H2p_Hm__Hex_H2_energy + e_H2__e_H2p_e_energy + Hp_H2v__H_H2p_energy + Hp_H_H__H_H2p_energy + H1s_H_2__H2p_e_energy + H1s_H_3__H2p_e_energy + H1s_H_4__H2p_e_energy + e_H2p__XXX__e_Hp_H_energy + e_H2p__e_Hp_Hp_e_energy + H1s_H2pv__Hp_H_H_energy + H2pvi_H2v0__Hp_H_H2v1_energy + H2pvi_H2v0__H3p_H1s_energy + H2p_Hm__H2N13Λσ_H1s_energy + H2p_Hm__H3p_e_energy + Hp_H_H__Hp_H2_energy + H_H_H__H_H2_energy + e_H2__e_Hex_H_energy + e_H2__Hp_H_2e_energy + e_H2X1Σg__e_H1s_H1s_energy + Hp_H2X1Σg__Hp_H1s_H1s_energy + H1s_H2v__H1s_2H1s_energy + H2v0_H2v__H2v0_2H1s_energy + e_H2p__H_H_energy)	# eV m^-3/s *1e-20
	power_potential_mol_plasma_heating = -(e_H__Hm_energy + Hp_Hm__Hex_H_energy + Hp_Hm__H2p_e_energy + Hm_H1s__H2_v_e_energy + H2p_Hm__Hex_H2_energy + Hp_H_H__H_H2p_energy + H1s_H_3__H2p_e_energy + H1s_H_4__H2p_e_energy + H2pvi_H2v0__Hp_H_H2v1_energy + H2p_Hm__H2N13Λσ_H1s_energy + Hp_H_H__Hp_H2_energy + H_H_H__H_H2_energy + e_H2p__Hex_H__or__Hex_Hp_e_energy_heating)	# eV m^-3/s *1e-20
	power_potential_mol_plasma_cooling = e_H2__Hm_H_energy + e_Hm__e_H_e_energy + Hm_H1s__H1s_H1s_e_energy + Hm_H2v__H_H2v_e_energy + e_H2__e_H2p_e_energy + Hp_H2v__H_H2p_energy + H1s_H_2__H2p_e_energy + e_H2p__e_Hp_Hp_e_energy + H1s_H2pv__Hp_H_H_energy + H2pvi_H2v0__H3p_H1s_energy + H2p_Hm__H3p_e_energy + e_H2__e_Hex_H_energy + e_H2__Hp_H_2e_energy + e_H2X1Σg__e_H1s_H1s_energy + Hp_H2X1Σg__Hp_H1s_H1s_energy + H1s_H2v__H1s_2H1s_energy + H2v0_H2v__H2v0_2H1s_energy + e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling	# eV m^-3/s *1e-20
	power_potential_mol = power_potential_mol_plasma_cooling - power_potential_mol_plasma_heating	# eV m^-3/s *1e-20
	power_potential_mol_sigma = (e_H__Hm_energy_sigma**2 + e_H2__Hm_H_energy_sigma**2 + e_Hm__e_H_e_energy_sigma**2 + Hp_Hm__Hex_H_energy_sigma**2 + Hp_Hm__H2p_e_energy_sigma**2 + Hm_H1s__H1s_H1s_e_energy_sigma**2 + Hm_H1s__H2_v_e_energy_sigma**2 + Hm_H2v__H_H2v_e_energy_sigma**2 + H2p_Hm__Hex_H2_energy_sigma**2 + e_H2__e_H2p_e_energy_sigma**2 + Hp_H2v__H_H2p_energy_sigma**2 + Hp_H_H__H_H2p_energy_sigma**2 + H1s_H_2__H2p_e_energy_sigma**2 + H1s_H_3__H2p_e_energy_sigma**2 + H1s_H_4__H2p_e_energy_sigma**2 + e_H2p__e_Hp_Hp_e_energy_sigma**2 + H1s_H2pv__Hp_H_H_energy_sigma**2 + H2pvi_H2v0__Hp_H_H2v1_energy_sigma**2 + H2pvi_H2v0__H3p_H1s_energy_sigma**2 + H2p_Hm__H2N13Λσ_H1s_energy_sigma**2 + H2p_Hm__H3p_e_energy_sigma**2 + Hp_H_H__Hp_H2_energy_sigma**2 + H_H_H__H_H2_energy_sigma**2 + e_H2__e_Hex_H_energy_sigma**2 + e_H2__Hp_H_2e_energy_sigma**2 + e_H2X1Σg__e_H1s_H1s_energy_sigma**2 + Hp_H2X1Σg__Hp_H1s_H1s_energy_sigma**2 + H1s_H2v__H1s_2H1s_energy_sigma**2 + H2v0_H2v__H2v0_2H1s_energy_sigma**2 + e_H2p__Hex_H__or__Hex_Hp_e_energy_heating_sigma**2 + e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling_sigma**2)**0.5 *1e10	# eV m^-3/s *1e-20
	power_potential_mol = power_potential_mol*(1e20/J_to_eV)	# J m^-3/s
	power_potential_mol_plasma_heating = power_potential_mol_plasma_heating*(1e20/J_to_eV)	# J m^-3/s
	power_potential_mol_plasma_cooling = power_potential_mol_plasma_cooling*(1e20/J_to_eV)	# J m^-3/s
	power_potential_mol_sigma = power_potential_mol_sigma*(1e20/J_to_eV)	# J m^-3/s

	if require_strongest:
		strongest_potential_mol_plasma_heating = (np.array([e_H__Hm_energy , Hp_Hm__Hex_H_energy , Hp_Hm__H2p_e_energy , Hm_H1s__H2_v_e_energy , H2p_Hm__Hex_H2_energy , Hp_H_H__H_H2p_energy , H1s_H_3__H2p_e_energy , H1s_H_4__H2p_e_energy , H2pvi_H2v0__Hp_H_H2v1_energy , H2p_Hm__H2N13Λσ_H1s_energy , Hp_H_H__Hp_H2_energy , H_H_H__H_H2_energy , recombination_energy , e_H2p__Hex_H__or__Hex_Hp_e_energy_heating]).argmin(axis=0) + 1).astype(int)
		strongest_potential_mol_plasma_cooling = (np.array([e_H2__Hm_H_energy , e_Hm__e_H_e_energy , Hm_H1s__H1s_H1s_e_energy , Hm_H2v__H_H2v_e_energy , e_H2__e_H2p_e_energy , Hp_H2v__H_H2p_energy , H1s_H_2__H2p_e_energy , e_H2p__e_Hp_Hp_e_energy , H1s_H2pv__Hp_H_H_energy , H2pvi_H2v0__H3p_H1s_energy , H2p_Hm__H3p_e_energy , e_H2__e_Hex_H_energy , e_H2__Hp_H_2e_energy , e_H2X1Σg__e_H1s_H1s_energy , Hp_H2X1Σg__Hp_H1s_H1s_energy , H1s_H2v__H1s_2H1s_energy , H2v0_H2v__H2v0_2H1s_energy , ionisation_energy , e_H2p__Hex_H__or__Hex_Hp_e_energy_cooling]).argmax(axis=0) + 21).astype(int)


		temp = np.array([e_H__Hm , Hp_Hm__Hex_H , Hp_Hm__H2p_e , Hm_H1s__H2_v_e , H2p_Hm__Hex_H2 , Hp_H_H__H_H2p , H1s_H_3__H2p_e , H1s_H_4__H2p_e , H2pvi_H2v0__Hp_H_H2v1 , H2p_Hm__H2N13Λσ_H1s , Hp_H_H__Hp_H2 , H_H_H__H_H2 , recombination , e_H2p__XXX__e_Hp_H + e_H2p__H_H , e_H2__Hm_H , e_Hm__e_H_e , Hm_H1s__H1s_H1s_e , Hm_H2v__H_H2v_e , e_H2__e_H2p_e , Hp_H2v__H_H2p , H1s_H_2__H2p_e , e_H2p__e_Hp_Hp_e , H1s_H2pv__Hp_H_H , H2pvi_H2v0__H3p_H1s , H2p_Hm__H3p_e , e_H2__e_Hex_H , e_H2__Hp_H_2e , e_H2X1Σg__e_H1s_H1s , Hp_H2X1Σg__Hp_H1s_H1s , H1s_H2v__H1s_2H1s , H2v0_H2v__H2v0_2H1s , ionisation , e_H2p__XXX__e_Hp_H + e_H2p__H_H])
		if np.nanmin(temp)<0:
			raise MyException("there is a problem in the particle balance, some reaction rate is negative!")
		strongest_rate = temp.argmax(axis=0).astype(int)
		strongest_rate[strongest_rate>=14]+=7
		strongest_rate[strongest_rate<14]+=1

		strongest_rate_description = 'plasma HEATING index: ' + '1 '+'e_H__Hm' + ', 2'+' Hp_Hm__Hex_H' + ', 3'+' Hp_Hm__H2p_e' + ', 4'+' Hm_H1s__H2_v_e' + '\n5'+' H2p_Hm__Hex_H2' + ', 6'+' Hp_H_H__H_H2p' + ', 7'+' H1s_H_3__H2p_e' + ', 8'+' H1s_H_4__H2p_e' + ', 9'+' H2pvi_H2v0__Hp_H_H2v1' + '\n10'+' H2p_Hm__H2N13Λσ_H1s' + ', 11'+' Hp_H_H__Hp_H2' + ', 12'+' H_H_H__H_H2' + ', 13'+' recombination' + ', 14'+' e_H2p__XXX__e_Hp_H + e_H2p__H_H' + ', \n' + 'plasma COOLING index: ' + ', 21'+' e_H2__Hm_H' + ', 22'+' e_Hm__e_H_e' + ', 23'+' Hm_H1s__H1s_H1s_e' + ', 24'+' Hm_H2v__H_H2v_e' + '\n25'+' e_H2__e_H2p_e' + ', 26'+' Hp_H2v__H_H2p' + ', 27'+' H1s_H_2__H2p_e' + ', 28'+' e_H2p__e_Hp_Hp_e' + ', 29'+' H1s_H2pv__Hp_H_H' + '\n30'+' H2pvi_H2v0__H3p_H1s' + ', 31'+' H2p_Hm__H3p_e' + ', 32'+' e_H2__e_Hn_H' + ', 33'+' e_H2__Hp_H_2e' + ', 34'+' e_H2X1Σg__e_H1s_H1s' + '\n35'+' Hp_H2X1Σg__Hp_H1s_H1s' + ', 36'+' H1s_H2v__H1s_2H1s' + ', 37'+' H2v0_H2v__H2v0_2H1s' + ', 38'+' ionisation' + ', 39'+' e_H2p__XXX__e_Hp_H + e_H2p__H_H'

		return rate_creation_Hm,rate_creation_Hm_sigma,rate_destruction_Hm,rate_destruction_Hm_sigma,rate_creation_H2p,rate_creation_H2p_sigma,rate_destruction_H2p,rate_destruction_H2p_sigma,rate_creation_H2,rate_creation_H2_sigma,rate_destruction_H2,rate_destruction_H2_sigma,rate_creation_H,rate_creation_H_sigma,rate_destruction_H,rate_destruction_H_sigma,rate_creation_Hp,rate_creation_Hp_sigma,rate_destruction_Hp,rate_destruction_Hp_sigma,rate_creation_e,rate_creation_e_sigma,rate_destruction_e,rate_destruction_e_sigma,power_potential_mol,power_potential_mol_sigma,power_potential_mol_plasma_heating,power_potential_mol_plasma_cooling,strongest_potential_mol_plasma_heating,strongest_potential_mol_plasma_cooling,strongest_rate,strongest_rate_description
	else:
		return rate_creation_Hm,rate_creation_Hm_sigma,rate_destruction_Hm,rate_destruction_Hm_sigma,rate_creation_H2p,rate_creation_H2p_sigma,rate_destruction_H2p,rate_destruction_H2p_sigma,rate_creation_H2,rate_creation_H2_sigma,rate_destruction_H2,rate_destruction_H2_sigma,rate_creation_H,rate_creation_H_sigma,rate_destruction_H,rate_destruction_H_sigma,rate_creation_Hp,rate_creation_Hp_sigma,rate_destruction_Hp,rate_destruction_Hp_sigma,rate_creation_e,rate_creation_e_sigma,rate_destruction_e,rate_destruction_e_sigma,power_potential_mol,power_potential_mol_sigma,power_potential_mol_plasma_heating,power_potential_mol_plasma_cooling


def RR_rate_creation_Hm(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0]
		out_sigma = ((molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0])**2)**0.5
	else:
		out = nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		out_sigma = ((molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_Hm(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		# return nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0] + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0] + nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0])**2 + (molecular_precision*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	else:
		# return nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_creation_H2p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0] + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne)
		out_sigma = ((molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0])**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne))**2)**0.5
	else:
		out = nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_H2p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	# return nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0] + nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0] + nH_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0] + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0] + nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(Te,ne)[0] + nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0] + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0] + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(Te,ne)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0])**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	else:
		out = nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__Hex_H__or__Hex_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_creation_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(Te,np.unique(T_H),np.unique(T_Hp),ne)[0] + (nH_ne_all**3)*RR_H_H_H__H_H2__r(Te,np.unique(T_H),ne)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(Te,np.unique(T_H),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*(nH_ne_all**3)*RR_H_H_H__H_H2__r(Te,np.unique(T_H),ne)[0])**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	else:
		out = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + (nH_ne_all**3)*RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all)
		out_sigma = ((molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*(nH_ne_all**3)*RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		# return nH2_ne_all*RR_e_H2v__e_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0] + nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0] + nH_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0] + (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0]
		out = nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0] + nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0] + nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0] + (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0]
		out_sigma = ((molecular_precision*nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0])**2)**0.5
	else:
		# return nH2_ne_all*RR_e_H2v__e_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		out = nH2_ne_all*RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + (nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2)
		out_sigma = ((molecular_precision*nH2_ne_all*RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_creation_H3p(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	return nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited)

def RR_rate_creation_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		# return nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0] + 2*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + 2*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + 2*nH2_ne_all*RR_e_H2v__e_H_H__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0] + 2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0] + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0] + 2*nH_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0] + 2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0] + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0] + 2*nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne)[0] + nH_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0] + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all) + nHp_ne_all*RR_recombination__r(Te,ne)[0]
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0] + 2*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + 2*nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0] + 2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0] + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0] + 2*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0] + 2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0] + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0] + 2*nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne)[0] + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0] + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0] + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all) + nHp_ne_all*RR_recombination__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0])**2 + (molecular_precision*2*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*2*nH2_ne_all*RR_e_H2__e_Hex_H__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0])**2 + (molecular_precision*2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(Te,ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(Te,np.unique(T_Hp),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0])**2 + (molecular_precision*2*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(Te,np.unique(T_H),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(Te,np.unique(T_H2),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0])**2 + (molecular_precision*2*nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne)[0])**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(Te,np.unique(T_H2),np.unique(T_H2p),ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(Te,ne)[0])**2)**0.5
	else:
		# return nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + 2*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH2_ne_all*RR_e_H2v__e_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + 2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + 2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all) + nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + 2*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH2_ne_all*RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + 2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + 2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all) + nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nH2_ne_all*RR_e_H2__e_Hex_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*2*nH2_ne_all*RR_e_H2X1Σg__e_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2X1Σg__Hp_H1s_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nH_ne_all_ground_state*nH2_ne_all*RR_H1s_H2v__H1s_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*2*(nH2_ne_all**2)*RR_H2v0_H2v__H2v0_2H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*nH2_ne_all*RR_H2pvi_H2v0__H3p_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H2N13Λσ_H1s__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*RR_H2p_Hm__Hex_H2(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nH2p_ne_all))**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + 2*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(Te,np.unique(T_H),np.unique(T_Hp),ne)[0] + 2*(nH_ne_all**3)*RR_H_H_H__H_H2__r(Te,np.unique(T_H),ne)[0] + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0] + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0] + nH_ne_all*RR_ionisation__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*2*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(Te,np.unique(T_H),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*2*(nH_ne_all**3)*RR_H_H_H__H_H2__r(Te,np.unique(T_H),ne)[0])**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0])**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0])**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(Te,ne)[0])**2)**0.5
	else:
		out = nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + 2*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + 2*(nH_ne_all**3)*RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__Hp_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*(nH_ne_all**3)*RR_H_H_H__H_H2__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_creation_Hp(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0] + 2*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0] + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne) + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0] + nH_ne_all*RR_ionisation__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(Te,ne)[0])**2 + (molecular_precision*2*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0])**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(Te,np.unique(T_H),np.unique(T_H2p),ne))**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(Te,np.unique(T_H2p),ne)[0])**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(Te,ne)[0])**2)**0.5
	else:
		out = nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + 2*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__XXX__e_Hp_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*2*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_ground_state*nH2p_ne_all*RR_H1s_H2pv__Hp_H_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*nH2p_ne_all*RR_H2pvi_H2v0__Hp_H_H2v1__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_Hp(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		# return nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*(nH_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0] + nHp_ne_all*RR_recombination__r(Te,ne)[0]
		out = nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0] + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0] + nHp_ne_all*RR_recombination__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(Te,np.unique(T_H2),np.unique(T_Hp),ne)[0])**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(Te,np.unique(T_Hp),np.unique(T_H),ne)[0])**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(Te,ne)[0])**2)**0.5
	else:
		# return nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_2_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H_3_H__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*(nH_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nHm_ne_all*RR_Hp_Hm__Hex_H(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all))**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*nH2_ne_all*RR_Hp_H2v__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_H2,T_Hp,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHp_ne_all*(nH_ne_all_ground_state**2)*RR_Hp_H_H__H_H2p__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_creation_e(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0] + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0] + nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0] + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0] + nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0] + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0] + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne) + nH_ne_all*RR_ionisation__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(Te,ne)[0])**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(Te,np.unique(T_Hp),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(Te,np.unique(T_H),np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(Te,ne)[0])**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(Te,np.unique(T_Hm),ne)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(Te,ne)[0])**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(Te,np.unique(T_Hm),np.unique(T_H2p),ne)[0])**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(Te,np.unique(T_H),ne))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(Te,np.unique(T_H),ne))**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(Te,ne)[0])**2)**0.5
	else:
		out = nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited) + nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nHm_ne_all*RR_e_Hm__e_H_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nHp_ne_all*RR_Hp_Hm__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H1s_H1s_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH_ne_all_ground_state*RR_Hm_H1s__H2_v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__e_H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hp_H_2e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nHm_ne_all*nH2_ne_all*RR_Hm_H2v__H_H2v_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__e_Hp_Hp_e__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2p_ne_all*nHm_ne_all*RR_H2p_Hm__H3p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_2*nH_ne_all_ground_state*RR_H1s_H_2__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_3*nH_ne_all_ground_state*RR_H1s_H_3__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH_ne_all_excited_state_4*nH_ne_all_ground_state*RR_H1s_H_4__H2p_e__r(merge_Te_prof_multipulse_interp_crop_limited,T_H,merge_ne_prof_multipulse_interp_crop_limited))**2 + (atomic_precision*nH_ne_all*RR_ionisation__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def RR_rate_destruction_e(merge_Te_prof_multipulse_interp_crop_limited,T_Hp,T_H,T_H2,T_Hm,T_H2p,merge_ne_prof_multipulse_interp_crop_limited,nHp_ne_all,nH_ne_all,nH2_ne_all,nHm_ne_all,nH2p_ne_all,fractional_population_states_H2,nH_ne_all_ground_state,nH_ne_all_excited_state_2,nH_ne_all_excited_state_3,nH_ne_all_excited_state_4,molecular_precision,atomic_precision):
	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		out = nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0] + nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0] + nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne) + nHp_ne_all*RR_recombination__r(Te,ne)[0]
		out_sigma = ((molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(Te,ne)[0])**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(Te,ne,fractional_population_states_H2)[0])**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__H_H__r(Te,ne))**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(Te,ne)[0])**2)**0.5
	else:
		out = nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2) + nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited) + nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited)
		out_sigma = ((molecular_precision*nH_ne_all_ground_state*RR_e_H__Hm__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (molecular_precision*nH2_ne_all*RR_e_H2__Hm_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited,fractional_population_states_H2))**2 + (molecular_precision*nH2p_ne_all*RR_e_H2p__H_H__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2 + (atomic_precision*nHp_ne_all*RR_recombination__r(merge_Te_prof_multipulse_interp_crop_limited,merge_ne_prof_multipulse_interp_crop_limited))**2)**0.5
	return out,out_sigma	# m^-3/s *1e-20

def calc_H_population_states(merge_ne_prof_multipulse_interp_crop_limited,merge_Te_prof_multipulse_interp_crop_limited,nH_ne_all,nHp_ne_all,T_Hp,T_Hm,T_H2p,nH2p_ne_all,nHm_ne_all,nH2_ne_all,nH3p_ne_all):
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	flag_single_point=False
	if np.shape(merge_Te_prof_multipulse_interp_crop_limited)==():
		flag_single_point=True
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		nH_ne_all = np.array([nH_ne_all])
		nHp_ne_all = np.array([nHp_ne_all])
		T_Hp = np.array([T_Hp])
		T_Hm = np.array([T_Hm])
		T_H2p = np.array([T_H2p])
		nH2p_ne_all = np.array([nH2p_ne_all])
		nHm_ne_all = np.array([nHm_ne_all])
		nH2_ne_all = np.array([nH2_ne_all])
		nH3p_ne_all = np.array([nH3p_ne_all])

	# NOTE  the first value of the "_full" arrays is for Lyman alpha. I use it only to estimate n=2 population density
	energy_difference_full = np.array([10.1988,1.88867, 2.54970, 2.85567, 3.02187, 3.12208, 3.18716, 3.23175, 3.26365, 3.28725, 3.30520, 3.31917])  # eV
	# Data from "Accurate Atomic Transition Probabilities for Hydrogen, Helium, and Lithium", W. L. Wiese and J. R. Fuhr 2009
	statistical_weigth_full = np.array([8, 18, 32, 50, 72, 98, 128, 162, 200, 242, 288, 338])  # gi-gk
	einstein_coeff_full = np.array([4.6986e+00, 4.4101e-01, 8.4193e-2, 2.53044e-2, 9.7320e-3, 4.3889e-3, 2.2148e-3, 1.2156e-3, 7.1225e-4, 4.3972e-4, 2.8337e-4, 1.8927e-04]) * 1e8  # 1/s

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		excitation_full = []
		for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
			if isel==0:
				temp = read_adf15(pecfile_2, 1, Te, ne*10**(20 - 6) )[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			else:
				temp = read_adf15(pecfile, isel, Te, ne*10**(20 - 6) )[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			temp[np.isnan(temp)] = 0
			temp[np.isinf(temp)] = 0
			temp = np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)*temp
			excitation_full.append(temp)
		excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
		excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
	else:
		excitation_full = []
		for isel in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
			if isel==0:
				temp = read_adf15(pecfile_2, 1, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			else:
				temp = read_adf15(pecfile, isel, merge_Te_prof_multipulse_interp_crop_limited.flatten(),(merge_ne_prof_multipulse_interp_crop_limited * 10 ** (20 - 6)).flatten())[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			temp[np.isnan(temp)] = 0
			temp[np.isinf(temp)] = 0
			temp = temp.reshape((np.shape(merge_Te_prof_multipulse_interp_crop_limited)))
			excitation_full.append(temp)
		excitation_full = np.array(excitation_full)  # in # photons cm^-3 s^-1
		excitation_full = (excitation_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		recombination_full = []
		for isel in [0, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]:
			if isel==0:
				temp = read_adf15(pecfile_2, 67, Te, ne*10**(20 - 6) )[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			else:
				temp = read_adf15(pecfile, isel, Te, ne*10**(20 - 6) )[0]  # ADAS database is in cm^3   # photons s^-1 cm^-3
			temp[np.isnan(temp)] = 0
			temp = np.ones_like(merge_Te_prof_multipulse_interp_crop_limited)*temp
			recombination_full.append(temp)
		recombination_full = np.array(recombination_full)  # in # photons cm^-3 s^-1 / (# cm^-3)**2
		recombination_full = (recombination_full.T * (10 ** -6) * (energy_difference_full / J_to_eV)).T  # in W m^-3 / (# / m^3)**2
	else:
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

	population_coefficients = ((excitation_full *  nH_ne_all).T /multiplicative_factor_full).T
	population_coefficients += ((recombination_full * nHp_ne_all).T /multiplicative_factor_full).T

	if len(np.shape(merge_Te_prof_multipulse_interp_crop_limited))==4:	# H, Hm, H2, H2p, ne, Te
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,:,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,:,0]
		T_Hp_temp = T_Hp[0,:,0]
		T_Hm_temp = T_Hm[0,:,0]
		T_H2p_temp = T_H2p[0,:,0]
		nHp_ne_all_temp = nHp_ne_all[0,:,0]
		temp1 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([Te.flatten(),T_Hp_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nHp_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		Te = merge_Te_prof_multipulse_interp_crop_limited[0,0,0]
		ne = merge_ne_prof_multipulse_interp_crop_limited[0,0,0]
		T_Hp_temp = T_Hp[0,0,0]
		T_Hm_temp = T_Hm[0,0,0]
		T_H2p_temp = T_H2p[0,0,0]
		nH2p_ne_all_temp = nH2p_ne_all[0,0,0]
		temp2 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([Te.flatten(),T_H2p_temp.flatten(),T_Hm_temp.flatten(),ne.flatten()*1e20,(nH2p_ne_all_temp*ne).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		temp2 = np.transpose([temp2.tolist()]*(np.shape(nHp_ne_all_temp)[1]) ,(1,2,0)).reshape(np.shape(temp1))
		temp = (temp1 + temp2).reshape((*np.shape(T_Hp[0,:,0]),len(np.unique(excited_states_From_Hn_with_Hp))))
		temp = np.transpose([[temp.tolist()]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[2]]*np.shape(merge_Te_prof_multipulse_interp_crop_limited)[0] ,(0,2,1,3,4))
		temp = temp.reshape((np.prod(np.shape(merge_Te_prof_multipulse_interp_crop_limited)) , len(np.unique(excited_states_From_Hn_with_Hp))))
		population_coefficients += (nHm_ne_all.flatten()*( temp ).T).reshape((np.shape(population_coefficients)))
	else:
		temp1 = From_Hn_with_Hp_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_Hp.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nHp_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_Hp))
		temp2 = From_Hn_with_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),T_H2p.flatten(),T_Hm.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20,(nH2p_ne_all*merge_ne_prof_multipulse_interp_crop_limited).flatten()*1e20]).T,np.unique(excited_states_From_Hn_with_H2p))
		population_coefficients += (nHm_ne_all.flatten()*( temp1 + temp2 ).T).reshape((np.shape(population_coefficients)))

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		temp = From_H2_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2))
		temp = np.ones((np.prod(np.shape(merge_Te_prof_multipulse_interp_crop_limited)),len(np.unique(excited_states_From_H2)))) * temp[0]
	else:
		temp = From_H2_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2))
	population_coefficients += (nH2_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		temp = From_H2p_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H2p))
		temp = np.ones((np.prod(np.shape(merge_Te_prof_multipulse_interp_crop_limited)),len(np.unique(excited_states_From_H2p)))) * temp[0]
	else:
		temp = From_H2p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H2p))
	population_coefficients += (nH2p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	if (len(np.unique(merge_Te_prof_multipulse_interp_crop_limited))==1 and len(np.unique(merge_ne_prof_multipulse_interp_crop_limited))==1):
		Te = np.unique(merge_Te_prof_multipulse_interp_crop_limited)
		ne = np.unique(merge_ne_prof_multipulse_interp_crop_limited)
		temp = From_H3p_pop_coeff_full_extra(np.array([Te,ne*1e20]).T,np.unique(excited_states_From_H3p))
		temp = np.ones((np.prod(np.shape(merge_Te_prof_multipulse_interp_crop_limited)),len(np.unique(excited_states_From_H3p)))) * temp[0]
	else:
		temp = From_H3p_pop_coeff_full_extra(np.array([merge_Te_prof_multipulse_interp_crop_limited.flatten(),merge_ne_prof_multipulse_interp_crop_limited.flatten()*1e20]).T,np.unique(excited_states_From_H3p))
	population_coefficients += (nH3p_ne_all.flatten()*temp.T).reshape((np.shape(population_coefficients)))

	population_states = population_coefficients * merge_ne_prof_multipulse_interp_crop_limited**2 * 1e20
	if flag_single_point:
		return population_states[:,0]
	else:
		return population_states.reshape((12,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))

def calc_H2_population_states(merge_ne_prof_multipulse_interp_crop_limited,T_H2,nH2_ne_all):
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	population_states_H2 = np.zeros((15,*np.shape(merge_ne_prof_multipulse_interp_crop_limited)))
	for v_index in range(15):
		if v_index==0:
			population_states_H2[0] = merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all * 1 * np.exp(0) / q_vibr
		else:
			population_states_H2[v_index] = merge_ne_prof_multipulse_interp_crop_limited*nH2_ne_all * 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	return population_states_H2
def calc_H2_fractional_population_states(T_H2):
	plank_constant_eV = 4.135667696e-15	# eV s
	plank_constant_J = 6.62607015e-34	# J s
	light_speed = 299792458	# m/s
	oscillator_frequency = 4161.166 * 1e2	# 1/m
	q_vibr = 1/(1-np.exp(-plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)))
	population_states_H2 = np.zeros((15,*np.shape(T_H2)))
	for v_index in range(15):
		if v_index==0:
			population_states_H2[0] = 1 * np.exp(0) / q_vibr
		else:
			population_states_H2[v_index] = 1 * np.exp(-(v_index)*plank_constant_eV*light_speed*oscillator_frequency/(T_H2*eV_to_K)) / q_vibr
	return population_states_H2



def H2p_H2_v_ratio_AMJUEL(ne,Te):
	# from AMJUEL
	# 12.59 Reaction 2.0c e +H2(v) →H+ 2 + ... , Ratio H+ 2 /H2
	coefficients_all = [[-5.179118614571,0.01286086917362,-0.01247224025136,0.004533679343348,-0.0008361635510932,0.00008553044625491,-0.000004999715093023,1.571098567732E-07,-2.060813085997E-09],[2.724390078109,0.02834163745797,-0.03864050364482,0.01706652099436,-0.003663038423578,0.0004202564138138,-0.00002662584829521,8.795062121815E-07,-1.182222955604E-08],[-4.38668601874,-0.02926061072808,0.02894878695435,-0.009446648200804,0.001518996106179,-0.0001265997327758,0.000005854399926391,-1.486122653306E-07,1.689012430939E-09],[2.264569877939,0.01815438259403,-0.01085596419844,0.003413100865417,-0.0005414461499166,0.0000558390402846,-0.000003494312204501,1.176151828469E-07,-1.631239647294E-09],[0.07274238145138,-0.008711469240181,0.001228482582919,-0.0002783712092681,0.00004118527168273,-0.00001416596336268,0.000001434270108351,-5.449461807998E-08,6.856555143001E-10],[-0.3332379782334,0.003737790746246,-0.0002689540529741,-0.00007626065770945,0.00003568228589947,-0.000001406315371249,-3.197342970575E-08,-4.539301068809E-10,9.975543797298E-11],[0.09526327139861,-0.001046333532366,0.0002549358736065,-0.00003059560467433,-0.000003460893323989,4.064186524949E-07,-3.342848157445E-08,2.46962240876E-09,-6.675572336239E-11],[-0.01096455316607,0.0001457992401832,-0.00005912048929304,0.00001303239138604,-0.000001086383020673,6.161003040376E-08,2.239916154599E-10,-2.455363614572E-10,7.954115052237E-12],[0.0004636081955869,-0.000007621573981592,0.000003973565261183,-0.000001046764002685,0.000000129171309722,-9.932517910526E-09,3.41248732465E-10,1.748563436447E-12,-2.514652180013E-13]]
	out = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te),np.log(ne*1e-14),coefficients_all))
	# I add this lines to avoid incorrect behaviour of the polynomial expression for low temperature (goes to infinity)
	out = np.array([out])
	Te = np.array([Te])
	out[Te<0.1]=0
	out=out[0]
	return out
def H2p_H2_v0_ratio_AMJUEL(ne,Te):
	# from AMJUEL
	# 12.58 Reaction 2.0b e +H2(v = 0) →H+ 2 + ... , Ratio H+ 2 /H2
	coefficients_all = [[-8.07333505146,0.006423193640255,-0.008948271203923,0.00458228890363,-0.001133062383784,0.0001472017794904,-0.00001037336914709,3.752297970376E-07,-5.454015313711E-09],[1.653303173229,-0.02467726829997,0.002781981866915,0.002885192609023,-0.0009313120366375,0.0001193603851194,-0.000007537317404876,2.293533267915E-07,-2.631009817091E-09],[-2.823571725913,0.04179798625064,-0.03843754761904,0.01457779901878,-0.002689882985994,0.0002781888663389,-0.00001566116716437,4.383284185329E-07,-4.647720359472E-09],[3.990452244578,0.0323496636898,0.0165243830532,-0.01096201612859,0.001873511746692,-0.0001169697988538,3.575003652413E-07,0.000000229573966706,-6.354279184897E-09],[-1.928017324234,-0.05924941119276,0.002747810096639,0.005081237207186,-0.001022266662048,0.00005028620238286,0.000001852056305253,-2.235055431776E-07,5.044313531628E-09],[0.4270719810226,0.03111963342548,-0.005375451827926,-0.001087834901368,0.0003309186604685,-0.00002068184308171,-1.431641793698E-07,4.978742338393E-08,-1.196745758163E-09],[-0.04448144242484,-0.007757664930424,0.002077957627729,-0.00004372642244363,-0.00003372497244723,0.000002250956728263,4.433372743456E-08,-7.031624330495E-09,1.525402495797E-10],[0.001689930248766,0.0009331136287171,-0.0003275780863917,0.00004092745909295,-0.000002274671467977,0.000000246396005485,-2.449746761192E-08,1.051457900024E-09,-1.574612982461E-11],[0.00001023775315217,-0.00004333393780782,0.00001830671388026,-0.000003393709830706,3.905523509303E-07,-3.774892100311E-08,2.386870809793E-09,-7.72169585831E-11,9.552566353384E-13]]
	out = np.exp(np.polynomial.polynomial.polyval2d(np.log(Te),np.log(ne*1e-14),coefficients_all))
	return out
def Hm_H2_v_ratio_AMJUEL(Te):
	# from AMJUEL
	# 11.11 Reaction 7.0a e +H2(v) →H− +H, Ratio H−/H2 from DA
	coefficients_all = [-6.001820741967,1.247273997745,-2.753387653632,0.2274419556537,0.01148400271668,0.08614331916062,-0.0348253743748,0.004822974299102,-0.0002291190247346]
	out = np.exp(np.polyval(np.flip(coefficients_all,axis=0),np.log(Te)))
	return out
def Hm_H2_v0_ratio_AMJUEL(Te):
	# from AMJUEL
	# 11.12 Reaction 7.0b e +H2(0) →H− +H, Ratio H−/H2 from DA
	coefficients_all = [-16.08434690479,2.105039374877,-2.553803267076,0.7038135447597,-0.065865842644,-0.002548302462129,0.0002922944743984,0.00008800611380131,-0.000007939105674896]
	out = np.exp(np.polyval(np.flip(coefficients_all,axis=0),np.log(Te)))
	return out

if False:# up to 06/10/2020. Ray gave me new and better data and I adapted this correlations
	TH_fit_from_simulations = interpolate.interp1d(np.log([1e-5,0.1,1.6,4]),np.log([0.99e-5,0.099,1.1,1.1]),fill_value='extrapolate')
	nH2_ne_fit_from_simulations = lambda Te: np.exp(np.polyval([0.22332556,-1.33762069,-1.11738361],np.log(Te)))
	TH2_fit_from_simulations = interpolate.interp1d(np.log([1e-5,0.1,2,4]),np.log([0.99e-5,0.099,0.3,0.3]),fill_value='extrapolate')
elif True:
	temp = np.arange(1e-2,30,0.05)
	# fit = [-0.38781087, -1.48232135,  0.23416186]
	fit = [-1.3934874, 0.02018249]	# I use this because a linear log/log relationship is representative of dilution via temperature increase, that is ultimately what happens.
	temp2 = np.exp(np.polyval(fit,np.log(temp)))
	# temp_max = np.exp(-fit[1]/(2*fit[0]))	# this is to avoid the density to decrease with decrease temperature passed the peak
	# temp2[temp<temp_max]=temp2.max()
	nH2_ne_fit_from_simulations = interpolate.interp1d(temp,temp2,fill_value='extrapolate')
	limit_H_up = interpolate.interp1d(np.log([0.1,4]),np.log([1e2,1e2]),fill_value='extrapolate')
	limit_H_up_prob = interpolate.interp1d(np.log([0.1,4]),np.log([20,20]),fill_value='extrapolate')
	# limit_H_up = interpolate.interp1d(np.log([0.1,4]),np.log([1000,1000]),fill_value='extrapolate')
	limit_H_down = interpolate.interp1d(np.log([1e-5,0.1,0.5,2,4]),np.log([0.2,0.2,0.006,0.001,0.001]),fill_value='extrapolate')
	limit_H_down_prob = interpolate.interp1d(np.log([1e-5,0.1,0.5,2,4]),np.log([0.2,0.2,0.006,0.003,0.003]),fill_value='extrapolate')
	nH_ne_fit_from_simulations = lambda Te: np.nanmax([np.exp(np.polyval([-0.15109974,-0.98294073,-2.03131663],np.log(Te))),np.exp(limit_H_down(np.log(Te)))*5],axis=0)
TH_fit_from_simulations = interpolate.interp1d(np.log([1e-5,0.1,3]),np.log([0.99e-5,0.099,1.4]),fill_value='extrapolate')
TH_low_fit_from_simulations = interpolate.interp1d(np.log([1e-5,0.1,3]),np.log([0.99e-5,0.099,0.5]),fill_value='extrapolate')
TH2_fit_from_simulations = interpolate.interp1d(np.log([1e-5,0.1,2,4]),np.log([0.99e-5,0.099,0.3,0.3]),fill_value='extrapolate')



# Here I want  the power lost from hydrogen plasma to neutrals via CX
# I need to do it in two pieces: first groud state hydrogen atoms usin AMJUEL and then for each excited state independently using Janev
def RR_Hp_H1s__H1s_Hp(T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited):
	if np.shape(T_Hp)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_ne_prof_multipulse_interp_crop_limited)==():
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
		T_Hp = np.array([T_Hp])
	# from AMJUEL
	# 1.8.5 Reaction 3.1.8R p +H(1s) →H(1s) + p
	def internal(*arg):
		T_H,T_Hp = arg[0]
		T_H = np.array([T_H])
		T_Hp = np.array([T_Hp])
		H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
		H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_energy = 0.5 * H_velocity**2 * (hydrogen_mass) * J_to_eV
		Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp.T),200))).T*(2*boltzmann_constant_J*T_Hp/(hydrogen_mass))**0.5).T
		Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hp*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		Hp_energy = 0.5 * Hp_velocity**2 * (hydrogen_mass) * J_to_eV
		bullet_impact_velocity = ((np.zeros((200,*np.shape(T_H),200))+H_velocity).T + Hp_velocity).T
		bullet_impact_energy = 0.5 * bullet_impact_velocity**2 * (hydrogen_mass) * J_to_eV
		coefficients_all = [-32.60293402651,-0.1302091929244,-0.003264584699247,-0.002837612246121,0.0002259716141071,0.0003105542152111,-0.00009613308889191,0.00001043010252591,-0.0000003944350620003]
		cross_section = np.exp(np.polyval(np.flip(coefficients_all,axis=0),np.log(bullet_impact_energy))) * 1e-4	# m^2
		cross_section[cross_section<0] = 0
		reaction_rate = np.nansum((cross_section*H_velocity*H_velocity_PDF).T * Hp_velocity*Hp_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hp_velocity))		# m^3 / s
		print(np.nansum((bullet_impact_energy*H_velocity_PDF).T * Hp_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hp_velocity)))
		reaction_rate[np.logical_not(np.isfinite(reaction_rate))]=0
		return reaction_rate
	reaction_rate = list(map(internal,(np.array([T_H.flatten(),T_Hp.flatten()]).T)))
	reaction_rate = np.reshape(reaction_rate,np.shape(T_H))
	Hp_H1s__H1s_Hp = (merge_ne_prof_multipulse_interp_crop_limited**2)*reaction_rate * 1e20
	return Hp_H1s__H1s_Hp

def P_RR_Hp_H1s__H1s_Hp(T_Hp,T_H,merge_ne_prof_multipulse_interp_crop_limited):
	# I want this to be the energy loss associated with the process, to I multimply the cross section vy the velocity and energy loss of the plasma
	if np.shape(T_Hp)!=np.shape(merge_ne_prof_multipulse_interp_crop_limited):
		print('error, Te and ne are different shapes')
		exit()
	if np.shape(merge_ne_prof_multipulse_interp_crop_limited)==():
		merge_Te_prof_multipulse_interp_crop_limited=np.array([merge_Te_prof_multipulse_interp_crop_limited])
		merge_ne_prof_multipulse_interp_crop_limited=np.array([merge_ne_prof_multipulse_interp_crop_limited])
		T_H = np.array([T_H])
		T_Hp = np.array([T_Hp])
	# from AMJUEL
	# 1.8.5 Reaction 3.1.8R p +H(1s) →H(1s) + p
	def internal(*arg):
		T_H,T_Hp = arg[0]
		T_H = np.array([T_H])
		T_Hp = np.array([T_Hp])
		H_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_H),200))).T*(2*boltzmann_constant_J*T_H.T/(hydrogen_mass))**0.5).T
		H_velocity_PDF = (4*np.pi*(H_velocity.T)**2 * gauss( H_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_H.T))**(3/2) , (T_H.T*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		H_energy = 0.5 * H_velocity**2 * (hydrogen_mass) * J_to_eV
		Hp_velocity = ((np.logspace(np.log10(0.001),np.log10(10),200)*np.ones((*np.shape(T_Hp.T),200))).T*(2*boltzmann_constant_J*T_Hp/(hydrogen_mass))**0.5).T
		Hp_velocity_PDF = (4*np.pi*(Hp_velocity.T)**2 * gauss( Hp_velocity.T, ((hydrogen_mass)/(2*np.pi*boltzmann_constant_J*T_Hp))**(3/2) , (T_Hp*boltzmann_constant_J/(hydrogen_mass))**0.5 ,0)).T
		Hp_energy = 0.5 * Hp_velocity**2 * (hydrogen_mass) * J_to_eV
		plasma_energy_loss = ((np.zeros((200,*np.shape(T_H),200))-H_energy).T + Hp_energy).T / J_to_eV
		bullet_impact_velocity = ((np.zeros((200,*np.shape(T_H),200))+H_velocity).T + Hp_velocity).T
		bullet_impact_energy = 0.5 * bullet_impact_velocity**2 * (hydrogen_mass) * J_to_eV	# eV
		coefficients_all = [-32.60293402651,-0.1302091929244,-0.003264584699247,-0.002837612246121,0.0002259716141071,0.0003105542152111,-0.00009613308889191,0.00001043010252591,-0.0000003944350620003]
		# print('START')
		# print(np.max(H_velocity))
		# print(np.max(Hp_velocity))
		# print(np.max(plasma_energy_loss))
		# print(np.max(bullet_impact_velocity))
		# print(np.max(bullet_impact_energy))
		cross_section = np.exp(np.polyval(np.flip(coefficients_all,axis=0),np.log(bullet_impact_energy))) * 1e-4	# m^2
		# print(np.max(cross_section))
		cross_section[cross_section<0] = 0
		power_rate_rate = np.nansum((cross_section*plasma_energy_loss*H_velocity*H_velocity_PDF).T * Hp_velocity*Hp_velocity_PDF ,axis=(0,-1)).T* np.mean(np.diff(H_velocity))*np.mean(np.diff(Hp_velocity))		# m^3 / s
		# print(np.max(power_rate_rate))
		power_rate_rate[np.logical_not(np.isfinite(power_rate_rate))]=0
		# print(np.max(power_rate_rate))
		return power_rate_rate
	power_rate_rate = list(map(internal,(np.array([T_H.flatten(),T_Hp.flatten()]).T)))
	# print(np.max(power_rate_rate))
	power_rate_rate = np.reshape(power_rate_rate,np.shape(merge_ne_prof_multipulse_interp_crop_limited))
	P_Hp_H1s__H1s_Hp = (merge_ne_prof_multipulse_interp_crop_limited**2)*power_rate_rate * 1e40
	return P_Hp_H1s__H1s_Hp
# I don't do the one resolved for excited state because it's almost the same and the cross section is slightly lower
# so assuming all ground state is a good upper bound

# I want to build an interpolator for the RR of destruction/creation of electrons/protons
def nH2p_nH2_values_Te_ne_2(Te,ne,to_find_steps):
	if np.shape(Te)==():
		nH2p_nH2_values = np.logspace(np.log10(max(1e-10,1e-3*min(H2p_H2_v_ratio_AMJUEL(ne*1e20,Te),H2p_H2_v0_ratio_AMJUEL(ne*1e20,Te)))),np.log10(2),num=to_find_steps)
	else:
		temp=[]
		for i in range(len(Te)):
			temp.append(np.logspace(np.log10(max(1e-10,1e-3*min(H2p_H2_v_ratio_AMJUEL(ne[i]*1e20,Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i]*1e20,Te[i])))),np.log10(2),num=to_find_steps))
		nH2p_nH2_values = np.array(temp).T
	return nH2p_nH2_values

def nHm_nH2_values_Te_2(Te,to_find_steps):
	if np.shape(Te)==():
		nHm_nH2_values = np.logspace(np.log10(max(1e-11,1e-3*min(Hm_H2_v_ratio_AMJUEL(Te),Hm_H2_v0_ratio_AMJUEL(Te)))),np.log10(2),num=to_find_steps)
	else:
		temp=[]
		for i in range(len(Te)):
			temp.append(np.logspace(np.log10(max(1e-11,1e-3*min(Hm_H2_v_ratio_AMJUEL(Te[i]),Hm_H2_v0_ratio_AMJUEL(Te[i])))),np.log10(2),num=to_find_steps))
		nHm_nH2_values = np.array(temp).T
	return nHm_nH2_values

if False:	# this seems very bad, in the sense that for low Te the concentration actually decrease
	# nH2_ne_fit_from_simulations = lambda Te: np.exp(np.polyval([0.22332556,-1.33762069,-1.11738361],np.log(Te)))
	nH2_ne_fit_from_simulations = lambda Te: np.exp(np.polyval([-0.38781087,-1.48232135,0.23416186],np.log(Te)))

# 2021/04/23 lowering the lower bound to avoid the best point being the lowest
def nH2_ne_values_Te_2(Te,H2_steps):
	H2_steps = H2_steps//2*2+1
	max_nH2_ne = nH2_ne_fit_from_simulations(Te)*1e2
	min_nH2_ne = nH2_ne_fit_from_simulations(Te)*1e-4
	nH2_ne_values = np.logspace(np.log10(min_nH2_ne),np.log10(max_nH2_ne),num=(H2_steps))
	return nH2_ne_values

# 2021/04/23 my regime is very far from SS, so I maybe should use a flat probability
def nH2_ne_log_probs_Te_2(Te,nH2_ne_values):
	nH2_ne_log_probs = np.ones_like(nH2_ne_values)
	return nH2_ne_log_probs
def nH_ne_log_probs_Te_2(Te,nH_ne_values):
	nH_ne_log_probs = np.ones_like(nH_ne_values)
	return nH_ne_log_probs


# 2020/07/13 at the moment these are not used	# 2020/07/15 let's try to
def nH_ne_values_Te(Te,H_steps):
	H_steps = H_steps//2*2+1
	max_nH_ne = np.exp(limit_H_up(np.log(Te)))
	min_nH_ne = np.exp(limit_H_down(np.log(Te)))
	centre_nH_ne = nH_ne_fit_from_simulations(Te)
	nH_ne_values = np.logspace(np.log10(min_nH_ne),np.log10(max_nH_ne),num=(H_steps-1))
	H_left_interval = nH_ne_values[nH_ne_values-centre_nH_ne<=0]
	H_right_interval = nH_ne_values[nH_ne_values-centre_nH_ne>0]
	nH_ne_values = np.array(H_left_interval.tolist() + [centre_nH_ne] + H_right_interval.tolist())
	return nH_ne_values
def nH_ne_values_Te_expanded(Te,H_steps,how_expand_nH_ne_indexes):
	H_steps = H_steps//2*2+1
	max_nH_ne = np.exp(limit_H_up(np.log(Te)))
	min_nH_ne = np.exp(limit_H_down(np.log(Te)))
	centre_nH_ne = nH_ne_fit_from_simulations(Te)
	nH_ne_values = np.logspace(np.log10(min_nH_ne),np.log10(max_nH_ne),num=(H_steps-1))
	H_left_interval = nH_ne_values[nH_ne_values-centre_nH_ne<=0]
	H_right_interval = nH_ne_values[nH_ne_values-centre_nH_ne>0]
	nH_ne_values = np.array(H_left_interval.tolist() + [centre_nH_ne] + H_right_interval.tolist())
	nH_ne_additional_values=[]
	for i in range(H_steps-1):
		if how_expand_nH_ne_indexes[i]!=0:
			nH_ne_additional_values.extend(np.logspace(np.log10(nH_ne_values[i]),np.log10(nH_ne_values[i+1]),num=how_expand_nH_ne_indexes[i]+2)[1:-1])
	return np.sort(np.concatenate((nH_ne_values,nH_ne_additional_values)))
def nH_ne_values_Te_3(Te,H_steps,ne_values_array,max_H_density_available):
	H_steps = H_steps//2*2+1
	nH_ne_values_all = []
	for ne in ne_values_array:
		max_nH_ne = min(np.exp(limit_H_up(np.log(Te))),max_H_density_available/ne)
		min_nH_ne = np.exp(limit_H_down(np.log(Te)))
		centre_nH_ne = nH_ne_fit_from_simulations(Te)
		nH_ne_values = np.logspace(np.log10(min_nH_ne),np.log10(max_nH_ne),num=(H_steps-1))
		H_left_interval = nH_ne_values[nH_ne_values-centre_nH_ne<=0]
		H_right_interval = nH_ne_values[nH_ne_values-centre_nH_ne>0]
		nH_ne_values_all.append(np.array(H_left_interval.tolist() + [centre_nH_ne] + H_right_interval.tolist()))
	return np.array(nH_ne_values_all)
def nH_ne_values_Te_expanded_3(Te,H_steps,how_expand_nH_ne_indexes,ne_values_array,max_H_density_available):
	H_steps = H_steps//2*2+1
	nH_ne_values_all = []
	for ne in ne_values_array:
		max_nH_ne = min(np.exp(limit_H_up(np.log(Te))),max_H_density_available/ne)
		min_nH_ne = np.exp(limit_H_down(np.log(Te)))
		centre_nH_ne = nH_ne_fit_from_simulations(Te)
		nH_ne_values = np.logspace(np.log10(min_nH_ne),np.log10(max_nH_ne),num=(H_steps-1))
		H_left_interval = nH_ne_values[nH_ne_values-centre_nH_ne<=0]
		H_right_interval = nH_ne_values[nH_ne_values-centre_nH_ne>0]
		nH_ne_values = np.array(H_left_interval.tolist() + [centre_nH_ne] + H_right_interval.tolist())
		nH_ne_additional_values=[]
		for i in range(H_steps-1):
			if how_expand_nH_ne_indexes[i]!=0:
				nH_ne_additional_values.extend(np.logspace(np.log10(nH_ne_values[i]),np.log10(nH_ne_values[i+1]),num=how_expand_nH_ne_indexes[i]+2)[1:-1])
		nH_ne_values_all.append(np.sort(np.concatenate((nH_ne_values,nH_ne_additional_values))))
	return np.array(nH_ne_values_all)
def nH_ne_log_probs_Te(Te,nH_ne_values):
	max_nH_ne = np.exp(limit_H_up(np.log(Te)))
	min_nH_ne = np.exp(limit_H_down(np.log(Te)))
	centre_nH_ne = nH_ne_fit_from_simulations(Te)
	H_left_interval = nH_ne_values[nH_ne_values-centre_nH_ne<=0]
	H_right_interval = nH_ne_values[nH_ne_values-centre_nH_ne>0]
	H_left_interval_log_probs = -(0.5*((np.log10(H_left_interval/centre_nH_ne)/(np.log10(centre_nH_ne/min_nH_ne)/2))**2))**1	# super gaussian order 1, centre_nH_ne-min_nH_ne is 2 sigma
	H_right_interval_log_probs = -(0.5*((np.log10(H_right_interval/centre_nH_ne)/(np.log10(centre_nH_ne/max_nH_ne)/3))**2))**2	# super gaussian order 2, centre_nH_ne-max_nH_ne is 3 sigma
	nH_ne_log_probs = np.array(H_left_interval_log_probs.tolist() + H_right_interval_log_probs.tolist())
	return nH_ne_log_probs
def nH_ne_log_probs_Te_3(Te,nH_ne_values):
	max_nH_ne = np.exp(limit_H_up_prob(np.log(Te)))
	min_nH_ne = np.exp(limit_H_down_prob(np.log(Te)))
	centre_nH_ne = nH_ne_fit_from_simulations(Te)
	nH_ne_log_probs = np.zeros_like(nH_ne_values)
	nH_ne_log_probs[nH_ne_values>=centre_nH_ne] = -(0.5*((np.log10(nH_ne_values[nH_ne_values>=centre_nH_ne]/centre_nH_ne)/(np.log10(centre_nH_ne/max_nH_ne)/3))**2))**2	# super gaussian order 2, centre_nH_ne-max_nH_ne is 3 sigma
	nH_ne_log_probs[nH_ne_values<centre_nH_ne] = -(0.5*((np.log10(nH_ne_values[nH_ne_values<centre_nH_ne]/centre_nH_ne)/(np.log10(centre_nH_ne/min_nH_ne)/2))**2))**1	# super gaussian order 1, centre_nH_ne-min_nH_ne is 2 sigma
	return nH_ne_log_probs
def nH2_ne_values_Te(Te,H2_steps):
	H2_steps = H2_steps//2*2+1
	max_nH2_ne = nH2_ne_fit_from_simulations(Te)*20
	min_nH2_ne = nH2_ne_fit_from_simulations(Te)/20
	centre_nH2_ne = nH2_ne_fit_from_simulations(Te)
	H2_left_interval = np.logspace(np.log10(min_nH2_ne),np.log10(centre_nH2_ne),num=(H2_steps+1)/2)[:-1]
	H2_right_interval = np.logspace(np.log10(centre_nH2_ne),np.log10(max_nH2_ne),num=(H2_steps+1)/2)[1:]
	nH2_ne_values = np.array(H2_left_interval.tolist() + [centre_nH2_ne] + H2_right_interval.tolist())
	return nH2_ne_values
def nH2_ne_log_probs_Te(Te,nH2_ne_values):
	max_nH2_ne = np.max(nH2_ne_fit_from_simulations(Te))*10
	min_nH2_ne = np.min(nH2_ne_fit_from_simulations(Te))/20
	centre_nH2_ne = nH2_ne_fit_from_simulations(Te)
	H2_left_interval = nH2_ne_values[nH2_ne_values-centre_nH2_ne<=0]
	H2_right_interval = nH2_ne_values[nH2_ne_values-centre_nH2_ne>0]
	H2_left_interval_log_probs = -0.5*((np.log10(H2_left_interval/centre_nH2_ne)/(np.log10(centre_nH2_ne/min_nH2_ne)/2))**2)	# centre_nH2_ne-min_nH2_ne is 2 sigma
	H2_right_interval_log_probs = -0.5*((np.log10(H2_right_interval/centre_nH2_ne)/(np.log10(centre_nH2_ne/max_nH2_ne)/2))**2)	# centre_nH2_ne-max_nH2_ne is 2 sigma
	nH2_ne_log_probs = np.array(H2_left_interval_log_probs.tolist() + H2_right_interval_log_probs.tolist())
	# nH2_ne_log_probs = nH2_ne_log_probs -np.log(np.sum(np.exp(nH2_ne_log_probs)))	# normalisation for logarithmic probabilities
	return nH2_ne_log_probs
def nH2_ne_log_probs_Te_3(Te,nH2_ne_values):
	max_nH2_ne = np.max(nH2_ne_fit_from_simulations(Te))*10
	min_nH2_ne = np.min(nH2_ne_fit_from_simulations(Te))/20
	centre_nH2_ne = nH2_ne_fit_from_simulations(Te)
	nH2_ne_log_probs = np.zeros_like(nH2_ne_values)
	nH2_ne_log_probs[nH2_ne_values>=centre_nH2_ne] = -(0.5*((np.log10(nH2_ne_values[nH2_ne_values>=centre_nH2_ne]/centre_nH2_ne)/(np.log10(centre_nH2_ne/max_nH2_ne)/2))**2))**1	# super gaussian order 1, centre_nH_ne-max_nH_ne is 2 sigma
	nH2_ne_log_probs[nH2_ne_values<centre_nH2_ne] = -(0.5*((np.log10(nH2_ne_values[nH2_ne_values<centre_nH2_ne]/centre_nH2_ne)/(np.log10(centre_nH2_ne/min_nH2_ne)/2))**2))**1	# super gaussian order 1, centre_nH_ne-min_nH_ne is 2 sigma
	return nH2_ne_log_probs

# 11/04/2021 I change the upper limit from 1 to 100 because in reality H2 is consumed way more than replenished fron the sides, so it's realistic that the balance soule be very squed towards ions vs H2
def nH2p_nH2_values_Te_ne(Te,ne,to_find_steps,H2_suppression=False):
	additional_low_multiplier = 1
	if H2_suppression:
		additional_low_multiplier = 0.1
	if np.shape(Te)==():
		# nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-2*min(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),np.log10(1/additional_low_multiplier*max(1,min(1,1e4*max(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te))))),num=to_find_steps)
		nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-8*min(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),np.log10(1/additional_low_multiplier*max(1e2,1e5*max(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),num=to_find_steps)
	else:
		temp=[]
		for i in range(len(Te)):
			# temp.append(np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-2*min(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),np.log10(1/additional_low_multiplier*max(1,min(1,1e4*max(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i]))))),num=to_find_steps))
			temp.append(np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-8*min(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),np.log10(1/additional_low_multiplier*max(1e2,1e5*max(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),num=to_find_steps))
		nH2p_nH2_values = np.array(temp).T
	return nH2p_nH2_values
def nH2p_nH2_values_Te_ne_expanded(Te,ne,to_find_steps,how_expand_nH2p_nH2_indexes,H2_suppression=False):
	additional_low_multiplier = 1
	if H2_suppression:
		additional_low_multiplier = 0.1
	if np.shape(Te)==():
		# nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-2*min(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),np.log10(1/additional_low_multiplier*max(1,min(1,1e4*max(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te))))),num=to_find_steps)
		nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-8*min(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),np.log10(1/additional_low_multiplier*max(1e2,1e5*max(H2p_H2_v_ratio_AMJUEL(ne,Te),H2p_H2_v0_ratio_AMJUEL(ne,Te)))),num=to_find_steps)
		nH2p_nH2_additional_values=[]
		for i in range(to_find_steps-1):
			if how_expand_nH2p_nH2_indexes[i]!=0:
				nH2p_nH2_additional_values.extend(np.logspace(np.log10(nH2p_nH2_values[i]),np.log10(nH2p_nH2_values[i+1]),num=how_expand_nH2p_nH2_indexes[i]+2)[1:-1])
		nH2p_nH2_values = np.sort(np.concatenate((nH2p_nH2_values,nH2p_nH2_additional_values)))
	else:
		temp=[]
		for i in range(len(Te)):
			# nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-2*min(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),np.log10(1/additional_low_multiplier*max(1,min(1,1e4*max(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i]))))),num=to_find_steps)
			nH2p_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-8*min(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),np.log10(1/additional_low_multiplier*max(1e2,1e5*max(H2p_H2_v_ratio_AMJUEL(ne[i],Te[i]),H2p_H2_v0_ratio_AMJUEL(ne[i],Te[i])))),num=to_find_steps)
			nH2p_nH2_additional_values=[]
			for i in range(to_find_steps-1):
				if how_expand_nH2p_nH2_indexes[i]!=0:
					nH2p_nH2_additional_values.extend(np.logspace(np.log10(nH2p_nH2_values[i]),np.log10(nH2p_nH2_values[i+1]),num=how_expand_nH2p_nH2_indexes[i]+2)[1:-1])
			temp.append(np.sort(np.concatenate((nH2p_nH2_values,nH2p_nH2_additional_values))))
		nH2p_nH2_values = np.array(temp).T
	return nH2p_nH2_values
def nHm_nH2_values_Te(Te,to_find_steps,H2_suppression=False):
	additional_low_multiplier = 1
	if H2_suppression:
		additional_low_multiplier = 0.1
	if np.shape(Te)==():
		nHm_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-4*min(Hm_H2_v_ratio_AMJUEL(Te),Hm_H2_v0_ratio_AMJUEL(Te)))),np.log10(1/additional_low_multiplier*max(1e1,1e5*max(Hm_H2_v_ratio_AMJUEL(Te),Hm_H2_v0_ratio_AMJUEL(Te)))),num=to_find_steps)
	else:
		temp=[]
		for i in range(len(Te)):
			temp.append(np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-4*min(Hm_H2_v_ratio_AMJUEL(Te[i]),Hm_H2_v0_ratio_AMJUEL(Te[i])))),np.log10(1/additional_low_multiplier*max(1e1,1e5*max(Hm_H2_v_ratio_AMJUEL(Te[i]),Hm_H2_v0_ratio_AMJUEL(Te[i])))),num=to_find_steps))
		nHm_nH2_values = np.array(temp).T
	return nHm_nH2_values
def nHm_nH2_values_Te_expanded(Te,to_find_steps,how_expand_nHm_nH2_indexes,H2_suppression=False):
	additional_low_multiplier = 1
	if H2_suppression:
		additional_low_multiplier = 0.1
	if np.shape(Te)==():
		nHm_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-5*min(Hm_H2_v_ratio_AMJUEL(Te),Hm_H2_v0_ratio_AMJUEL(Te)))),np.log10(1/additional_low_multiplier*max(1e1,1e5*max(Hm_H2_v_ratio_AMJUEL(Te),Hm_H2_v0_ratio_AMJUEL(Te)))),num=to_find_steps)
		nHm_nH2_additional_values = []
		for i in range(to_find_steps-1):
			if how_expand_nHm_nH2_indexes[i]!=0:
				nHm_nH2_additional_values.extend(np.logspace(np.log10(nHm_nH2_values[i]),np.log10(nHm_nH2_values[i+1]),num=how_expand_nHm_nH2_indexes[i]+2)[1:-1])
		nHm_nH2_values = np.sort(np.concatenate((nHm_nH2_values,nHm_nH2_additional_values)))
	else:
		temp=[]
		for i in range(len(Te)):
			nHm_nH2_values = np.logspace(np.log10(additional_low_multiplier*max(1e-9,1e-5*min(Hm_H2_v_ratio_AMJUEL(Te[i]),Hm_H2_v0_ratio_AMJUEL(Te[i])))),np.log10(1/additional_low_multiplier*max(1e1,1e5*max(Hm_H2_v_ratio_AMJUEL(Te[i]),Hm_H2_v0_ratio_AMJUEL(Te[i])))),num=to_find_steps)
			nHm_nH2_additional_values=[]
			for i in range(to_find_steps-1):
				if how_expand_nHm_nH2_indexes[i]!=0:
					nHm_nH2_additional_values.extend(np.logspace(np.log10(nHm_nH2_values[i]),np.log10(nHm_nH2_values[i+1]),num=how_expand_nHm_nH2_indexes[i]+2)[1:-1])
			temp.append(np.sort(np.concatenate((nHm_nH2_values,nHm_nH2_additional_values))))
		nHm_nH2_values = np.array(temp).T
	return nHm_nH2_values



if False:
	mkl.set_num_threads(number_cpu_available)
	# H, Hm, H2, H2p, ne, Te
	samples_Te_array = np.logspace(np.log10(0.01),np.log10(15),num=40,dtype=np.float32)
	samples_T_Hp = samples_Te_array/eV_to_K	# K
	samples_T_Hp[samples_T_Hp<300]=300
	samples_T_Hm = np.exp(TH2_fit_from_simulations(np.log(samples_Te_array)))/eV_to_K	# K
	samples_T_Hm[samples_T_Hm<300]=300
	samples_T_H2p = np.exp(TH2_fit_from_simulations(np.log(samples_Te_array)))/eV_to_K	# K
	samples_T_H2p[samples_T_H2p<300]=300
	samples_T_H2 = np.exp(TH2_fit_from_simulations(np.log(samples_Te_array)))/eV_to_K	# K
	samples_T_H2[samples_T_H2<300]=300
	samples_T_H = np.exp(TH_fit_from_simulations(np.log(samples_Te_array)))/eV_to_K	# K
	samples_T_H[samples_T_H<300]=300
	samples_ne_array = np.logspace(np.log10(0.01),np.log10(70),num=40,dtype=np.float32)
	samples_nH2p_nH2_array = np.logspace(np.log10(1e-10),np.log10(2),num=20,dtype=np.float32)	# 40
	samples_nH2_ne_array = np.logspace(np.log10(0.005),np.log10(5000),num=30,dtype=np.float32)
	samples_nHm_nH2_array = np.logspace(np.log10(1e-11),np.log10(2),num=20,dtype=np.float32)	# 40
	samples_nH_ne_array = np.logspace(np.log10(1e-3),np.log10(40),num=30,dtype=np.float32)	# 40
	# samples_ne = np.array([[[[(np.ones((len(samples_Te_array),len(samples_ne_array)))*samples_ne_array).T]*len(samples_nH2p_nH2)]*len(samples_nH2_ne)]*len(samples_nHm_nH2)]*len(samples_nH_ne))
	# samples_Te = np.array([[[[np.ones((len(samples_ne_array),len(samples_Te_array)))*samples_Te_array]*len(samples_nH2p_nH2)]*len(samples_nH2_ne)]*len(samples_nHm_nH2)]*len(samples_nH_ne))
	samples_nH2p_nH2 = np.array([[[samples_nH2p_nH2_array]*len(samples_nH2_ne_array)]*len(samples_nHm_nH2_array)]*len(samples_nH_ne_array))
	samples_nH2_ne = np.array([[(np.ones((len(samples_nH2_ne_array),len(samples_nH2p_nH2_array))).T * samples_nH2_ne_array).T]*len(samples_nHm_nH2_array)]*len(samples_nH_ne_array))
	samples_nHm_nH2 = np.array([(np.ones((len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array))).T * samples_nHm_nH2_array).T]*len(samples_nH_ne_array))
	samples_nHm_ne = samples_nHm_nH2 * samples_nH2_ne
	samples_nH2p_ne = samples_nH2p_nH2 * samples_nH2_ne
	samples_nHp_ne = 1 - samples_nH2p_ne + samples_nHm_ne
	# samples_nHp_ne_negative = samples_nHp_ne<=0
	samples_nHp_ne[samples_nHp_ne<=0] = 1e-20
	samples_nH_ne = (np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array))).T * samples_nH_ne_array).T
	net_rate_Hp_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	net_rate_e_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	net_rate_Hm_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	net_rate_H2p_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	net_rate_H2_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	net_rate_H_destruction = np.inf*np.ones((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	sum_nH_excited_states = np.zeros((len(samples_nH_ne_array),len(samples_nHm_nH2_array),len(samples_nH2_ne_array),len(samples_nH2p_nH2_array),len(samples_ne_array),len(samples_Te_array)),dtype=np.float32)
	for i6 in range(len(samples_Te_array)):
		# class out:
		# 	def __init__(self, net_rate_Hp_destruction,net_rate_e_destruction,nH_vs_population_state_ok):
		# 		self.net_rate_Hp_destruction = net_rate_Hp_destruction
		# 		self.net_rate_e_destruction = net_rate_e_destruction
		# 		self.nH_vs_population_state_ok = nH_vs_population_state_ok
		#
		# def calc(i6):
		# 	population_states = calc_H_population_states(np.ones_like(samples_nH2_ne)*samples_ne_array[i5],np.ones_like(samples_nH2_ne)*samples_Te_array[i6],samples_nH_ne,samples_nHp_ne,np.ones_like(samples_nH2_ne)*samples_T_Hp[i6],np.ones_like(samples_nH2_ne)*samples_T_Hm[i6],np.ones_like(samples_nH2_ne)*samples_T_H2p[i6],samples_nH2p_ne,samples_nHm_ne,samples_nH2_ne,np.zeros_like(samples_nH2p_ne))
		# 	nH_vs_population_state_ok = samples_ne_array[i5] *samples_nH_ne > np.sum(population_states,axis=0)
		# 	arguments = (np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_Te_array[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_T_Hp[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_T_H[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_T_H2[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_T_Hm[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_T_H2p[i6],np.ones_like(samples_nH2_ne)[nH_vs_population_state_ok]*samples_ne_array[i5],samples_nHp_ne[nH_vs_population_state_ok],samples_nH_ne[nH_vs_population_state_ok],samples_nH2_ne[nH_vs_population_state_ok],samples_nHm_ne[nH_vs_population_state_ok],samples_nH2p_ne[nH_vs_population_state_ok])
		# 	temp = RR_rate_destruction_Hp(*arguments,population_states=population_states[:,nH_vs_population_state_ok])
		# 	temp -=	RR_rate_creation_Hp(*arguments,population_states=population_states[:,nH_vs_population_state_ok])
		# 	net_rate_Hp_destruction = temp
		# 	temp = RR_rate_destruction_e(*arguments,population_states=population_states[:,nH_vs_population_state_ok])
		# 	temp -=	RR_rate_creation_e(*arguments,population_states=population_states[:,nH_vs_population_state_ok])
		# 	net_rate_e_destruction = temp
		# 	print((i5,i6))
		# 	# this rates are in #/m^3 /s *1e-20
		# 	return out(i5,i6,nH_vs_population_state_ok)
		#
		# out = set(map(calc,range(len(samples_Te_array))))
		fractional_population_states_H2 = calc_H2_fractional_population_states(samples_T_H2p[i6])
		for i5 in range(len(samples_ne_array)):
			population_states = calc_H_population_states(np.ones_like(samples_nH2_ne)*samples_ne_array[i5],np.ones_like(samples_nH2_ne)*samples_Te_array[i6],samples_nH_ne,samples_nHp_ne,np.ones_like(samples_nH2_ne)*samples_T_Hp[i6],np.ones_like(samples_nH2_ne)*samples_T_Hm[i6],np.ones_like(samples_nH2_ne)*samples_T_H2p[i6],samples_nH2p_ne,samples_nHm_ne,samples_nH2_ne,np.zeros_like(samples_nH2p_ne))
			# population_states_H2 = samples_ne_array[i5]*((np.array([samples_nH2_ne]*len(calc_H2_fractional_population_states(samples_T_H2p[i6]))).T)*calc_H2_fractional_population_states(samples_T_H2p[i6])).T
			sum_nH_excited_states[:,:,:,:,i5,i6] = np.sum(population_states,axis=0)
			nH_vs_population_state_ok = samples_ne_array[i5] *samples_nH_ne > np.sum(population_states,axis=0)
			nH_all_ground_state = samples_ne_array[i5] *samples_nH_ne - np.sum(population_states,axis=0)
			nH_all_ground_state[nH_all_ground_state<0]=0
			molecular_precision = 1
			atomic_precision = 0.2
			arguments = (np.ones_like(samples_nH2_ne)*samples_Te_array[i6],np.ones_like(samples_nH2_ne)*samples_T_Hp[i6],np.ones_like(samples_nH2_ne)*samples_T_H[i6],np.ones_like(samples_nH2_ne)*samples_T_H2[i6],np.ones_like(samples_nH2_ne)*samples_T_Hm[i6],np.ones_like(samples_nH2_ne)*samples_T_H2p[i6],np.ones_like(samples_nH2_ne)*samples_ne_array[i5],samples_nHp_ne,samples_nH_ne,samples_nH2_ne,samples_nHm_ne,samples_nH2p_ne,fractional_population_states_H2,nH_all_ground_state,molecular_precision,atomic_precision)
			temp = RR_rate_destruction_Hp(*arguments,population_states)[0]
			temp -=	RR_rate_creation_Hp(*arguments,population_states)[0]
			net_rate_Hp_destruction[:,:,:,:,i5,i6] = temp
			temp = RR_rate_destruction_e(*arguments,population_states)[0]
			temp -=	RR_rate_creation_e(*arguments,population_states)[0]
			net_rate_e_destruction[:,:,:,:,i5,i6] = temp
			temp = RR_rate_destruction_Hm(*arguments,population_states)[0]
			temp -=	RR_rate_creation_Hm(*arguments,population_states)[0]
			net_rate_Hm_destruction[:,:,:,:,i5,i6] = temp
			temp = RR_rate_destruction_H2p(*arguments,population_states)[0]
			temp -=	RR_rate_creation_H2p(*arguments,population_states)[0]
			net_rate_H2p_destruction[:,:,:,:,i5,i6] = temp
			temp = RR_rate_destruction_H2(*arguments,population_states)[0]
			temp -=	RR_rate_creation_H2(*arguments,population_states)[0]
			net_rate_H2_destruction[:,:,:,:,i5,i6] = temp
			temp = RR_rate_destruction_H(*arguments,population_states)[0]
			temp -=	RR_rate_creation_H(*arguments,population_states)[0]
			net_rate_H_destruction[:,:,:,:,i5,i6] = temp
			# net_rate_Hp_destruction[nH_vs_population_state_ok,i5,i6],net_rate_e_destruction[nH_vs_population_state_ok,i5,i6] = calc(arguments,population_states[:,nH_vs_population_state_ok])
			# this rates are in #/m^3 /s *1e-20
			print((i5,i6))

	np.savez_compressed('/home/ffederic/work/Collaboratory/test/experimental_data/functions/MolRad_Yacora/Yacora_FF/' +'interpolator_particle_balance',net_rate_Hp_destruction=net_rate_Hp_destruction,net_rate_e_destruction=net_rate_e_destruction,net_rate_Hm_destruction=net_rate_Hm_destruction,net_rate_H2p_destruction=net_rate_H2p_destruction,net_rate_H2_destruction=net_rate_H2_destruction,net_rate_H_destruction=net_rate_H_destruction,sum_nH_excited_states=sum_nH_excited_states,samples_nH_ne_array=samples_nH_ne_array,samples_nHm_nH2_array=samples_nHm_nH2_array,samples_nH2_ne_array=samples_nH2_ne_array,samples_nH2p_nH2_array=samples_nH2p_nH2_array,samples_ne_array=samples_ne_array,samples_Te_array=samples_Te_array)

	print('end imported')
	exit()
