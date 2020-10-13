import numpy as np

overwrite_everything = [True, True, True, True,True]
# this is for, in order:
# ... merge_tot.npz'
# ... composed_array.npy' and ...new_timesteps
# ... binned_data.npy'
# ... all_fits.npy'
# ... gain_scaled_composed_array.npy'
# if nothing is specified [False, False, False, False, False] is used


type_of_sensitivity = 11
# type of imputs are: 4, 5, 7, 8, 9:
# 4 = high int time scaling down with no LOS discretization
# 5 = high int time scaling down with LOS discretization
# 7 = low int time scaling up with no LOS discretization and profile smoothing
# 8 = low int time profile smoothing
# 9 = like 8 but with a new function to correct for the minimum signal.
# 11 = like 9 but with the new functions to correct minimum signal.
# if nothing is specified 5 is used

perform_convolution = True
# This is to select if you want to do a convolution of the sensitivity with a constant (20 pixels) gaussian width, or a width based on the line shape fitted

# for merge_ID_target in np.flip([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41],axis=0):#, 40,42,43,44,45,46,47,48,49,50,51,52,54]:#,84]:
# for merge_ID_target in np.flip([66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],axis=0):
# for merge_ID_target in np.flip([36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52],axis=0):
# for merge_ID_target in np.flip([40,41,42,43,44,45,46,47,48,49,50,51,52,54],axis=0):
# for merge_ID_target in [86,87,88,89,90,91,92, 93, 94, 95, 96, 97, 98]:
# for merge_ID_target in [86,87,88,89,90,91,92]:
for merge_ID_target in [93, 94, 95]:
# for merge_ID_target in [95]:
# for merge_ID_target in [85]:
	# merge_time_window=[-10,10]
	merge_time_window=[-1,2]
	# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray.py").read())
	# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray_4_points.py").read())
	exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray_4_points_neg_sig_after.py").read())
	# % reset - f
