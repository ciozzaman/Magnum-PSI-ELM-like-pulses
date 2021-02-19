import os,sys
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import mkl
mkl.set_num_threads(1)
number_cpu_available = 4	#cpu_count()
import numpy as np

# overwrite_everything = [False, False, False, False, False]
overwrite_everything = [True, True, True, True, True]
# this is for, in order:
# ... merge_tot.npz'
# ... composed_array.npy' and ...new_timesteps
# ... binned_data.npy'
# ... all_fits.npy'
# ... gain_scaled_composed_array.npy'
# if nothing is specified [False, False, False, False, False] is used


type_of_sensitivity = 12
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

# merge_ID_target_all = np.flip([86,87,88,90,91,92,93,94,95,96,97,98,99,89,85],axis=0)
merge_ID_target_all = [76,75,77,78,79]
# for merge_ID_target in np.flip([17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41],axis=0):#, 40,42,43,44,45,46,47,48,49,50,51,52,54]:#,84]:
# for merge_ID_target in np.flip([66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84],axis=0):
# for merge_ID_target in np.flip([36,37,38,39,41,42,43,44,45,46,47,48,49,50,51,52],axis=0):
# for merge_ID_target in np.flip([40,41,42,43,44,45,46,47,48,49,50,51,52,54],axis=0):
# for merge_ID_target in [86,87,88,89,90,91,92, 93, 94, 95, 96, 97, 98]:
# for merge_ID_target in np.flip([851,86,87,88,89,90,91,92,93,94,95],axis=0):
# for merge_ID_target in np.flip([99,85,97,89,92, 93,95,94],axis=0):
# for merge_ID_target in np.flip([90,91,89,99,85,97],axis=0):
# for merge_ID_target in [89]:
for merge_ID_target in merge_ID_target_all:
# for merge_ID_target in np.flip([96,97,98,89,85,86,87,88,90,91,92],axis=0):
# for merge_ID_target in np.flip([75,76],axis=0):
# for merge_ID_target in [85,95]:
	# if (merge_ID_target>=93 and merge_ID_target<100):
	# 	merge_time_window = [-1,2]
	# else:
	# 	merge_time_window = [-10,10]
	# merge_time_window=[-10,10]
	print('All that will be done: ' + str(merge_ID_target_all))
	merge_time_window=[-1,2]
	# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray.py").read())
	# exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray_4_points.py").read())
	if __name__ == '__main__':
		exec(open("/home/ffederic/work/Collaboratory/test/experimental_data/image_process_no_xarray_4_points_neg_sig_after.py").read())
	# % reset - f
