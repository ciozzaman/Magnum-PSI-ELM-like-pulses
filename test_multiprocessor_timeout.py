import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import time as tm
import numpy as np
from scipy.special import factorial

def worker(x, y, z):
	a = 1
	while a<factorial(y):
		a+=1
	# tm.sleep(y)
	return y

def collectMyResult(result):
	print("Got result {}".format(result))
	return result

timeout_ext = 5

# def abortable_worker(func, *arg, **kwargs):
# 	timeout = kwargs.get('timeout', None)
def abortable_worker(func, *arg):
	timeout = timeout_ext
	p = ThreadPool(1)
	res = p.apply_async(func, args=arg)
	print('starting ' + str(arg[1]))
	start = tm.time()
	try:
		out = res.get(timeout)  # Wait timeout seconds for func to complete.
		return out
	except multiprocessing.TimeoutError:
		print(str(arg[1]) + " Aborting due to timeout in " + str(tm.time() - start))
		return tm.time() - start

if __name__ == "__main__":
	start = tm.time()
	pool = multiprocessing.Pool(5,maxtasksperchild=1)
	featureClass = [[1000,k,1] for k in range(1,30,1)] #list of arguments
	results = []
	for f in featureClass:
	  abortable_func = partial(abortable_worker, worker)
	  results.append(pool.apply_async(abortable_func, args=f))#,callback=collectMyResult))
	pool.close()
	pool.join()
	print('finished in '+str(tm.time() - start) + ' s')
