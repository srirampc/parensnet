import numpy as np

from ..types import NDFloatArray, NPDType

def block_bins(data: NDFloatArray, dtype: NPDType) -> NDFloatArray:
	unique_data = np.unique(data)
	unique_data = np.sort(unique_data)

	n = unique_data.size # Number of observations

	edges = np.zeros(n+1, dtype=dtype)
	edges[0] = unique_data[0]
	edges[1:-1] = 0.5*(unique_data[:-1] + unique_data[1:])
	edges[-1] = unique_data[-1]
	block_length = unique_data[-1] - edges

	if len(unique_data) == len(data):
		nn_vec = np.ones(len(data), dtype=data.dtype)
	else:
		ctx = {x:0 for x in unique_data}
		for x in data:
			ctx[x] += 1
		nn_vec = np.array([ctx[x] for x in unique_data], dtype=data.dtype) 

	count_vec = np.zeros(n, dtype=data.dtype)
	best = np.zeros(n, dtype=data.dtype)
	last = np.zeros(n, dtype=np.int64)

	for K in range(1, n+1):
		cindex = K-1
		widths = block_length[:K] - block_length[K]
		count_vec[:K] += nn_vec[cindex]

		# Fitness function (eq. 19 from Scargle 2012)
		fit_vec = count_vec[:K] * np.log(count_vec[:K] / widths)
		# Prior (eq. 21 from Scargle 2012)
		fit_vec -= 4 - np.log(73.53 * 0.05 * (K**(-0.478)))
		fit_vec[1:] += best[:cindex]

		i_max = np.argmax(fit_vec)
		last[cindex] = i_max
		best[cindex] = fit_vec[i_max]

	change_points = np.zeros(n, dtype=np.int64)
	i_cp = n
	ind = n
	while True:
		i_cp -= 1
		change_points[i_cp] = ind
		if ind == 0:
			break
		ind = last[ind-1]
	change_points = change_points[i_cp:]
	
	return edges[change_points]
