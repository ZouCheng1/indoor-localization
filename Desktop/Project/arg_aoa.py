# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy
import numpy.ma
import scipy
import scipy.misc
import scipy.ndimage
import scipy.signal
import scipy.cluster
#import matplotlib
#import matplotlib.mlab
#import pylab
#import pretty_logger
def get_Z_offset_guess(room):
	# Assume we are ~2.5 m away in Z
	DEFAULT_OFFSET = 2.5

	if room.units == 'm':
		offset = DEFAULT_OFFSET
	elif room.units == 'cm':
		offset = DEFAULT_OFFSET * 100
	elif room.units == 'in':
		offset = (DEFAULT_OFFSET * 100) / 2.54
	else:
		raise NotImplementedError('Unknown unit: {}'.format(room.units))

	if room.user_is == 'below':
		pass
	elif room.user_is == 'above':
		offset = -offset
	else:
		raise NotImplementedError('Unkonwn user position: {}'.format(room.user_is))

	return offset
def aoa(room, lights, Zf=2950, k_init_method= 'YS_brute', actual_location=None):
    centers = numpy.array(list(zip(*lights))[0])
    transmitters = numpy.array(list(zip(*lights))[1])
    centers = numpy.append(centers, Zf * numpy.ones((len(lights), 1)), axis=1)
    image_squared_distance = numpy.sum(numpy.square(centers), axis=1)
    pair_shape = (len(lights)-1, len(lights))
    transmitter_pair_squared_distance = numpy.zeros(pair_shape)
    pairwise_image_inner_products     = numpy.zeros(pair_shape)
    for i in range(len(lights)-1):
        for j in range(i+1, len(lights)):
            transmitter_pair_squared_distance[i][j] =\
            numpy.sum(numpy.square(transmitters[i] - transmitters[j]))
            pairwise_image_inner_products[i][j]=\
            numpy.dot(centers[i], centers[j])
    def least_squares_scaling_factors(k_vals):
        errs = []
        for i in range(len(lights)-1):
            for j in range(i+1, len(lights)):
                errs.append(
                        k_vals[i]**2 * image_squared_distance[i] +\
						      k_vals[j]**2 * image_squared_distance[j] -\
						      2*k_vals[i]*k_vals[j] * pairwise_image_inner_products[i][j] -\
						      transmitter_pair_squared_distance[i][j]
						      )
        return errs
    def scalar_scaling(k_vals):
        errs = numpy.array(least_squares_scaling_factors(k_vals))
        return numpy.sum(errs)
    def sol_guess_subset(index, var_cnt, sol_guess):
        sol_guess_sub = numpy.array([sol_guess[0,0]])
        for i in range(1,len(lights)):
            if sol_guess[i, 1] < 0:
                sol_guess_sub = numpy.append(sol_guess_sub, sol_guess[i, int((index%(2**var_cnt))/2**(var_cnt-1))])
                var_cnt -= 1
            else:
                sol_guess_sub = numpy.append(sol_guess_sub, sol_guess[i, 0])
        return sol_guess_sub
    def brute_force_k():
        number_of_iteration = 500
        k0_vals = numpy.linspace(-0.1, -0.01, number_of_iteration)
        err_history = []
        idx_history = []
        k_vals = numpy.array([])
        for j in range(number_of_iteration+1):
            if (j==number_of_iteration):
                min_error_history_idx = err_history.index(min(err_history))
                min_idx = idx_history[min_error_history_idx]
                k0_val = k0_vals[min_idx]
            else:
                k0_val = k0_vals[j]
            sol_guess = numpy.array([[k0_val, 0]])
            sol_found = 1
            multiple_sol = 0
            for i in range(1, len(lights)):
                sol = numpy.roots([image_squared_distance[i], -2*sol_guess[0,0]*pairwise_image_inner_products[0,i], (sol_guess[0,0]**2*image_squared_distance[0]-transmitter_pair_squared_distance[0, i])]);
                if numpy.isreal(sol)[0]:
                    if (sol[0] < 0) and (sol[1] < 0):
                        sol_guess = numpy.append(sol_guess, [sol], axis=0)
                        multiple_sol += 1
                    elif sol[0] < 0:
                        sol_guess = numpy.append(sol_guess, numpy.array([[sol[0], 0]]), axis=0)
                    elif sol[1] < 0:
                        sol_guess = numpy.append(sol_guess, numpy.array([[sol[1], 0]]), axis=0)
                    else:
                        sol_found = 0
                        break
                else:
                    sol_found = 0
                    break
            if sol_found:
                scaling_factors_error_combination = []
                for m in range(1, 2**multiple_sol+1):
                    sol_guess_combination = sol_guess_subset(m, multiple_sol, sol_guess)
                    scaling_factors_error_arr = least_squares_scaling_factors(sol_guess_combination)
                    scaling_factors_error = 0
                    for n in scaling_factors_error_arr:
                        scaling_factors_error += n**2
                    scaling_factors_error_combination.append(scaling_factors_error)
                k_vals = sol_guess_subset(numpy.argmin(scaling_factors_error_combination)+1, multiple_sol, sol_guess)
                err_history.append(min(scaling_factors_error_combination))
                idx_history.append(j)
        return k_vals
        
    if k_init_method == 'static':
        k_val_from_z = get_Z_offset_guess(room) / Zf
        k_vals_init= [k_val_from_z] * len(lights)
    elif k_init_method == 'YS_brute':
        k_vals = brute_force_k()
    elif k_init_method == 'scipy_brute':
        k_ranges = [slice(.01, .1, (.1-.01)/10) for i in range(len(lights))]
        k_vals_init = scipy.optimize.brute(scalar_scaling, k_ranges, disp=True)
    elif k_init_method == 'scipy_basin':
        k_vals_init = [-.05] * len(lights)
        res = scipy.optimize.basinhopping(scalar_scaling, k_vals_init,
				T=1e12,
				stepsize=0.01,
				)
        k_vals_init = res.x
    elif k_init_method == 'actual':
        assert actual_location is not None
        k_vals_init = k_vals_actual
    else:
        k_vals, ier = scipy.optimize.leastsq(least_squares_scaling_factors, k_vals_init)
    def least_squares_rx_location(rx_location):
        dists = []
        for i in range(len(lights)):
            dists.append(
					numpy.sum(numpy.square(rx_location - transmitters[i])) -\
					k_vals[i]**2 * image_squared_distance[i]
					)
        return dists
    def initial_position_guess(room, transmitters):
        guess = numpy.mean(transmitters, axis=0)[0]
        offset = get_Z_offset_guess(room)
        guess[2] = guess[2] - offset
        return guess
    if k_init_method == 'actual':
        rx_location_init = actual_location
    else:
        rx_location_init = initial_position_guess(room, transmitters)
        rx_location, ier = scipy.optimize.leastsq(least_squares_rx_location, rx_location_init)
    def least_squares_rotation(rotation):
        rotation = rotation.reshape((3,3))
        r = transmitters.T - rotation.dot(absolute_centers) - rx_location.reshape(3,1)
        r = numpy.square(r)
        r = r.flatten()
        return r
    absolute_centers = centers.T * numpy.vstack([k_vals, k_vals, k_vals])
    rx_rotation_init = numpy.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rx_rotation, ier = scipy.optimize.leastsq(least_squares_rotation, rx_rotation_init)
    rx_rotation = numpy.array(rx_rotation).reshape((3,3))
    rx_location_error = numpy.sum(numpy.abs(
			numpy.sqrt(numpy.sum(numpy.square(rx_location - transmitters[i]))) -\
			numpy.sqrt(k_vals[i]**2 * image_squared_distance[i])
			))
    return (rx_location, rx_rotation, rx_location_error)
