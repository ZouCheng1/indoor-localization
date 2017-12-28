from __future__ import print_function
import sys, os
import argparse

import numpy
import numpy.ma
parser = argparse.ArgumentParser(description='room_parameters')
parser.add_argument('-r', '--room', type=str,
			default='test_rig',
			help='room the image was taken in; must be in rooms/')
args = parser.parse_args()
room = __import__('rooms.' + args.room, fromlist=[1,])
def cround(x, base):
	return base * round(float(x)/base)
def pick_freq (centers,gray_image,radius,number_of_transmitters,estimated_frequencies):
    centers = numpy.array(centers)
    positions_of_lights = centers
    radii_of_lights = radius
    frequencies_of_lights = estimated_frequencies
    image_shape = gray_image.shape
    center_point = tuple([p /2 for p in image_shape])  
    positions_of_lights[:,0] = center_point[0] - positions_of_lights[:,0]  
    positions_of_lights[:,1] = center_point[1] - positions_of_lights[:,1]
    last_f = None  
    min_freq_diff = 10000   
    for f in sorted(room.transmitters):   
        if last_f is None:   
            last_f = f   
            continue  
        min_freq_diff = min(min_freq_diff, f-last_f)   
        last_f = f   
    del(last_f)
    actual_frequencies = [
			 cround(f, min_freq_diff)
			 for f in frequencies_of_lights
			 ]  
    del(frequencies_of_lights) 
    del(min_freq_diff)
    to_del = []  
    for i in range(len(actual_frequencies)):   
        if actual_frequencies[i] not in room.transmitters:    
            to_del.append(i)
    positions_of_lights = numpy.delete(positions_of_lights, to_del, axis=0)  
    radii_of_lights = numpy.delete(radii_of_lights, to_del) 
    actual_frequencies = numpy.delete(actual_frequencies, to_del)
    to_del = []
    
    
    dups = {}
    for i in range(len(actual_frequencies)):
        if actual_frequencies[i] in dups:
            to_del.append(i)
            l1 = positions_of_lights[i]
            l2 = positions_of_lights[dups[actual_frequencies[i]]]
            if (l1[0] - l2[0])**2 + (l1[0] - l2[0])**2 > 100**2:
                to_del.append(dups[actual_frequencies[i]])
        else:
            dups[actual_frequencies[i]] = i
    positions_of_lights = numpy.delete(positions_of_lights, to_del, axis=0)
    radii_of_lights = numpy.delete(radii_of_lights, to_del)
    actual_frequencies = numpy.delete(actual_frequencies, to_del)
    
    lights = [
			(
				positions_of_lights[i],
				room.transmitters[actual_frequencies[i]]
			) for i in range(len(positions_of_lights))]
    return lights
    
    
    
    
    
    
    