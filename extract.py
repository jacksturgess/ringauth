#python extract.py [userID(s)] [param(s)] [gestureset] [device(s)] [featureset(s)]
# - userID(s) (may be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - gestureset: p (payment only), d (door only), or a (all)
# - device(s): a (all) or list of any of {door, ring, watch}
# - featureset(s): integer (type of features to be extracted; may be a list)
#
#opens the file <userdir>/2-filtered/payment-<device>/<param>-gestures.csv or <userdir>/2-filtered/door-<device>/gestures.csv and reduces each gesture to a set of values (i.e., extracts features)
#outputs <userdir>/<featureset>-extracted-payment-<device>/<param>-features.csv or <userdir>/<featureset>-extracted-door-<device>/<param>-features.csv
#where:
# <featureset> = 3 for watchauth-style features

import csv, math, os, re, statistics, sys
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

userIDs = re.split(',', (sys.argv[1]).lower())
params = re.split(',', (sys.argv[2]).lower())
gestureset = (sys.argv[3]).lower()
devices = ['door', 'ring', 'watch'] if 'a' == (sys.argv[4]).lower() else re.split(',', sys.argv[4])
featuresets = re.split(',', sys.argv[5])

#configs
maxprewindowsize = 4

#consts
g_index = 0 #gesture name and index
s_index = 1 #sensor name
t_index = 2 #gesture timestamp (in seconds, range from prewindowsize to 0)
x_index = 3 #x-value
y_index = 4 #y-value
z_index = 5 #z-value
w_index = 6 #w-value (GRV only)
e_unf_index = 7 #Euclidean norm of unfiltered values
e_fil_index = 8 #Euclidean norm of filtered values

#values
f_names = [] #container to hold the names of the features
sensors = [] #container to hold the sensors used by the dataset
data_all = []
data_acc = []
data_gyr = []
data_grv = []
data_lac = []

def tidy_userIDs():
	global userIDs
	t_userIDs = []
	for userID in userIDs:	
		if 'user' in userID:
			t_userIDs.append('user' + f'{int(userID[4:]):03}')
		else:
			t_userIDs.append('user' + f'{int(userID):03}')
	t_userIDs.sort(reverse = False)
	userIDs = t_userIDs

def tidy_params():
	global params
	t_params = []
	for param in params:
		windowsize = 0
		offset = 0
		if 'o' in param:
			if 'om' in param:
				param = re.sub('om', 'o-', param)
			windowsize = float(param[1:param.index('o')])
			offset = float(param[param.index('o') + 1:])
		else:
			windowsize = float(param[1:])
		if windowsize + offset > maxprewindowsize:
			windowsize = maxprewindowsize - offset
		windowsize = str('%.1f' % windowsize)
		offset = str('%.1f' % offset)
		if 'f' == param[0]:
			t_params.append('f' + windowsize + 'o' + offset)
		elif 'u' == param[0]:
			t_params.append('u' + windowsize + 'o' + offset)
		else:
			sys.exit('ERROR: param not valid: ' + param)
	params = t_params

def rewrite_param(param):
	t_param = param
	if 'o-' in t_param:
		t_param = re.sub('o-', 'om', t_param)
	return t_param

def write_f_name(s):
	if 'Acc' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('Acc-' + dimension + s)
	if 'Gyr' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('Gyr-' + dimension + s)
	if 'GRV' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'w-']:
			f_names.append('GRV-' + dimension + s)
	if 'LAc' in sensors:
		for dimension in ['x-', 'y-', 'z-', 'e_unf-', 'e_fil-']:
			f_names.append('LAc-' + dimension + s)

def feature_min(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('min')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % min(datum)))
	return ','.join(f)

def feature_max(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('max')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % max(datum)))
	return ','.join(f)

def feature_mean(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('mean')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.mean(datum)))
	return ','.join(f)

def feature_med(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('med')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % statistics.median(datum)))
	return ','.join(f)
	
def feature_stdev(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('stdev')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.std(datum, ddof = 1)))
	return ','.join(f)

def feature_var(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('var')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % np.var(datum, ddof = 1)))
	return ','.join(f)

def feature_iqr(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('iqr')
	
	f = []
	for datum in g_data:
		q75, q25 = np.percentile(datum, [75, 25])
		f.append(str('%.6f' % (q75 - q25)))
	return ','.join(f)

def feature_kurt(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('kurt')
	
	f = []
	for datum in g_data:
		f.append(str('%.6f' % kurtosis(datum)))
	return ','.join(f)

def feature_skew(is_firstparse, g_data):
	if is_firstparse:
		write_f_name('skew')
	f = []
	for datum in g_data:
		f.append(str('%.6f' % skew(datum)))
	return ','.join(f)

def feature_pkcount(is_firstparse, g_data, threshold):
	if is_firstparse:
		write_f_name('pkcount' + str(threshold))
	f = []
	for datum in g_data:
		f.append(str('%.0f' % len(find_peaks(datum, prominence = threshold)[0])))
	return ','.join(f)

def feature_velo_disp(is_firstparse):
	f = []
	for sensor in ['Acc', 'Gyr', 'LAc']:
		if sensor in sensors:
			s_data = []
			if 'Acc' == sensor:
				s_data = list(data_acc)
			elif 'Gyr' == sensor:
				s_data = list(data_gyr)
			elif 'LAc' == sensor:
				s_data = list(data_lac)
			
			vx = [0]
			dx = [0]
			vy = [0]
			dy = [0]
			vz = [0]
			dz = [0]
			d = [0]
			n = len(s_data) - 1 #number of samples
			dt = float((s_data[n][0] - s_data[0][0]) / n) #sample interval
			for j in range(n):
				vx.append(vx[j] + (s_data[j][1] + s_data[j + 1][1]) / 2 * dt / 10)
				dx.append(dx[j] + vx[j + 1] * dt / 10)
				vy.append(vy[j] + (s_data[j][2] + s_data[j + 1][2]) / 2 * dt / 10)
				dy.append(dy[j] + vy[j + 1] * dt / 10)
				vz.append(vz[j] + (s_data[j][3] + s_data[j + 1][3]) / 2 * dt / 10)
				dz.append(dz[j] + vz[j + 1] * dt / 10)
				d.append(math.sqrt(dx[j] * dx[j] + dy[j] * dy[j] + dz[j] * dz[j]))
			vx.pop(0)
			vy.pop(0)
			vz.pop(0)
			
			if is_firstparse:
				f_names.append(sensor + '-x-velomean')
				f_names.append(sensor + '-y-velomean')
				f_names.append(sensor + '-z-velomean')
				f_names.append(sensor + '-x-velomax')
				f_names.append(sensor + '-y-velomax')
				f_names.append(sensor + '-z-velomax')
				f_names.append(sensor + '-x-disp')
				f_names.append(sensor + '-y-disp')
				f_names.append(sensor + '-z-disp')
				f_names.append(sensor + '-disptotal')
			
			f.append(str('%.6f' % (sum(vx) / len(vx))))
			f.append(str('%.6f' % (sum(vy) / len(vy))))
			f.append(str('%.6f' % (sum(vz) / len(vz))))
			f.append(str('%.6f' % max(vx, key = abs)))
			f.append(str('%.6f' % max(vy, key = abs)))
			f.append(str('%.6f' % max(vz, key = abs)))
			f.append(str('%.6f' % dx[len(dx) - 1]))
			f.append(str('%.6f' % dy[len(dy) - 1]))
			f.append(str('%.6f' % dz[len(dz) - 1]))
			f.append(str('%.6f' % d[len(d) - 1]))
	return ','.join(f)

def extractFeatures(is_firstparse, is_filter):
	#prepare data
	g_data = []
	if 'Acc' in sensors:
		g_data.append([row[1] for row in data_acc])
		g_data.append([row[2] for row in data_acc])
		g_data.append([row[3] for row in data_acc])
		g_data.append([row[4] for row in data_acc])
		if is_filter:
			g_data.append([row[5] for row in data_acc])
	if 'Gyr' in sensors:
		g_data.append([row[1] for row in data_gyr])
		g_data.append([row[2] for row in data_gyr])
		g_data.append([row[3] for row in data_gyr])
		g_data.append([row[4] for row in data_gyr])
		if is_filter:
			g_data.append([row[5] for row in data_gyr])
	if 'GRV' in sensors:
		g_data.append([row[1] for row in data_grv])
		g_data.append([row[2] for row in data_grv])
		g_data.append([row[3] for row in data_grv])
		g_data.append([row[4] for row in data_grv])
	if 'LAc' in sensors:
		g_data.append([row[1] for row in data_lac])
		g_data.append([row[2] for row in data_lac])
		g_data.append([row[3] for row in data_lac])
		g_data.append([row[4] for row in data_lac])
		if is_filter:
			g_data.append([row[5] for row in data_lac])
	
	#call features for this gesture
	f_data = []
	f_data.append(feature_min(is_firstparse, g_data))
	f_data.append(feature_max(is_firstparse, g_data))
	f_data.append(feature_mean(is_firstparse, g_data))
	f_data.append(feature_med(is_firstparse, g_data))
	f_data.append(feature_stdev(is_firstparse, g_data))
	f_data.append(feature_var(is_firstparse, g_data))
	f_data.append(feature_iqr(is_firstparse, g_data))
	f_data.append(feature_kurt(is_firstparse, g_data))
	f_data.append(feature_skew(is_firstparse, g_data))
	f_data.append(feature_pkcount(is_firstparse, g_data, 0.5))
	f_data.append(feature_velo_disp(is_firstparse))
	return ','.join(f_data)

def extractFeatureset3(is_filter, f_outfilename):
	global f_names, sensors
	
	#build list of gesture indices
	gestureindices = []
	for datum in data_all:
		if datum[g_index] not in gestureindices:
			gestureindices.append(datum[g_index])
	
	#for each gesture index: grab its data, extract its features, and write them
	is_firstparse = True
	f_names.clear()
	sensors.clear()
	for gestureindex in gestureindices:
		data_acc.clear()
		data_gyr.clear()
		data_grv.clear()
		data_lac.clear()
		
		#get gesture data
		for datum in data_all:
			if datum[g_index] == gestureindex:
				s = datum[s_index]
				d = []
				d.append(float(datum[t_index]))
				d.append(float(datum[x_index]))
				d.append(float(datum[y_index]))
				d.append(float(datum[z_index]))
				if 'GRV' == s:
					d.append(float(datum[w_index]))
				else:
					d.append(float(datum[e_unf_index]))
					if is_filter:
						d.append(float(datum[e_fil_index]))
				
				if 'Acc' == s:
					data_acc.append(d)	
				elif 'Gyr' == s:
					data_gyr.append(d)
				elif 'GRV' == s:
					data_grv.append(d)
				elif 'LAc' == s:
					data_lac.append(d)
				
				if len(data_acc) > 0:
					sensors.append('Acc')
				if len(data_gyr) > 0:
					sensors.append('Gyr')
				if len(data_grv) > 0:
					sensors.append('GRV')
				if len(data_lac) > 0:
					sensors.append('LAc')
		
		#extract features
		f_output = extractFeatures(is_firstparse, is_filter)
		
		#output features to the combined file
		f_outfile = open(f_outfilename, 'a')
		if is_firstparse:
			if not is_filter:
				f_names = list(filter(lambda a: not 'e_fil-' in a, f_names)) #removes every occurrence containing 'e_fil-'
			f_outfile.write('GESTURE,' + ','.join(f_names))
			is_firstparse = False
		f_outfile.write('\n' + gestureindex + ',' + f_output)
		f_outfile.close()
	print('OUTPUT: ' + f_outfilename)

if __name__ == '__main__':
	tidy_userIDs()
	tidy_params()
	
	for userID in userIDs:
		hand = ''
		if os.path.exists(userID + '-left/0-raw/'):
			hand = 'left'
		elif os.path.exists(userID + '-right/0-raw/'):
			hand = 'right'
		else:
			sys.exit('ERROR: no such sourcedir for user: ' + userID)
		
		userdir = 'C:/temp/processed_data/ringauth/' + userID + '-' + hand
		if not os.path.exists(userdir):
			sys.exit('ERROR: no such userdir for user: ' + userdir)
		cleandir = userdir + '/1-cleaned/'
		if not os.path.exists(cleandir):
			sys.exit('ERROR: no such cleandir for user: ' + cleandir)
		filterdir = userdir + '/2-filtered/'
		if not os.path.exists(filterdir):
			sys.exit('ERROR: no such filterdir for user: ' + filterdir)
		
		gesturesets = []
		if 'a' == gestureset:
			gesturesets.extend(['payment', 'door'])
		elif 'p' == gestureset:
			gesturesets.append('payment')
		elif 'd' == gestureset:
			gesturesets.append('door')
		else:
			sys.exit('ERROR: gestureset not valid: ' + gestureset)
		
		print("EXTRACTING FEATURES: " + userID)
		
		for gs in gesturesets:
			for device in devices:
				for fs in featuresets:
					params_gs = [0]
					if 'payment' == gs:
						params_gs = list(params)
					
					for param in params_gs:
						extractdir = ''
						f_infile = ''
						f_outfilename = ''
						is_filter = ''
						if 'payment' == gs:
							extractdir = userdir + '/' + fs + '-extracted-payment-' + device + '/'
							f_infile = filterdir + 'payment-' + device + '/' + rewrite_param(param) + '-gestures.csv'
							f_outfilename = extractdir + rewrite_param(param) + '-features.csv'
							is_filter = True if 'f' == param[0] else False
						elif 'door' == gs:
							extractdir = userdir + '/' + fs + '-extracted-door-' + device + '/'
							f_infile = filterdir + 'door-' + device + '/gestures.csv'
							f_outfilename = extractdir + 'features.csv'
							is_filter = True
						
						data_all.clear()
						
						if os.path.exists(f_infile):
							if not os.path.exists(extractdir):
								os.mkdir(extractdir)
							
							if os.path.exists(f_outfilename):
								os.remove(f_outfilename)
							
							with open(f_infile, 'r') as f:
								data_all = list(csv.reader(f)) #returns a list of lists (each line is a list)
								data_all.pop(0) #removes the column headers
								
								#watchauth-style features
								if '3' == fs:
									extractFeatureset3(is_filter, f_outfilename)