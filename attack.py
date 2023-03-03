#python attack.py [basedir] [userIDs] [param(s)] [mode(s)] [sensors] [gesture(s)] [gestureset] [device(s)] [featureset(s)] [poollimit] [is_combinedmodel(s)]
# - basedir
# - userIDs (must be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - mode(s): naive (zero effort, victim-attacker pairings), naiveother (naive, using other device data), observing (terminal-known, uses ATK tags), or observingother (may be a list)
# - sensors: a (all) or list of any of {Acc, Gyr, GRV, LAc}
# - gesture(s): a (all) or specific gesture (e.g. TAP1) (may be a list)
# - gestureset: payment or door
# - device(s): a (all) or list of any of {door, ring, watch} (the device whose data is being used)
# - featureset(s): integer (type of features to be used; may be a list)
# - poollimit: integer (number of parallel threads to use)
# - is_combinedmodel(s): true or false (optional; default false; to run the combined-features model; may be a list)
#
#opens all <userdir>/<featureset>-extracted-payment-<device>/<param>-features.csv or <userdir>/<featureset>-extracted-door-<device>/<param>-features.csv files, grabs all the feature data, classifies under attack conditions, and then tests the model
#results are output in the format: <userID> or 'average', precision, std dev. of precision, recall, std dev. of recall, F1, std dev. of F1
#outputs <datetime>-<userIDs>-<gestureset>-<device>-<featureset>-<mode>-<param>-<sensors>-<gesture>.csv

import datetime, csv, os, re, sys
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

basedir = sys.argv[1] + '/'
userIDs = re.split(',', (sys.argv[2]).lower())
params = re.split(',', (sys.argv[3]).lower())
modes = re.split(',', (sys.argv[4]).lower())
sensors = ['Acc', 'Gyr', 'GRV', 'LAc'] if 'a' == (sys.argv[5]).lower() else re.split(',', sys.argv[5])
gestures = re.split(',', (sys.argv[6]).lower())
for i in range(len(gestures)):
	if gestures[i] != 'a':
		gestures[i] = gestures[i].upper()
gestureset = (sys.argv[7]).lower()
devices = ['door', 'ring', 'watch'] if 'a' == (sys.argv[8]).lower() else re.split(',', sys.argv[8])
featuresets = re.split(',', sys.argv[9])
poollimit = int(sys.argv[10])
is_combinedmodels = [False]
if len(sys.argv) > 11:
	is_combinedmodels = [(True if 'TRUE' == i.upper() or '1' == i else False) for i in re.split(',', sys.argv[11])]

#configs
classifier = 'rfc'
fontsize_legends = 20
maxpoolsize = 36
maxprewindowsize = 4
repetitions = 10

def get_tidy_userIDs():
	global userIDs
	t_userIDs = []
	for userID in userIDs:	
		if 'user' in userID:
			t_userIDs.append('user' + f'{int(userID[4:]):03}')
		else:
			t_userIDs.append('user' + f'{int(userID):03}')
	t_userIDs.sort(reverse = False)
	return(t_userIDs)

def get_tidy_params():
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
	return(t_params)

def rewrite_param(param):
	t_param = param
	if 'o-' in t_param:
		t_param = re.sub('o-', 'om', t_param)
	return t_param

def get_features(data, sensors):
	featurecolumns = []
	f_names = data[1:]
	f_column = 1
	for f_name in f_names:
		s = re.split('-', f_name)[0]
		for sensor in sensors:
			if s == sensor:
				featurecolumns.append(f_column)
				break;
		f_column = f_column + 1
	featurecolumns.sort(reverse = False)
	featurenames = [f_names[c - 1] for c in featurecolumns]
	return featurenames, featurecolumns

def get_features_combined(data1, data2, sensors):
	featurecolumns = [[], []]
	f_names = [data1[1:], data2[1:]]
	for index in [0,1]:
		f_column = 1
		for f_name in f_names[index]:
			s = re.split('-', f_name)[0]
			for sensor in sensors:
				if s == sensor:
					featurecolumns[index].append(f_column)
					break;
			f_column = f_column + 1
		featurecolumns[index].sort(reverse = False)
	featurenames = [f_names[0][c - 1] for c in featurecolumns[0]] + [f_names[1][c - 1] for c in featurecolumns[1]]
	return featurenames, featurecolumns[0], featurecolumns[1]

def get_features_combined_door(data1, data2, data3, sensors):
	featurecolumns = [[], [], []]
	f_names = [data1[1:], data2[1:], data3[1:]]
	for index in [0,1,2]:
		f_column = 1
		for f_name in f_names[index]:
			s = re.split('-', f_name)[0]
			for sensor in sensors:
				if s == sensor:
					featurecolumns[index].append(f_column)
					break;
			f_column = f_column + 1
		featurecolumns[index].sort(reverse = False)
	featurenames = [f_names[0][c - 1] for c in featurecolumns[0]] + [f_names[1][c - 1] for c in featurecolumns[1]] + [f_names[2][c - 1] for c in featurecolumns[2]]
	return featurenames, featurecolumns[0], featurecolumns[1], featurecolumns[2]

def get_average(l):
	return 0 if 0 == len(l) else sum(l) / len(l)

def get_eer(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point where FRR crosses FAR
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr >= far:
			return threshold, far
	return 1, 1

def get_ascending_userID_list_string(userIDs):
	for u in userIDs:
		if not 'user' in u and len(u) != 7:
			sys.exit('ERROR: userID not valid: ' + str(u))
	IDs = [int(u[4:]) for u in userIDs]
	IDs.sort(reverse = False)
	return ','.join([f'{i:03}' for i in IDs])

def get_descending_feature_list_string(weights, labels, truncate = 0):
	indicies = [i for i in range(len(weights))]
	for i in range(len(indicies)):
		for j in range(len(indicies)):
			if i != j and weights[indicies[i]] > weights[indicies[j]]:
				temp = indicies[i]
				indicies[i] = indicies[j]
				indicies[j] = temp
	if truncate != 0:
		del indicies[truncate:]
	return ','.join([str('%.6f' % weights[i]) + ' (' + labels[i] + ')' for i in indicies])

def get_score_list_string(scores):
	return ','.join([str(s) for s in scores])

def write_file(filename, s):
	outfile = open(filename, 'a')
	outfile.write(s + '\n')
	outfile.close()

def attack(args):
	basedir, userIDs, gestureset, classifier, sensors, device, fs, gesture, mode, is_combinedmodel, param = args
	
	mode_string = mode
	if is_combinedmodel:
		mode_string = mode_string + '(combined)'
	param_string = ''
	param2_string = ''
	if 'payment' == gestureset:
		param_string = '-' + rewrite_param(param)
		param2_string = 'param: ' + param + ', '
	f_outfilename = datetime.datetime.now().strftime('%Y%m%d') + '-' + get_ascending_userID_list_string(userIDs) + '-' + gestureset + '-' + device + '-' + fs + '-' + mode_string + '-' + classifier + param_string + '-' + ','.join(sensors) + '-' + gesture + '.csv'
	
	if os.path.exists(f_outfilename):
		os.remove(f_outfilename)
	
	print('ATTACK: ' + gestureset + '-' + device + '-' + fs + ', ' + param2_string + 'classifier: ' + classifier + ', mode: ' + mode_string + ', sensor(s): ' + ','.join(sensors) + ', gesture: ' + gesture)
	
	a_data = [] #container to hold the feature data for all users
	a_labels = [] #container to hold the corresponding labels
	a_results = [] #container to hold the results
	featurenames = [] #container to hold the names of the features
	featurecolumns = [] #container to hold the column indices of the features to be used (determined by sensors)
	featurecolumns1 = [] #container to hold the column indicies of features associated with the first device if using a combined model
	featurecolumns2 = [] #container to hold the column indicies of features associated with the second device if using a combined model
	is_firstparse = True
	
	usergroup1 = ['user020', 'user021', 'user022']
	usergroup2 = ['user023', 'user024', 'user025', 'user026', 'user027', 'user028', 'user029', 'user030', 'user031', 'user032', 'user033', 'user034', 'user035', 'user036', 'user037', 'user038', 'user039', 'user040']
	
################################################################################
	if 'payment' == gestureset and ('naive' == mode or 'naiveother' == mode) and 'a' == gesture:
		write_file(f_outfilename, 'victimID|attackerID|gesture|repetition|pred_scores|labels_test|prec|rec|fm|eer|eer_theta|legit_scores|adv_scores|features')
		
		#get feature data and labels for all users
		for userID in userIDs:
			hand = ''
			if os.path.exists(basedir + userID + '-left/1-cleaned/'):
				hand = 'left'
			elif os.path.exists(basedir + userID + '-right/1-cleaned/'):
				hand = 'right'
			else:
				sys.exit('ERROR: no such sourcedir for user: ' + userID)
			
			userdir = basedir + userID + '-' + hand
			if not os.path.exists(userdir):
				sys.exit('ERROR: no such userdir for user: ' + userdir)
			
			if is_combinedmodel:
				data1 = []
				data2 = []
				other_device = 'ring' if 'watch' == device else 'watch'
				
				f_infile1 = userdir + '/' + fs + '-extracted-payment-' + device + '/' + rewrite_param(param) + '-features.csv'
				if os.path.exists(f_infile1):
					with open(f_infile1, 'r') as f:
						data1 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				f_infile2 = userdir + '/' + fs + '-extracted-payment-' + other_device + '/' + rewrite_param(param) + '-features.csv'
				if os.path.exists(f_infile2):
					with open(f_infile2, 'r') as f:
						data2 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				if is_firstparse:
					featurenames, featurecolumns1, featurecolumns2 = get_features_combined([h + '-' + device for h in data1[0]], [h + '-' + other_device for h in data2[0]], sensors)
					featurecolumns = featurecolumns1 + featurecolumns2
					is_firstparse = False
				data1.pop(0) #removes the column headers
				data2.pop(0) #removes the column headers
				
				for datum1 in data1:
					g = datum1[0]
					for datum2 in data2:
						if datum2[0] == g:
							d = [g]
							d.extend([float(datum1[n]) for n in featurecolumns1])
							d.extend([float(datum2[n]) for n in featurecolumns2])
							a_data.append(d)
							a_labels.append(userID)
							break
			else:
				f_infile = userdir + '/' + fs + '-extracted-payment-' + device + '/' + rewrite_param(param) + '-features.csv'
				if os.path.exists(f_infile):
					with open(f_infile, 'r') as f:
						data = list(csv.reader(f)) #returns a list of lists (each line is a list)
						if is_firstparse:
							featurenames, featurecolumns = get_features(data[0], sensors)
							is_firstparse = False
						data.pop(0) #removes the column headers
						
						for datum in data:
							d = [datum[0]]
							d.extend([float(datum[n]) for n in featurecolumns])
							a_data.append(d)
							a_labels.append(userID)
		
		#run tests
		device_string = device
		if 'naiveother' == mode:
			device_string = 'ring' if 'watch' == device else 'watch'
		for test_terminal in ['TAP1-' + device_string, 'TAP2-' + device_string, 'TAP3-' + device_string, 'TAP4-' + device_string, 'TAP5-' + device_string, 'TAP6-' + device_string]:
			for victimID in userIDs:
				for attackerID in userIDs:
					if attackerID != victimID:
						data_train = []
						data_test = []
						labels_train = []
						labels_test = []
						for i in range(len(a_data)):
							g_parts = re.split('-', a_data[i][0])
							if device_string == g_parts[1]:
								if victimID == a_labels[i]:
									if g_parts[0] in test_terminal:
										data_test.append(a_data[i][1:])
										labels_test.append(1)
									else:
										data_train.append(a_data[i][1:])
										labels_train.append(1)
								elif attackerID == a_labels[i]:
									if g_parts[0] in test_terminal:
										data_test.append(a_data[i][1:])
										labels_test.append(0)
								else:
									if g_parts[0] not in test_terminal:
										data_train.append(a_data[i][1:])
										labels_train.append(0)
						
						#use the victim's first and second sessions' tap gestures for training, rejecting the third session's
						removetrainindicies = []
						victimtrainsplit = int(labels_train.count(1) * 2 / 3) + 1
						victimtraincounter = 0
						for i in range(len(labels_train)):
							if 1 == labels_train[i]:
								if victimtraincounter < victimtrainsplit:
									victimtraincounter = victimtraincounter + 1
								else:
									removetrainindicies.append(i)
						removetrainindicies.sort(reverse = True)
						for r in removetrainindicies:
							del data_train[r]
							del labels_train[r]
						
						#use the victim's third session's tap gestures for testing, rejecting the first and second sessions'
						removetestindicies = []
						victimtestsplit = int(labels_test.count(1) * 2 / 3) + 1
						victimtestcounter = 0
						for i in range(len(labels_test)):
							if 1 == labels_test[i]:
								if victimtestcounter < victimtestsplit:
									removetestindicies.append(i)
									victimtestcounter = victimtestcounter + 1
								else:
									break
						removetestindicies.sort(reverse = True)
						for r in removetestindicies:
							del data_test[r]
							del labels_test[r]
						
						for repetition in range(repetitions):
							model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
							labels_pred = model.predict(data_test)
							
							#get precision, recall, and F-measure scores
							precision = str('%.6f' % precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred)))
							recall = str('%.6f' % recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred)))
							fmeasure = str('%.6f' % f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred)))
							
							#get EER
							labels_scores = model.predict_proba(data_test)[:, 1]
							scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
							scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
							eer_theta, eer = get_eer(scores_legit, scores_adv)
							eer = str('%.6f' % eer)
							eer_theta = str('%.6f' % eer_theta)
							
							write_file(f_outfilename, victimID + '|' + attackerID + '|' + test_terminal + '|' + str(repetition) + '|' +
							 get_score_list_string(labels_pred) + '|' + get_score_list_string(labels_test) + '|' +
							 precision + '|' + recall + '|' + fmeasure + '|' + eer + '|' + eer_theta + '|' +
							 get_score_list_string(scores_legit) + '|' + get_score_list_string(scores_adv) + '|' +
							 get_descending_feature_list_string(model.feature_importances_, featurenames, 20))
							
							print(' v=' + victimID + ', a=' + attackerID + ', ' + test_terminal + '(' + str(repetition) + '), prec=' +
							 precision + ', rec=' + recall + ', fm=' + fmeasure + ', eer=' + eer)
	
################################################################################
	elif 'observing' == mode or 'observingother' == mode:
		write_file(f_outfilename, 'victimID|repetition|attackerID|pred_scores|labels_test|fm|eer|eer_theta|legit_scores|adv_scores|features')
		
		#get feature data and labels for all users
		for userID in userIDs:
			hand = ''
			if os.path.exists(basedir + userID + '-left/1-cleaned/'):
				hand = 'left'
			elif os.path.exists(basedir + userID + '-right/1-cleaned/'):
				hand = 'right'
			else:
				sys.exit('ERROR: no such sourcedir for user: ' + userID)
			
			userdir = basedir + userID + '-' + hand
			if not os.path.exists(userdir):
				sys.exit('ERROR: no such userdir for user: ' + userdir)
			
			if 'payment' == gestureset and is_combinedmodel:
				data1 = []
				data2 = []
				other_device = 'ring' if 'watch' == device else 'watch'
				
				f_infile1 = userdir + '/' + fs + '-extracted-payment-' + device + '/' + rewrite_param(param) + '-features.csv'
				if os.path.exists(f_infile1):
					with open(f_infile1, 'r') as f:
						data1 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				f_infile2 = userdir + '/' + fs + '-extracted-payment-' + other_device + '/' + rewrite_param(param) + '-features.csv'
				if os.path.exists(f_infile2):
					with open(f_infile2, 'r') as f:
						data2 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				if is_firstparse:
					featurenames, featurecolumns1, featurecolumns2 = get_features_combined([h + '-' + device for h in data1[0]], [h + '-' + other_device for h in data2[0]], sensors)
					featurecolumns = featurecolumns1 + featurecolumns2
					is_firstparse = False
				data1.pop(0) #removes the column headers
				data2.pop(0) #removes the column headers
				
				for datum1 in data1:
					g = datum1[0]
					for datum2 in data2:
						if datum2[0] == g:
							d = [g]
							d.extend([float(datum1[n]) for n in featurecolumns1])
							d.extend([float(datum2[n]) for n in featurecolumns2])
							a_data.append(d)
							a_labels.append(userID)
							break
			elif 'door' == gestureset and is_combinedmodel:
				data1 = []
				data2 = []
				data3 = []
				device1 = 'watch'
				device2 = 'ring'
				device3 = 'door'
				
				f_infile1 = userdir + '/' + fs + '-extracted-door-' + device1 + '/' + 'features.csv'
				if os.path.exists(f_infile1):
					with open(f_infile1, 'r') as f:
						data1 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				f_infile2 = userdir + '/' + fs + '-extracted-door-' + device2 + '/' + 'features.csv'
				if os.path.exists(f_infile2):
					with open(f_infile2, 'r') as f:
						data2 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				f_infile3 = userdir + '/' + fs + '-extracted-door-' + device3 + '/' + 'features.csv'
				if os.path.exists(f_infile3):
					with open(f_infile3, 'r') as f:
						data3 = list(csv.reader(f)) #returns a list of lists (each line is a list)
				if is_firstparse:
					featurenames, featurecolumns1, featurecolumns2, featurecolumns3 = get_features_combined_door([h + '-' + device1 for h in data1[0]], [h + '-' + device2 for h in data2[0]], [h + '-' + device3 for h in data3[0]], sensors)
					featurecolumns = featurecolumns1 + featurecolumns2 + featurecolumns3
					is_firstparse = False
				data1.pop(0) #removes the column headers
				data2.pop(0) #removes the column headers
				data3.pop(0) #removes the column headers
				
				for datum1 in data1:
					g = datum1[0]
					for datum2 in data2:
						if datum2[0] == g:
							for datum3 in data3:
								if datum3[0] == g:
									d = [g]
									d.extend([float(datum1[n]) for n in featurecolumns1])
									d.extend([float(datum2[n]) for n in featurecolumns2])
									d.extend([float(datum3[n]) for n in featurecolumns3])
									a_data.append(d)
									a_labels.append(userID)
									break
							break
			else:
				f_infile = userdir + '/' + fs + '-extracted-payment-' + device + '/' + rewrite_param(param) + '-features.csv' if 'payment' == gestureset else userdir + '/' + fs + '-extracted-door-' + device + '/' + 'features.csv'
				if os.path.exists(f_infile):
					with open(f_infile, 'r') as f:
						data = list(csv.reader(f)) #returns a list of lists (each line is a list)
						if is_firstparse:
							featurenames, featurecolumns = get_features(data[0], sensors)
							is_firstparse = False
						data.pop(0) #removes the column headers
						
						for datum in data:
							d = [datum[0]]
							d.extend([float(datum[n]) for n in featurecolumns])
							a_data.append(d)
							a_labels.append(userID)
		
		#run tests
		victim_index = 3 if 'payment' == gestureset else 2
		device_string = device
		if 'observingother' == mode:
			device_string = 'ring' if 'watch' == device else 'watch'
		for victimID in userIDs:
			data_train = []
			data_test = []
			data_othertests = []
			labels_train = []
			labels_test = []
			labels_othertests = []
			attackerIDs_othertests = []
			for i in range(len(a_data)):
				g_parts = re.split('-', a_data[i][0])
				if 'ATK' != g_parts[0] and (('payment' == gestureset and device_string == g_parts[1]) or ('door' == gestureset and gesture == g_parts[0])):
					if ('payment' == gestureset and 0 < int(g_parts[2]) % 4) or ('door' == gestureset and int(g_parts[1]) < 22):
						data_train.append(a_data[i][1:])
						labels_train.append(1 if victimID == a_labels[i] else 0)
					else:
						data_test.append(a_data[i][1:])
						labels_test.append(1 if victimID == a_labels[i] else 0)
			
			for attackerID in userIDs:
				if (victimID in usergroup1 and attackerID in usergroup2) or (victimID in usergroup2 and attackerID in usergroup1):
					d = []
					l = []
					terminal_counter = []
					for i in range(len(a_data)):
						if attackerID == a_labels[i]:
							g_parts = re.split('-', a_data[i][0])
							if 'ATK' == g_parts[0] and (('payment' == gestureset and device_string == g_parts[2]) or ('door' == gestureset and gesture == g_parts[1])) and victimID == g_parts[victim_index]:
								terminal = g_parts[1]
								if terminal_counter.count(terminal) < 3 and len(terminal_counter) < 6:
									d.append(a_data[i][1:])
									l.append(0)
									terminal_counter.append(terminal)
					if len(d) > 0:
						data_othertests.append(d)
						labels_othertests.append(l)
						attackerIDs_othertests.append(attackerID)
			
			for repetition in range(repetitions):
				model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
				labels_pred = model.predict(data_test)
				
				#get F-measure score and EER
				fmeasure = str('%.6f' % f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred)))
				labels_scores = model.predict_proba(data_test)[:, 1]
				scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
				scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
				eer_theta, eer = get_eer(scores_legit, scores_adv)
				eer = str('%.6f' % eer)
				eer_theta = str('%.6f' % eer_theta)
				
				output_string = (victimID + '|' + str(repetition) + '|-|' +
				 get_score_list_string(labels_pred) + '|' + get_score_list_string(labels_test) + '|' +
				 fmeasure + '|' + eer + '|' + eer_theta + '|' +
				 get_score_list_string(scores_legit) + '|' + get_score_list_string(scores_adv) + '|' +
				 get_descending_feature_list_string(model.feature_importances_, featurenames, 20))
				
				#perform attacks
				for a in range(len(attackerIDs_othertests)):
					labels_pred = model.predict(data_othertests[a])
					fmeasure = str('%.6f' % f1_score(labels_othertests[a], labels_pred, average = 'macro', labels = np.unique(labels_pred)))
					labels_scores = model.predict_proba(data_othertests[a])[:, 1]
					scores_legit = [labels_scores[i] for i in range(len(labels_othertests[a])) if 1 == labels_othertests[a][i]]
					scores_adv = [labels_scores[i] for i in range(len(labels_othertests[a])) if 0 == labels_othertests[a][i]]
					eer_theta, eer = get_eer(scores_legit, scores_adv)
					eer = str('%.6f' % eer)
					eer_theta = str('%.6f' % eer_theta)
					output_string = (output_string + '\n' + victimID + '|' + str(repetition) + '|' + attackerIDs_othertests[a] + '|' +
					 get_score_list_string(labels_pred) + '|' + get_score_list_string(labels_othertests[a]) + '|' +
					 fmeasure + '|' + eer + '|' + eer_theta + '|' +
					 get_score_list_string(scores_legit) + '|' + get_score_list_string(scores_adv) + '|' +
					 get_descending_feature_list_string(model.feature_importances_, featurenames, 20))
				
				write_file(f_outfilename, output_string)
	
################################################################################
	else:
		sys.exit('ERROR: mode not valid: ' + mode)
	
	print('\nOUTPUT: ' + f_outfilename)

if __name__ == '__main__':
	userIDs = get_tidy_userIDs()
	params = get_tidy_params()
	
	configs = []
	for device in devices:
		if 'payment' == gestureset and 'door' == device:
			continue
		
		for fs in featuresets:
			for mode in modes:
				if 'door' == gestureset and 'other' in mode:
					continue
				
				if 'door' == gestureset:
					if 'a' == gestures[0]:
						gestures = ['NOC5', 'NSEC'] if 'observing' in mode else ['NOC3', 'NOC5', 'NSEC']
				
				for gesture in gestures:
					for is_combinedmodel in is_combinedmodels:
						if is_combinedmodel and 'other' in mode:
							continue
						
						if is_combinedmodel and 'door' == gestureset and ('ring' == device or 'watch' == device):
							continue
						
						params_gs = list(params) if 'payment' == gestureset else [0]
						
						for param in params_gs:
							config = [basedir, userIDs, gestureset, classifier, sensors, device, fs, gesture, mode, is_combinedmodel, param]
							configs.append(config)
	
	poolsize = min(maxpoolsize, poollimit, len(configs))
	print('Poolsize: ' + str(poolsize) + '  (Configs to run: ' + str(len(configs)) + ')')
	with Pool(poolsize) as p:
		p.map(attack, configs)