#python classify.py [basedir] [userIDs] [param(s)] [mode(s)] [sensors] [gesture(s)] [gestureset] [device(s)] [featureset(s)] [poollimit] [is_combinedmodel(s)] [trainingsizemultiplier(s)]
# - basedir
# - userIDs (must be a list)
# - param(s): f<windowsize>o<offset> or u<windowsize>o<offset> (may be a list)
# - mode(s): authpayment (auth by terminal, tested on untrained terminals), authpaymentother (authpayment, gestures made with other device),
# authpayment6n (authpayment, trained on only 6 x trainingsizemultiplier user samples), authpaymentother6n (authpayment6n, gestures made with other device),
# authpaymentmerged (auth but without terminal agnosticity), authpaymentmergedother (authpaymentmerged, gestures made with other device),
# age, sex, height, heightfixed3, or 
# authdoor, authdoor6n (may be a list)
# - sensors: a (all) or list of any of {Acc, Gyr, GRV, LAc}
# - gesture(s): a (all) or specific gesture (e.g. TAP1) (may be a list)
# - gestureset: payment or door
# - device(s): a (all) or list of any of {door, ring, watch} (the device whose data is being used)
# - featureset(s): integer (type of features to be used; may be a list)
# - poollimit: integer (number of parallel threads to use)
# - is_combinedmodel(s): true or false (optional; default false; to run the combined-features model; may be a list)
# - trainingsizemultiplier(s): number of samples per user in the training set (optional; only for authblind6n or authdoor6n mode; may be a list)
#
#opens all <userdir>/<featureset>-extracted-payment-<device>/<param>-features.csv or <userdir>/<featureset>-extracted-door-<device>/<param>-features.csv files, grabs all the feature data, classifies, and then tests the model
#results are output in the format: <userID> or 'average', precision, std dev. of precision, recall, std dev. of recall, F1, std dev. of F1
#outputs <datetime>-<userIDs>-<gestureset>-<device>-<featureset>-<mode>-<classifier>-<param>-<sensors>-<gesture>.csv

import datetime, csv, os, re, sys
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import auc, f1_score, precision_score, precision_recall_curve, recall_score, roc_curve, accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import KFold

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
trainingsizemultipliers = [0]
if len(sys.argv) > 12:
	trainingsizemultipliers = [int(i) for i in re.split(',', sys.argv[12])]

#configs
classifier = 'rfc'
fontsize_legends = 20
maxpoolsize = 36
maxprewindowsize = 4
repetitions = 10
folds = 10

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

def get_far_when_zero_frr(scores_legit, scores_adv):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	#treat each legitimate sample distance as a possible threshold, determine the point with the lowest FAR that satisfies the condition that FRR = 0
	for c, threshold in enumerate(scores_legit):
		frr = c * 1.0 / len(scores_legit)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far = 1 - (adv_index * 1.0 / len(scores_adv))
		if frr > 0.001:
			return threshold, far

def plot_threshold_by_far_frr(scores_legit, scores_adv, far_theta):
	scores_legit = sorted(scores_legit)
	scores_adv = sorted(scores_adv)
	
	frr = []
	far = []
	thresholds = []
	for c, threshold in enumerate(scores_legit):
		frr.append((c * 1.0 / len(scores_legit)) * 100)
		adv_index = next((x[0] for x in enumerate(scores_adv) if x[1] > threshold), len(scores_adv))
		far.append((1 - (adv_index * 1.0 / len(scores_adv))) * 100)
		thresholds.append(threshold)
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.plot(thresholds, far, 'tab:blue', label = 'FAR')
	plt.plot(thresholds, frr, 'tab:orange', label = 'FRR')
	plt.ylabel('error rate (%)')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.axvline(x = far_theta, c = 'red')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_threshold_by_precision_recall(labels_test, labels_scores):
	p, r, thresholds = precision_recall_curve(labels_test, labels_scores)
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('Precision and Recall Scores as a Function of the Decision Threshold', fontsize = 12)
	plt.plot(thresholds, p[:-1], 'tab:blue', label = 'precision')
	plt.plot(thresholds, r[:-1], 'tab:orange', label = 'recall')
	plt.ylabel('score')
	plt.xlabel(r'decision threshold, $\theta$')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

def plot_roc_curve(labels_test, labels_scores):
	fpr, tpr, auc_thresholds = roc_curve(labels_test, labels_scores)
	print('AUC of ROC = ' + str(auc(fpr, tpr)))
	plt.figure(figsize = (6, 6))
	plt.rcParams.update({'font.size': fontsize_legends})
	plt.title('ROC Curve', fontsize = 12)
	plt.plot(fpr, tpr, 'tab:orange', label = 'recall optimised')
	plt.plot([0, 1], [0, 1], 'k--')
	plt.axis([-0.005, 1, 0, 1.005])
	plt.xticks(np.arange(0, 1, 0.05), rotation = 90)
	plt.xlabel('false positive rate')
	plt.ylabel('true positive rate (recall)')
	plt.legend(loc = 'best')
	plt.tight_layout(pad = 0.05)
	plt.show()

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
	return '\n'.join([str('%.6f' % weights[i]) + ' (' + labels[i] + ')' for i in indicies])

def write_verbose(f, s):
	f_outfilename = f + '-verbose.txt'
	outfile = open(f_outfilename, 'a')
	outfile.write(s + '\n')
	outfile.close()

def classify(args):
	basedir, userIDs, gestureset, classifier, sensors, device, fs, gesture, mode, is_combinedmodel, trainingsizemultiplier, param = args
	
	mode_string = mode
	if 'authpayment6n' == mode or 'authpaymentother6n' == mode or 'authdoor6n' == mode:
		mode_string = mode_string + '(n=' + str(trainingsizemultiplier) + ')'
	if is_combinedmodel:
		mode_string = mode_string + '(combined)'
	param_string = ''
	param2_string = ''
	if 'payment' == gestureset:
		param_string = '-' + rewrite_param(param)
		param2_string = 'param: ' + param + ', '
	filename_string = datetime.datetime.now().strftime('%Y%m%d') + '-' + get_ascending_userID_list_string(userIDs) + '-' + gestureset + '-' + device + '-' + fs + '-' + mode_string + '-' + classifier + param_string + '-' + ','.join(sensors) + '-' + gesture
	
	output = []
	print('CLASSIFY: ' + gestureset + '-' + device + '-' + fs + ', ' + param2_string + 'classifier: ' + classifier + ', mode: ' + mode_string + ', sensor(s): ' + ','.join(sensors) + ', gesture: ' + gesture)
	
	a_data = [] #container to hold the feature data for all users
	a_labels = [] #container to hold the corresponding labels
	a_precisions = []
	a_recalls = []
	a_fmeasures = []
	a_accuracies = []
	a_pr_stdev = []
	a_re_stdev = []
	a_fm_stdev = []
	a_eers = []
	a_eer_thetas = []
	a_fars = []
	a_far_thetas = []
	a_ee_stdev = []
	a_ee_th_stdev = []
	a_fa_stdev = []
	a_fa_th_stdev = []
	featurenames = [] #container to hold the names of the features
	featurecolumns = [] #container to hold the column indices of the features to be used (determined by sensors)
	featurecolumns1 = [] #container to hold the column indicies of features associated with the first device if using a combined model
	featurecolumns2 = [] #container to hold the column indicies of features associated with the second device if using a combined model
	is_firstparse = True
	
	ages = {'user020': 2, 'user021': 1, 'user022': 1, 'user023': 1, 'user024': 1, 'user025': 1, 'user026': 2, 'user027': 1, 'user028': 0, 'user029': 1, 'user030': 1, 'user031': 2, 'user032': 1, 'user033': 0, 'user034': 0, 'user035': 1, 'user036': 1, 'user037': 1, 'user038': 0, 'user039': 1, 'user040': 1}	
	sexes = {'user020': 0, 'user021': 0, 'user022': 0, 'user023': 1, 'user024': 0, 'user025': 1, 'user026': 0, 'user027': 0, 'user028': 1, 'user029': 0, 'user030': 0, 'user031': 1, 'user032': 0, 'user033': 0, 'user034': 0, 'user035': 1, 'user036': 0, 'user037': 1, 'user038': 0, 'user039': 0, 'user040': 0}
	heights = {'user020': 175, 'user021': 192, 'user022': 192, 'user023': 163, 'user024': 185, 'user025': 174, 'user026': 181, 'user027': 179, 'user028': 170, 'user029': 192, 'user030': 162, 'user031': 163, 'user032': 185, 'user033': 183, 'user034': 178.5, 'user035': 173, 'user036': 174, 'user037': 158.5, 'user038': 181, 'user039': 175, 'user040': 176}
	
################################################################################
	if 'payment' == gestureset and ('authpayment' == mode or 'authpaymentother' == mode or ('authpayment6n' == mode and trainingsizemultiplier > 0) or ('authpaymentother6n' == mode and trainingsizemultiplier > 0)) and 'a' == gesture:
		output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev')
		
		other_device = 'ring' if 'watch' == device else 'watch'
		
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
					if not 'ATK' in g:
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
							g = datum[0]
							if not 'ATK' in g:
								d = [g]
								d.extend([float(datum[n]) for n in featurecolumns])
								a_data.append(d)
								a_labels.append(userID)
		
		#run tests
		device_string = other_device if 'other' in mode else device
		for test_terminal in ['TAP1-' + device_string, 'TAP2-' + device_string, 'TAP3-' + device_string, 'TAP4-' + device_string, 'TAP5-' + device_string, 'TAP6-' + device_string]:
			t_precisions = []
			t_recalls = []
			t_fmeasures = []
			t_eers = []
			t_eer_thetas = []
			t_fars = []
			t_far_thetas = []
			
			for userID in userIDs:
				data_train = []
				data_test = []
				labels_train = []
				labels_test = []
				counter = [0, 0, 0, 0, 0, 0, 0]
				for i in range(len(a_data)):
					g_parts = re.split('-', a_data[i][0])
					if device_string == g_parts[1]:
						if g_parts[0] in test_terminal:
							data_test.append(a_data[i][1:])
							labels_test.append(1 if userID == a_labels[i] else 0)
						else:
							if 'authpayment6n' == mode or 'authpaymentother6n' == mode:
								if userID == a_labels[i]:
									n = re.sub('TAP', '', g_parts[0])
									n = 6 if 'F' == n else int(n) - 1
									if counter[n] < trainingsizemultiplier:
										data_train.append(a_data[i][1:])
										labels_train.append(1)
										counter[n] = counter[n] + 1
								else:
									data_train.append(a_data[i][1:])
									labels_train.append(0)
							else:
								data_train.append(a_data[i][1:])
								labels_train.append(1 if userID == a_labels[i] else 0)
				
				#use the user's first and second sessions' tap gestures for training, rejecting the third session's
				if 'authpayment' == mode or 'authpaymentother' == mode or ('authpayment6n' == mode and trainingsizemultiplier > 20) or ('authpaymentother6n' == mode and trainingsizemultiplier > 20):
					removetrainindicies = []
					usertrainsplit = int(labels_train.count(1) * 2 / 3) + 1
					usertraincounter = 0
					for i in range(len(labels_train)):
						if 1 == labels_train[i]:
							if usertraincounter < usertrainsplit:
								usertraincounter = usertraincounter + 1
							else:
								removetrainindicies.append(i)
					removetrainindicies.sort(reverse = True)
					for r in removetrainindicies:
						del data_train[r]
						del labels_train[r]
				
				#use the user's third session's tap gestures for testing, rejecting the first and second sessions'
				removetestindicies = []
				usertestsplit = int(labels_test.count(1) * 2 / 3) + 1
				usertestcounter = 0
				for i in range(len(labels_test)):
					if 1 == labels_test[i]:
						if usertestcounter < usertestsplit:
							removetestindicies.append(i)
							usertestcounter = usertestcounter + 1
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
					precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
					recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
					fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
					t_precisions.append(precision)
					t_recalls.append(recall)
					t_fmeasures.append(fmeasure)
					
					#get EER and find the decision threshold and FAR when optimised for FRR
					labels_scores = model.predict_proba(data_test)[:, 1]
					scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
					scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
					eer_theta, eer = get_eer(scores_legit, scores_adv)
					t_eers.append(eer)
					t_eer_thetas.append(eer_theta)
					far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
					t_fars.append(far)
					t_far_thetas.append(far_theta)
					
					write_verbose(filename_string, '----\n----EXCLUDED TERMINAL ' + test_terminal + ', USERID ' + userID + ', REPETITION ' + str(repetition) +
					 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
					 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
					 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
					 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
			t_pr_stdev = np.std(t_precisions, ddof = 1)
			t_re_stdev = np.std(t_recalls, ddof = 1)
			t_fm_stdev = np.std(t_fmeasures, ddof = 1)
			t_ee_stdev = np.std(t_eers, ddof = 1)
			t_ee_th_stdev = np.std(t_eer_thetas, ddof = 1)
			t_fa_stdev = np.std(t_fars, ddof = 1)
			t_fa_th_stdev = np.std(t_far_thetas, ddof = 1)
			
			result_string = (test_terminal + ',' + str('%.6f' % get_average(t_precisions)) + ',' + str('%.6f' % t_pr_stdev) + ','
			 + str('%.6f' % get_average(t_recalls)) + ',' + str('%.6f' % t_re_stdev) + ','
			 + str('%.6f' % get_average(t_fmeasures)) + ',' + str('%.6f' % t_fm_stdev) + ','
			 + str('%.6f' % get_average(t_eers)) + ',' + str('%.6f' % t_ee_stdev) + ','
			 + str('%.6f' % get_average(t_eer_thetas)) + ',' + str('%.6f' % t_ee_th_stdev) + ','
			 + str('%.6f' % get_average(t_fars)) + ',' + str('%.6f' % t_fa_stdev) + ','
			 + str('%.6f' % get_average(t_far_thetas)) + ',' + str('%.6f' % t_fa_th_stdev)
			 )
			output.append(result_string)
			#print(result_string)
			
			a_precisions.extend(t_precisions)
			a_recalls.extend(t_recalls)
			a_fmeasures.extend(t_fmeasures)
			a_pr_stdev.append(t_pr_stdev)
			a_re_stdev.append(t_re_stdev)
			a_fm_stdev.append(t_fm_stdev)
			a_eers.extend(t_eers)
			a_eer_thetas.extend(t_eer_thetas)
			a_fars.extend(t_fars)
			a_far_thetas.extend(t_far_thetas)
			a_ee_stdev.append(t_ee_stdev)
			a_ee_th_stdev.append(t_ee_th_stdev)
			a_fa_stdev.append(t_fa_stdev)
			a_fa_th_stdev.append(t_fa_th_stdev)
		result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
		 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
		 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
		 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
		 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
		 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
		 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev))
		 )
		output.append(result_string)
		#print(result_string + '\n----')
	
################################################################################
	elif 'payment' == gestureset and ('authpaymentmerged' == mode or 'authpaymentmergedother' == mode):
		output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev')
		
		other_device = 'ring' if 'watch' == device else 'watch'
		
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
					if not 'ATK' in g:
						if 'a' == gesture or re.split('-', g)[0] == gesture:
							d = [g]
							d.extend([float(datum1[n]) for n in featurecolumns1])
							for datum2 in data2:
								if datum2[0] == g:
									d.extend([float(datum2[n]) for n in featurecolumns2])
									break
							a_data.append(d)
							a_labels.append(userID)
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
							g = datum[0]
							if not 'ATK' in g:
								if 'a' == gesture or re.split('-', g)[0] == gesture:
									d = [g]
									d.extend([float(datum[n]) for n in featurecolumns])
									a_data.append(d)
									a_labels.append(userID)
		
		#run tests
		device_string = other_device if 'other' in mode else device
		for userID in userIDs:
			u_precisions = []
			u_recalls = []
			u_fmeasures = []
			u_eers = []
			u_eer_thetas = []
			u_fars = []
			u_far_thetas = []
			
			data_train = []
			data_test = []
			labels_train = []
			labels_test = []
			for i in range(len(a_data)):
				g_parts = re.split('-', a_data[i][0])
				if g_parts[1] == device_string:
					if int(g_parts[2]) > 3 and int(g_parts[2]) < 28:
						data_train.append(a_data[i][1:])
						labels_train.append(1 if userID == a_labels[i] else 0)
					else:
						data_test.append(a_data[i][1:])
						labels_test.append(1 if userID == a_labels[i] else 0)
			
			for repetition in range(repetitions):
				model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
				labels_pred = model.predict(data_test)
				
				#get precision, recall, and F-measure scores
				precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				u_precisions.append(precision)
				u_recalls.append(recall)
				u_fmeasures.append(fmeasure)
				
				#get EER and find the decision threshold and FAR when optimised for FRR
				labels_scores = model.predict_proba(data_test)[:, 1]
				scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
				scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
				eer_theta, eer = get_eer(scores_legit, scores_adv)
				u_eers.append(eer)
				u_eer_thetas.append(eer_theta)
				far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
				u_fars.append(far)
				u_far_thetas.append(far_theta)
				
				write_verbose(filename_string, '----\n----USERID ' + userID + ', REPETITION ' + str(repetition) +
				 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
				 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
				 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
				 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
			u_pr_stdev = np.std(u_precisions, ddof = 1)
			u_re_stdev = np.std(u_recalls, ddof = 1)
			u_fm_stdev = np.std(u_fmeasures, ddof = 1)
			u_ee_stdev = np.std(u_eers, ddof = 1)
			u_ee_th_stdev = np.std(u_eer_thetas, ddof = 1)
			u_fa_stdev = np.std(u_fars, ddof = 1)
			u_fa_th_stdev = np.std(u_far_thetas, ddof = 1)
			
			result_string = (str('%.6f' % get_average(u_precisions)) + ',' + str('%.6f' % u_pr_stdev) + ','
			 + str('%.6f' % get_average(u_recalls)) + ',' + str('%.6f' % u_re_stdev) + ','
			 + str('%.6f' % get_average(u_fmeasures)) + ',' + str('%.6f' % u_fm_stdev) + ','
			 + str('%.6f' % get_average(u_eers)) + ',' + str('%.6f' % u_ee_stdev) + ','
			 + str('%.6f' % get_average(u_eer_thetas)) + ',' + str('%.6f' % u_ee_th_stdev) + ','
			 + str('%.6f' % get_average(u_fars)) + ',' + str('%.6f' % u_fa_stdev) + ','
			 + str('%.6f' % get_average(u_far_thetas)) + ',' + str('%.6f' % u_fa_th_stdev)
			 )
			output.append(result_string)
			#print(result_string)
			
			a_precisions.extend(u_precisions)
			a_recalls.extend(u_recalls)
			a_fmeasures.extend(u_fmeasures)
			a_pr_stdev.append(u_pr_stdev)
			a_re_stdev.append(u_re_stdev)
			a_fm_stdev.append(u_fm_stdev)
			a_eers.extend(u_eers)
			a_eer_thetas.extend(u_eer_thetas)
			a_fars.extend(u_fars)
			a_far_thetas.extend(u_far_thetas)
			a_ee_stdev.append(u_ee_stdev)
			a_ee_th_stdev.append(u_ee_th_stdev)
			a_fa_stdev.append(u_fa_stdev)
			a_fa_th_stdev.append(u_fa_th_stdev)
		result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
		 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
		 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
		 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
		 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
		 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
		 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev))
		 )
		output.append(result_string)
		#print(result_string + '\n----')
	
################################################################################
	elif 'age' == mode or 'sex' == mode or 'height' == mode or 'heightfixed3' == mode:
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
				
				if 'heightfixed3' == mode:
					for datum1 in data1:
						g = datum1[0]
						if re.split('-', g)[0] == 'TAP3':
							for datum2 in data2:
								if datum2[0] == g:
									d = []
									d.extend([float(datum1[n]) for n in featurecolumns1])
									d.extend([float(datum2[n]) for n in featurecolumns2])
									a_data.append(d)
									a_labels.append(heights[userID])
									break
				elif 'height' == mode:
					for datum1 in data1:
						g = datum1[0]
						if not 'ATK' in g:
							for datum2 in data2:
								if datum2[0] == g:
									d = []
									d.extend([float(datum1[n]) for n in featurecolumns1])
									d.extend([float(datum2[n]) for n in featurecolumns2])
									a_data.append(d)
									a_labels.append(heights[userID])
									break
				else:
					for datum1 in data1:
						g = datum1[0]
						if not 'ATK' in g:
							for datum2 in data2:
								if datum2[0] == g:
									d = []
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
				
				if 'height' == mode:
					for datum1 in data1:
						g = datum1[0]
						if gesture == re.split('-', g)[0]:
							for datum2 in data2:
								if datum2[0] == g:
									for datum3 in data3:
										if datum3[0] == g:
											d = []
											d.extend([float(datum1[n]) for n in featurecolumns1])
											d.extend([float(datum2[n]) for n in featurecolumns2])
											d.extend([float(datum3[n]) for n in featurecolumns3])
											a_data.append(d)
											a_labels.append(heights[userID])
											break
									break
				else:
					for datum1 in data1:
						g = datum1[0]
						if gesture == re.split('-', g)[0]:
							for datum2 in data2:
								if datum2[0] == g:
									for datum3 in data3:
										if datum3[0] == g:
											d = []
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
						
						if 'heightfixed3' == mode:
							for datum in data:
								if re.split('-', datum[0])[0] == 'TAP3':
									a_data.append([float(datum[n]) for n in featurecolumns])
									a_labels.append(heights[userID])
						elif 'height' == mode:
							for datum in data:
								if not 'ATK' in datum[0]:
									a_data.append([float(datum[n]) for n in featurecolumns])
									a_labels.append(heights[userID])
						else:
							for datum in data:
								if not 'ATK' in datum[0]:
									a_data.append([float(datum[n]) for n in featurecolumns])
									a_labels.append(userID)
		
		#run tests
		if 'age' == mode or 'sex' == mode:
			output.append('-,prec_avg,fm_avg,acc_avg')
			
			for userID in userIDs:
				data_train = []
				data_test = []
				labels_train = []
				labels_test = []
				for i in range(len(a_data)):
					if userID == a_labels[i]:
						data_test.append(a_data[i][1:])
						labels_test.append(ages[userID] if 'age' == mode else sexes[userID])
					else:
						data_train.append(a_data[i][1:])
						labels_train.append(ages[a_labels[i]] if 'age' == mode else sexes[a_labels[i]])
				
				for repetition in range(repetitions):
					model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
					labels_pred = model.predict(data_test)
					
					#get precision, F-measure, and accuracy scores
					precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
					fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
					accuracy = accuracy_score(labels_test, labels_pred)
					a_precisions.append(precision)
					a_fmeasures.append(fmeasure)
					a_accuracies.append(accuracy)
					
					write_verbose(filename_string, '----\n----USER LEFT OUT ' + str(userID) + ', REPETITION ' + str(repetition) +
					 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', fmeasure=' + str('%.6f' % fmeasure) + ', accuracy=' + str('%.6f' % accuracy) +
					 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
					 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
		else:
			output.append('-,mse_avg,rmse_avg,r2_avg')
			for repetition in range(repetitions):
				clf = RandomForestRegressor(n_estimators = 100, random_state = repetition)
				
				#apply k-fold cross-validation and fit the model on each fold
				kf = KFold(n_splits = folds, shuffle = True, random_state = 0)
				fold = 0
				for train, test in kf.split(a_data):
					data_train = [a_data[i] for i in train]
					data_test = [a_data[i] for i in test]
					labels_train = [a_labels[i] for i in train]
					labels_test = [a_labels[i] for i in test]
					model = clf.fit(data_train, labels_train)
					labels_pred = model.predict(data_test)
					
					#get MSE, RMSE, and R2 scores
					mse = mean_squared_error(labels_test, labels_pred, squared = True)
					rmse = mean_squared_error(labels_test, labels_pred, squared = False)
					r2 = r2_score(labels_test, labels_pred)
					a_precisions.append(mse) #use this container for simplicity
					a_fmeasures.append(rmse) #use this container for simplicity
					a_accuracies.append(r2) #use this container for simplicity
					
					write_verbose(filename_string, '----\n----REPETITION ' + str(repetition) + ', FOLD ' + str(fold) +
					 '\n----\nVALUES: mse=' + str('%.6f' % mse) + ', rmse=' + str('%.6f' % rmse) + ', r2=' + str('%.6f' % r2) +
					 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
					 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
					
					fold = fold + 1
		result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ','
		 + str('%.6f' % get_average(a_fmeasures)) + ','
		 + str('%.6f' % get_average(a_accuracies))
		 )
		output.append(result_string)
		#print(result_string + '\n----')
	
################################################################################
	elif 'door' == gestureset and ('authdoor' == mode or ('authdoor6n' == mode and trainingsizemultiplier > 0)):
		output.append('userID,prec_avg,prec_stdev,rec_avg,rec_stdev,fm_avg,fm_stdev,eer_avg,eer_stdev,eer_theta_avg,eer_theta_stdev,far_avg,far_stdev,far_theta_avg,far_theta_stdev')
		
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
			
			device1 = 'watch'
			device2 = 'ring'
			device3 = 'door'
			if is_combinedmodel:
				data1 = []
				data2 = []
				data3 = []
				
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
					if gesture == re.split('-', g)[0]:
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
				f_infile = userdir + '/' + fs + '-extracted-door-' + device + '/' + 'features.csv'
				if os.path.exists(f_infile):
					with open(f_infile, 'r') as f:
						data = list(csv.reader(f)) #returns a list of lists (each line is a list)
						if is_firstparse:
							featurenames, featurecolumns = get_features(data[0], sensors)
							is_firstparse = False
						data.pop(0) #removes the column headers
						
						for datum in data:
							g = datum[0]
							if gesture == re.split('-', g)[0]:
								d = [g]
								d.extend([float(datum[n]) for n in featurecolumns])
								a_data.append(d)
								a_labels.append(userID)
		
		#run tests
		for userID in userIDs:
			u_precisions = []
			u_recalls = []
			u_fmeasures = []
			u_eers = []
			u_eer_thetas = []
			u_fars = []
			u_far_thetas = []
			
			data_train = []
			data_test = []
			labels_train = []
			labels_test = []
			counter = 0
			for i in range(len(a_data)):
				g_parts = re.split('-', a_data[i][0])
				if int(g_parts[1]) < 22: #gesture indicies from the first two sessions
					if 'authdoor6n' == mode:
						if userID == a_labels[i]:
							if counter < trainingsizemultiplier:
								data_train.append(a_data[i][1:])
								labels_train.append(1)
								counter = counter + 1
						else:
							data_train.append(a_data[i][1:])
							labels_train.append(0)
					else:
						data_train.append(a_data[i][1:])
						labels_train.append(1 if userID == a_labels[i] else 0)
				else:
					data_test.append(a_data[i][1:])
					labels_test.append(1 if userID == a_labels[i] else 0)
			
			for repetition in range(repetitions):
				model = RandomForestClassifier(n_estimators = 100, random_state = repetition).fit(data_train, labels_train)
				labels_pred = model.predict(data_test)
				
				#get precision, recall, and F-measure scores
				precision = precision_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				recall = recall_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				fmeasure = f1_score(labels_test, labels_pred, average = 'macro', labels = np.unique(labels_pred))
				u_precisions.append(precision)
				u_recalls.append(recall)
				u_fmeasures.append(fmeasure)
				
				#get EER and find the decision threshold and FAR when optimised for FRR
				labels_scores = model.predict_proba(data_test)[:, 1]
				scores_legit = [labels_scores[i] for i in range(len(labels_test)) if 1 == labels_test[i]]
				scores_adv = [labels_scores[i] for i in range(len(labels_test)) if 0 == labels_test[i]]
				eer_theta, eer = get_eer(scores_legit, scores_adv)
				u_eers.append(eer)
				u_eer_thetas.append(eer_theta)
				far_theta, far = get_far_when_zero_frr(scores_legit, scores_adv)
				u_fars.append(far)
				u_far_thetas.append(far_theta)
				
				write_verbose(filename_string, '----\n----USERID ' + userID + ', REPETITION ' + str(repetition) +
				 '\n----\nVALUES: precision=' + str('%.6f' % precision) + ', recall=' + str('%.6f' % recall) + ', fmeasure=' + str('%.6f' % fmeasure) +
				 ', eer=' + str('%.6f' % eer) + ', eer_theta=' + str('%.6f' % eer_theta) + ', far=' + str('%.6f' % far) + ', far_theta=' + str('%.6f' % far_theta) +
				 '\n----\nFEATURE COUNT: ' + str(len(featurenames)) +
				 '\n----\nORDERED FEATURE LIST:\n' + get_descending_feature_list_string(model.feature_importances_, featurenames))
			u_pr_stdev = np.std(u_precisions, ddof = 1)
			u_re_stdev = np.std(u_recalls, ddof = 1)
			u_fm_stdev = np.std(u_fmeasures, ddof = 1)
			u_ee_stdev = np.std(u_eers, ddof = 1)
			u_ee_th_stdev = np.std(u_eer_thetas, ddof = 1)
			u_fa_stdev = np.std(u_fars, ddof = 1)
			u_fa_th_stdev = np.std(u_far_thetas, ddof = 1)
			
			result_string = (str('%.6f' % get_average(u_precisions)) + ',' + str('%.6f' % u_pr_stdev) + ','
			 + str('%.6f' % get_average(u_recalls)) + ',' + str('%.6f' % u_re_stdev) + ','
			 + str('%.6f' % get_average(u_fmeasures)) + ',' + str('%.6f' % u_fm_stdev) + ','
			 + str('%.6f' % get_average(u_eers)) + ',' + str('%.6f' % u_ee_stdev) + ','
			 + str('%.6f' % get_average(u_eer_thetas)) + ',' + str('%.6f' % u_ee_th_stdev) + ','
			 + str('%.6f' % get_average(u_fars)) + ',' + str('%.6f' % u_fa_stdev) + ','
			 + str('%.6f' % get_average(u_far_thetas)) + ',' + str('%.6f' % u_fa_th_stdev)
			 )
			output.append(result_string)
			#print(result_string)
			
			a_precisions.extend(u_precisions)
			a_recalls.extend(u_recalls)
			a_fmeasures.extend(u_fmeasures)
			a_pr_stdev.append(u_pr_stdev)
			a_re_stdev.append(u_re_stdev)
			a_fm_stdev.append(u_fm_stdev)
			a_eers.extend(u_eers)
			a_eer_thetas.extend(u_eer_thetas)
			a_fars.extend(u_fars)
			a_far_thetas.extend(u_far_thetas)
			a_ee_stdev.append(u_ee_stdev)
			a_ee_th_stdev.append(u_ee_th_stdev)
			a_fa_stdev.append(u_fa_stdev)
			a_fa_th_stdev.append(u_fa_th_stdev)
		result_string = ('average,' + str('%.6f' % get_average(a_precisions)) + ',' + str('%.6f' % get_average(a_pr_stdev)) + ','
		 + str('%.6f' % get_average(a_recalls)) + ',' + str('%.6f' % get_average(a_re_stdev)) + ','
		 + str('%.6f' % get_average(a_fmeasures)) + ',' + str('%.6f' % get_average(a_fm_stdev)) + ','
		 + str('%.6f' % get_average(a_eers)) + ',' + str('%.6f' % get_average(a_ee_stdev)) + ','
		 + str('%.6f' % get_average(a_eer_thetas)) + ',' + str('%.6f' % get_average(a_ee_th_stdev)) + ','
		 + str('%.6f' % get_average(a_fars)) + ',' + str('%.6f' % get_average(a_fa_stdev)) + ','
		 + str('%.6f' % get_average(a_far_thetas)) + ',' + str('%.6f' % get_average(a_fa_th_stdev))
		 )
		output.append(result_string)
		#print(result_string + '\n----')
	
################################################################################
	else:
		sys.exit('ERROR: mode not valid: ' + mode)
	
	f_outfilename = filename_string + '.csv'
	outfile = open(f_outfilename, 'w')
	outfile.write('\n'.join(output))
	outfile.close()
	print('OUTPUT: ' + f_outfilename)

if __name__ == '__main__':
	userIDs = get_tidy_userIDs()
	params = get_tidy_params()
	
	if 'door' == gestureset:
		if 'a' == gestures[0]:
			gestures = ['NOC3', 'NOC5', 'NSEC']
	
	configs = []
	for device in devices:
		if 'payment' == gestureset and 'door' == device:
			continue
		
		for fs in featuresets:	
			for gesture in gestures:
				for mode in modes:
					if 'door' == gestureset and 'other' in mode:
						continue
					
					for is_combinedmodel in is_combinedmodels:
						if is_combinedmodel and 'other' in mode:
							continue
						
						if is_combinedmodel and 'door' == gestureset and ('ring' == device or 'watch' == device):
							continue
						
						for trainingsizemultiplier in trainingsizemultipliers:
							params_gs = list(params) if 'payment' == gestureset else [0]
							
							for param in params_gs:
								config = [basedir, userIDs, gestureset, classifier, sensors, device, fs, gesture, mode, is_combinedmodel, trainingsizemultiplier, param]
								configs.append(config)
	
	poolsize = min(maxpoolsize, poollimit, len(configs))
	print('Poolsize: ' + str(poolsize) + '  (Configs to run: ' + str(len(configs)) + ')')
	with Pool(poolsize) as p:
		p.map(classify, configs)