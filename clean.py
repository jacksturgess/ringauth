#python clean.py [userID(s)] [is_downsamplering] [is_summariseonly]
# - userID(s) (may be a list)
# - is_downsamplering: true or false (optional; default true; to downsample ring to 50Hz)
# - is_summariseonly: true of false (optional; default false; to run the summarise method only)
#
#opens the file(s) in <userID>-<hand>/0-raw/ and uses the NFC timestamps to extract the watch and ring data for each payment gesture in a time window of up to <prewindowsizems> before the timestamp and <postwindowsizems> after,
# and uses the NFC and button-press timestamps to extract each door gesture
#outputs gestures-<gestureset>-<device>.csv and timestamps-<gestureset>.csv files in <cleandir>/

import csv, math, os, re, shutil, sys
from _csv import Error as CSV_Error

userIDs = re.split(',', (sys.argv[1]).lower())
is_downsamplering = True
if len(sys.argv) > 2:
	is_downsamplering = True if 'TRUE' == (sys.argv[2]).upper() or '1' == sys.argv[2] else False
is_summariseonly = False
if len(sys.argv) > 3:
	is_summariseonly = True if 'TRUE' == (sys.argv[3]).upper() or '1' == sys.argv[3] else False

#configs
sensors = ['Acc', 'Gyr', 'GRV', 'LAc']
prewindowsizems = 4000
postwindowsizems = 1000
buttonsizems = 7000
minsamplespergesture_payment = [400, 400] if is_downsamplering else [400, 800]
minsamplespergesture_door = [200, 200, 80] if is_downsamplering else [200, 400, 80]

#consts
NFC_t_index = 1 #time value in NFC data
NFC_tag_index = 2 #action data, as dictated by NFC tags
bp_t_index = 1 #time value in button press data
 #watch data indicies
w_s_index = 0 #sensor name
w_t_index = 1 #UNIX timestamp
w_x_index = 2 #x-value
w_y_index = 3 #y-value
w_z_index = 4 #z-value
w_w_index = 5 #w-value (GRV only)
 #ring data indicies
r_t_index = 25 #UNIX timestamp
r_acc_x_index = 3
r_acc_y_index = 4
r_acc_z_index = 5
r_gyr_x_index = 0
r_gyr_y_index = 1
r_gyr_z_index = 2
r_lac_x_index = 20
r_lac_y_index = 21
r_lac_z_index = 22
r_grv_x_index = 14
r_grv_y_index = 15
r_grv_z_index = 16
r_grv_w_index = 13
 #door data indices
d_t_index = 0 #UNIX timestamp
d_acc_x_index = 1
d_acc_y_index = 2
d_acc_z_index = 3
d_gyr_x_index = 4
d_gyr_y_index = 5
d_gyr_z_index = 6

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

def remove_gestures(files, removegestures):
	for file in files:
		d_out = []
		with open(file, 'r') as f:
			d = list(csv.reader(f))
			for datum in d:
				if not datum[0] in removegestures:
					d_out.append(datum)
		
		outfile = open(file, 'w')
		outfile.write('\n'.join([','.join(o) for o in d_out]))
		outfile.close()

def summarise_gestures(files):
	s_outfilename = cleandir + 'summary.csv'
	if os.path.exists(s_outfilename):
		os.remove(s_outfilename)
	
	for file in files:
		uniquegestures = []
		with open(file, 'r') as f:
			d = list(csv.reader(f))
			d.pop(0)
			for datum in d:
				g_name = datum[0]
				if not g_name in uniquegestures:
					uniquegestures.append(g_name)
		
		summary = []
		unindexedgestures = []
		uniqueunindexedgestures = []
		for gesture in uniquegestures:
			g = re.split('-', gesture)
			u_name = g[0]
			if 3 == len(g):
				u_name = g[0] + '-' + g[1]
			if u_name not in uniqueunindexedgestures:
				uniqueunindexedgestures.append(u_name)
			unindexedgestures.append(u_name)
		for gesture in uniqueunindexedgestures:
			summary.append(gesture + '=' + str(unindexedgestures.count(gesture)))
		s_summary = 'total=' + str(len(uniquegestures)) + '; ' + ', '.join(summary)
		s_outfile = open(s_outfilename, 'a')
		s_outfile.write('\n' + file + ':\n ' + s_summary + '\n')
		s_outfile.close()
		print('GESTURE SUMMARY: ' + file + '; ' + s_summary)

if __name__ == '__main__':
	tidy_userIDs()
	
	for userID in userIDs:
		hand = ''
		if os.path.exists(userID + '-left/0-raw/'):
			hand = 'left'
		elif os.path.exists(userID + '-right/0-raw/'):
			hand = 'right'
		else:
			sys.exit('ERROR: no such sourcedir for user: ' + userID)
		sourcedir = userID + '-' + hand + '/0-raw/'
		
		basedir = 'C:/temp/processed_data/ringauth/'
		if not os.path.exists(basedir):
			os.mkdir(basedir)
		userdir = basedir + userID + '-' + hand
		if not os.path.exists(userdir):
			os.mkdir(userdir)
		cleandir = userdir + '/1-cleaned/'
		
		if is_summariseonly:
			print("SUMMARISING DATA: " + userID)
			summarise_gestures([cleandir + 'gestures-payment-watch.csv', cleandir + 'gestures-payment-ring.csv', cleandir + 'gestures-door-watch.csv', cleandir + 'gestures-door-ring.csv', cleandir + 'gestures-door-door.csv'])
			continue;
		else:
			if os.path.exists(cleandir):
				shutil.rmtree(cleandir)
			os.mkdir(cleandir)
		
		print("CLEANING DATA: " + userID)
		
		p_data = [] #container to hold all of the payment timestamps for this user
		d_data_temp = [] #container to temporarily hold the door timestamps for this user from the NFC timestamper
		d_data = [] #container to hold all of the door timestamps for this user
		
		#extract the NFC timestamps
		if True:
			with open(sourcedir + 'nfctimestamps.csv', 'r') as f:
				data = csv.reader((line.replace('\0','') for line in f)) #returns a list of lists (each line is a list)
				prev_timestamps = []
				prev_gesture = ''
				for datum in data:
					if len(datum) > 0: #disregards empty lines
						if int(datum[NFC_t_index]) not in prev_timestamps: #disregards repeated lines
							prev_timestamps.append(int(datum[NFC_t_index]))
							tag = datum[NFC_tag_index].strip()
							if 'ATK' in tag:
								prev_gesture = tag
								if 'NOC5' in tag or 'NSEC' in tag:
									d_data_temp.append([int(datum[NFC_t_index]), tag])
							elif 'TAP' in tag:
								prev_gesture = tag
							elif 'WTC' in tag:
								g = prev_gesture + '-' if 'ATK' in prev_gesture else prev_gesture + '-watch-'
								counter = 1
								for p in p_data:
									if g in p[1]:
										counter = counter + 1
								p_data.append([datum[NFC_t_index], g + f'{counter:03}', 0, 0])
							elif 'RNG' in tag:
								g = prev_gesture + '-' if 'ATK' in prev_gesture else prev_gesture + '-ring-'
								counter = 1
								for p in p_data:
									if g in p[1]:
										counter = counter + 1
								p_data.append([datum[NFC_t_index], g + f'{counter:03}', 0, 0])
							elif 'NOC3' in tag or 'NOC5' in tag or 'NSEC' in tag:
								d_data_temp.append([int(datum[NFC_t_index]), tag])
							elif 'ATTN' in tag:
								d_data_temp.append([int(datum[NFC_t_index]), tag])
		
		#extract and associate button-press timestamps
		if True:
			bp_data_inc = [] #container to temporarily hold the button-press data for this user from the *-ring-buttons.csv files
			bp_data_temp = [] #container to temporarily hold the button-press timestamps for this user after cleaning
			(_, _, files) = next(os.walk(sourcedir))
			for file in files:
				filename = re.split('-', os.path.splitext(file)[0])
				if len(filename) > 3 and 'ring' == filename[2] and 'buttons' == filename[3] and '.csv' == os.path.splitext(file)[1]:
					with open(sourcedir + file, 'r') as f:
						d = list(csv.reader(f))
						for datum in d:
							if len(datum) > 0:
								bp_data_inc.append(int(int(datum[bp_t_index]) / 1000)) #convert microseconds to milliseconds
			
			#do not include rapid-fire button-presses that occur less than a second after another one
			for datum in bp_data_inc:
				check = True
				for i in bp_data_temp:
					if datum >= i and datum <= i + 1000:
						check = False
						break
				if check:
					bp_data_temp.append(datum)
			
			#associate NFC tags with button-press timestamps
			prev_timestamp = ''
			prev_gesture = ''
			prev_count = ''
			suffix_alternator = 0
			for datum in d_data_temp:
				if not '' == prev_timestamp:
					for i in range(len(bp_data_temp)):
						if datum[0] > bp_data_temp[i] and prev_timestamp < bp_data_temp[i]:
							g = prev_gesture + '-'
							if 0 == suffix_alternator:
								counter = 1
								for d in d_data:
									if g in d[1]:
										counter = counter + 1
								counter = int((counter + 1) / 2)
								prev_count = f'{counter:03}'
								
								#unless it's an ATTN tag, only include a button-press as a start tag if there is a follow-up (end tag) button-press within <buttonsizems> milliseconds
								if 'ATTN' == prev_gesture:
									d_data.append([str(bp_data_temp[i]), 'ATTN', 0, 0, 0])
								else:
									if i < len(bp_data_temp) - 1 and bp_data_temp[i + 1] < bp_data_temp[i] + buttonsizems:
										d_data.append([str(bp_data_temp[i]), g + prev_count, 0, 0, 0])
										suffix_alternator = (suffix_alternator + 1) % 2
							else:
								d_data.append([str(bp_data_temp[i]), g + prev_count + '-end', 0, 0, 0])
								suffix_alternator = (suffix_alternator + 1) % 2
				prev_timestamp = datum[0]
				prev_gesture = datum[1]
				suffix_alternator = 0
			for i in range(len(bp_data_temp)):
				if prev_timestamp < bp_data_temp[i]:
					g = prev_gesture + '-'
					if 0 == suffix_alternator:
						counter = 1
						for d in d_data:
							if g in d[1]:
								counter = counter + 1
						counter = int((counter + 1) / 2)
						prev_count = f'{counter:03}'
						
						#unless it's an ATTN tag, only include a button-press as a start tag if there is a follow-up (end tag) button-press within <buttonsizems> milliseconds
						if 'ATTN' == prev_gesture:
							d_data.append([str(bp_data_temp[i]), 'ATTN', 0, 0, 0])
						else:
							if i < len(bp_data_temp) - 1 and bp_data_temp[i + 1] < bp_data_temp[i] + buttonsizems:
								d_data.append([str(bp_data_temp[i]), g + prev_count, 0, 0, 0])
								suffix_alternator = (suffix_alternator + 1) % 2
					else:
						d_data.append([str(bp_data_temp[i]), g + prev_count + '-end', 0, 0, 0])
						suffix_alternator = (suffix_alternator + 1) % 2
			
			#remove any ATTN tags and any (start and end) timestamps to which attention was drawn using an ATTN tag from the list, printing out the latter
			attn_indicies = []
			for i in range(len(d_data)):
				if 'ATTN' == d_data[i][1]:
					attn_indicies.append(i)
			for datum in d_data_temp:
				if 'ATTN' in datum[1]:
					attn_timestamp = int(datum[0])
					for i in range(len(d_data)):
						if i < len(d_data) - 1 and not '-end' in d_data[i][1] and int(d_data[i][0]) < attn_timestamp and int(d_data[i + 1][0]) > attn_timestamp:
							attn_indicies.append(i)
							print("REMOVAL: for ATTN " + str(attn_timestamp) + ", removed " + d_data[i][0] + " (" + d_data[i][1] + ")")
						if i > 0 and '-end' in d_data[i][1] and int(d_data[i - 1][0]) < attn_timestamp and int(d_data[i][0]) > attn_timestamp:
							attn_indicies.append(i)
							print("REMOVAL: for ATTN " + str(attn_timestamp) + ", removed " + d_data[i][0] + " (" + d_data[i][1] + ")")
			if len(attn_indicies) > 0:
				attn_indicies.sort(reverse = True)
				for i in attn_indicies:
					d_data.pop(i)
		
		#open all watch/ring data files and extract payment gesture data by timestamp
		if True:
			for device in ['watch', 'ring']:
				w_data = [] #container to hold all of the watch/ring sensor data for this user
				w_countindex = 2 if 'watch' == device else 3
				
				(_, _, files) = next(os.walk(sourcedir))
				for file in files:
					filename = re.split('-', os.path.splitext(file)[0])
					if len(filename) > 3 and device == filename[2] and 'sensors' == filename[3] and '.csv' == os.path.splitext(file)[1]:
						with open(sourcedir + file, 'r') as f:
							d = list(csv.reader(f))
							if 'ring' == device:
								d.pop(0)
								
								#calculate and apply the time delta to convert local timestamps in the current ring sensor data file into UNIX time
								r_delta = 0
								r_deltafilename = sourcedir + filename[0] + '-' + filename[1] + '-ring-timesync.csv'
								if not os.path.exists(r_deltafilename):
									sys.exit('ERROR: no such timesync file: ' + r_deltafilename)
								with open(r_deltafilename, 'r') as f2:
									d2 = list(csv.reader(f2))
									r_delta = int((int(d2[0][2]) - int(d2[0][1])) / 1000)
								for datum in d:
									datum[r_t_index] = str(int(int(datum[r_t_index]) / 1000) + r_delta)
								
								#downsample ring sensor data if applicable
								if is_downsamplering:
									d2 = []
									acceptor = True
									for datum in d:
										if acceptor:
											d2.append(datum)
										acceptor = not acceptor
									d = d2
							w_data.append(d)
				
				w_outfilename = cleandir + 'gestures-payment-' + device + '.csv'
				w_outfile = open(w_outfilename, 'w')
				w_outfile.write('GESTURE,SENSOR,ORIGINAL_TIMESTAMP,GESTURE_TIMESTAMP,X-VALUE,Y-VALUE,Z-VALUE,UNFILTERED_EUCLIDEAN_NORM')
				w_outfile.close()
				
				#for each NFC timestamp: project a time window (backwards and forwards), grab the watch/ring sensor data inside that window, clean it, and write it
				for i in range(len(p_data)):
					triggertime = int(p_data[i][0])
					starttime = triggertime - prewindowsizems
					if i > 0 and triggertime > int(p_data[i - 1][0]) and starttime < int(p_data[i - 1][0]):
						starttime = int(p_data[i - 1][0]) #avoids window overlaps
					endtime = triggertime + postwindowsizems
					
					#get gesture data
					g_data = [] #container to hold the gesture data
					g_is_found = False
					for data in w_data:
						if not g_is_found:
							index = r_t_index if 'ring' == device else w_t_index
							for datum in data:
								t = int(datum[index])
								if t >= starttime and t <= endtime:
									g_data.append(datum)
							if len(g_data) > 0:
								g_is_found = True
					
					if g_is_found:
						#clean gesture data
						g_output = []
						for datum in g_data:
							if 'watch' == device:
								s = datum[w_s_index]
								if s in sensors:
									t = datum[w_t_index]
									x = str('%.6f' % float(datum[w_x_index]))
									y = str('%.6f' % float(datum[w_y_index]))
									z = str('%.6f' % float(datum[w_z_index]))
									w = '0'
									norm = '0' #adds Euclidean norm of unfiltered values
									if 'GRV' == s:
										w = str('%.6f' % float(datum[w_w_index]))
									else:
										norm = str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z)))
									d = []
									d.append(s)
									d.append(t)
									d.append(str(float(int(t) - triggertime) / 1000)) #adds a normalised timestamp ending the gesture at 0 at the trigger point
									d.append(x)
									d.append(y)
									d.append(z)
									d.append(w)
									d.append(norm)
									g_output.append(d)
							elif 'ring' == device:
								t = datum[r_t_index]
								for s in sensors:
									if 'Acc' == s:
										x = str('%.6f' % float(datum[r_acc_x_index]))
										y = str('%.6f' % float(datum[r_acc_y_index]))
										z = str('%.6f' % float(datum[r_acc_z_index]))
										d = []
										d.append(s)
										d.append(t)
										d.append(str(float(int(t) - triggertime) / 1000))
										d.append(x)
										d.append(y)
										d.append(z)
										d.append('0')
										d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z)))) #adds Euclidean norm of unfiltered values
										g_output.append(d)
									elif 'Gyr' == s:
										x = str('%.6f' % float(datum[r_gyr_x_index]))
										y = str('%.6f' % float(datum[r_gyr_y_index]))
										z = str('%.6f' % float(datum[r_gyr_z_index]))
										d = []
										d.append(s)
										d.append(t)
										d.append(str(float(int(t) - triggertime) / 1000))
										d.append(x)
										d.append(y)
										d.append(z)
										d.append('0')
										d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
										g_output.append(d)
									elif 'GRV' == s:
										d = []
										d.append(s)
										d.append(t)
										d.append(str(float(int(t) - triggertime) / 1000))
										d.append(str('%.6f' % float(datum[r_grv_x_index])))
										d.append(str('%.6f' % float(datum[r_grv_y_index])))
										d.append(str('%.6f' % float(datum[r_grv_z_index])))
										d.append(str('%.6f' % float(datum[r_grv_w_index])))
										d.append('0')
										g_output.append(d)
									elif 'LAc' == s:
										x = str('%.6f' % float(datum[r_lac_x_index]))
										y = str('%.6f' % float(datum[r_lac_y_index]))
										z = str('%.6f' % float(datum[r_lac_z_index]))
										d = []
										d.append(s)
										d.append(t)
										d.append(str(float(int(t) - triggertime) / 1000))
										d.append(x)
										d.append(y)
										d.append(z)
										d.append('0')
										d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
										g_output.append(d)
						p_data[i][w_countindex] = len(g_output)
						
						#output the gesture
						w_output = []
						for datum in g_output:
							d = [p_data[i][1]]
							d.extend(datum)
							w_output.append(d)
						
						w_outfile = open(w_outfilename, 'a')
						w_outfile.write('\n' + '\n'.join([','.join(o) for o in w_output]))
						w_outfile.close()
				print('OUTPUT: ' + w_outfilename)
		
		#open all watch/ring/door data files and extract door gesture data by timestamp
		if True:
			for device in ['watch', 'ring', 'door']:
				w_data = [] #container to hold all of the watch/ring/door data for this user
				w_countindex = 2 if 'watch' == device else 3 if 'ring' == device else 4
				
				(_, _, files) = next(os.walk(sourcedir))
				for file in files:
					filename = re.split('-', os.path.splitext(file)[0])
					if len(filename) > 3 and device == filename[2] and 'sensors' == filename[3] and '.csv' == os.path.splitext(file)[1]:
						with open(sourcedir + file, 'r') as f:
							d = list(csv.reader(f))
							if 'ring' == device:
								d.pop(0)
								
								#calculate and apply the time delta to convert local timestamps in the current ring data file into UNIX time
								r_delta = 0
								r_deltafilename = sourcedir + filename[0] + '-' + filename[1] + '-ring-timesync.csv'
								if not os.path.exists(r_deltafilename):
									sys.exit('ERROR: no such timesync file: ' + r_deltafilename)
								with open(r_deltafilename, 'r') as f2:
									d2 = list(csv.reader(f2))
									r_delta = int((int(d2[0][2]) - int(d2[0][1])) / 1000)
								for datum in d:
									datum[r_t_index] = str(int(int(datum[r_t_index]) / 1000) + r_delta)
								
								#downsample ring sensor data if applicable
								if is_downsamplering:
									d2 = []
									acceptor = True
									for datum in d:
										if acceptor:
											d2.append(datum)
										acceptor = not acceptor
									d = d2
							elif 'door' == device:
								d.pop(0)
							w_data.append(d)
				
				w_outfilename = cleandir + 'gestures-door-' + device + '.csv'
				w_outfile = open(w_outfilename, 'w')
				w_outfile.write('GESTURE,SENSOR,ORIGINAL_TIMESTAMP,GESTURE_TIMESTAMP,X-VALUE,Y-VALUE,Z-VALUE,UNFILTERED_EUCLIDEAN_NORM')
				w_outfile.close()
				
				#for each door start tag, grab the watch/ring/door data between that timestamp and the following (end) tag timestamp, clean it, and write it
				for i in range(len(d_data)):
					if not '-end' in d_data[i][1]:
						starttime = int(d_data[i][0])
						endtime = int(d_data[i + 1][0])
						
						#get gesture data
						g_data = [] #container to hold the gesture data
						g_is_found = False
						for data in w_data:
							if not g_is_found:
								index = r_t_index if 'ring' == device else d_t_index if 'door' == device else w_t_index
								for datum in data:
									t = int(datum[index])
									if t >= starttime and t <= endtime:
										g_data.append(datum)
								if len(g_data) > 0:
									g_is_found = True
						
						if g_is_found:
							#clean gesture data
							g_output = []
							for datum in g_data:
								if 'watch' == device:
									s = datum[w_s_index]
									if s in sensors:
										t = datum[w_t_index]
										x = str('%.6f' % float(datum[w_x_index]))
										y = str('%.6f' % float(datum[w_y_index]))
										z = str('%.6f' % float(datum[w_z_index]))
										w = '0'
										norm = '0' #adds Euclidean norm of unfiltered values
										if 'GRV' == s:
											w = str('%.6f' % float(datum[w_w_index]))
										else:
											norm = str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z)))
										d = []
										d.append(s)
										d.append(t)
										d.append(str(float(int(t) - starttime) / 1000)) #adds a normalised timestamp ending the gesture at 0 at the starting button press
										d.append(x)
										d.append(y)
										d.append(z)
										d.append(w)
										d.append(norm)
										g_output.append(d)
								elif 'ring' == device:
									t = datum[r_t_index]
									for s in sensors:
										if 'Acc' == s:
											x = str('%.6f' % float(datum[r_acc_x_index]))
											y = str('%.6f' % float(datum[r_acc_y_index]))
											z = str('%.6f' % float(datum[r_acc_z_index]))
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000)) #adds a normalised timestamp ending the gesture at 0 at the starting button press
											d.append(x)
											d.append(y)
											d.append(z)
											d.append('0')
											d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
											g_output.append(d)
										elif 'Gyr' == s:
											x = str('%.6f' % float(datum[r_gyr_x_index]))
											y = str('%.6f' % float(datum[r_gyr_y_index]))
											z = str('%.6f' % float(datum[r_gyr_z_index]))
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000))
											d.append(x)
											d.append(y)
											d.append(z)
											d.append('0')
											d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
											g_output.append(d)
										elif 'GRV' == s:
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000))
											d.append(str('%.6f' % float(datum[r_grv_x_index])))
											d.append(str('%.6f' % float(datum[r_grv_y_index])))
											d.append(str('%.6f' % float(datum[r_grv_z_index])))
											d.append(str('%.6f' % float(datum[r_grv_w_index])))
											d.append('0')
											g_output.append(d)
										elif 'LAc' == s:
											x = str('%.6f' % float(datum[r_lac_x_index]))
											y = str('%.6f' % float(datum[r_lac_y_index]))
											z = str('%.6f' % float(datum[r_lac_z_index]))
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000))
											d.append(x)
											d.append(y)
											d.append(z)
											d.append('0')
											d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
											g_output.append(d)
								elif 'door' == device:
									t = datum[d_t_index]
									for s in ['Acc', 'Gyr']:
										if 'Acc' == s:
											x = str('%.6f' % float(datum[d_acc_x_index]))
											y = str('%.6f' % float(datum[d_acc_y_index]))
											z = str('%.6f' % float(datum[d_acc_z_index]))
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000)) #adds a normalised timestamp ending the gesture at 0 at the starting button press
											d.append(x)
											d.append(y)
											d.append(z)
											d.append('0')
											d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
											g_output.append(d)
										elif 'Gyr' == s:
											x = str('%.6f' % float(datum[d_gyr_x_index]))
											y = str('%.6f' % float(datum[d_gyr_y_index]))
											z = str('%.6f' % float(datum[d_gyr_z_index]))
											d = []
											d.append(s)
											d.append(t)
											d.append(str(float(int(t) - starttime) / 1000))
											d.append(x)
											d.append(y)
											d.append(z)
											d.append('0')
											d.append(str('%.6f' % math.sqrt(float(x) * float(x) + float(y) * float(y) + float(z) * float(z))))
											g_output.append(d)
							d_data[i][w_countindex] = len(g_output)
							
							#output the gesture
							w_output = []
							for datum in g_output:
								d = [d_data[i][1]]
								d.extend(datum)
								w_output.append(d)
							
							w_outfile = open(w_outfilename, 'a')
							w_outfile.write('\n' + '\n'.join([','.join(o) for o in w_output]))
							w_outfile.close()
				print('OUTPUT: ' + w_outfilename)
		
		#remove any gesture that does not have at least the minimum number of samples for every device
		if True:
			#payment gestures
			removegestures = []
			for d in p_data:
				if d[2] < minsamplespergesture_payment[0] or d[3] < minsamplespergesture_payment[1]:
					removegestures.append(d[1])
			if len(removegestures) > 0:			
				for r in removegestures:
					#remove gesture from list
					index = ''
					for i in range(len(p_data)):
						if p_data[i][1] == r:
							index = i
							break
					if index != '':
						p_data.pop(i)
				
				#remove gestures from file
				remove_gestures([cleandir + 'gestures-payment-watch.csv', cleandir + 'gestures-payment-ring.csv'], removegestures)
			
			#door gestures
			removegestures = []
			for d in d_data:
				if not '-end' in d[1]:
					if d[2] < minsamplespergesture_door[0] or d[3] < minsamplespergesture_door[1] or d[4] < minsamplespergesture_door[2]:
						removegestures.append(d[1])
			if len(removegestures) > 0:			
				for r in removegestures:
					#remove gesture from list
					index = ''
					for i in range(len(d_data)):
						if d_data[i][1] == r:
							index = i
							break
					if index != '':
						d_data.pop(i + 1)
						d_data.pop(i)
				
				#remove gestures from file
				remove_gestures([cleandir + 'gestures-door-watch.csv', cleandir + 'gestures-door-ring.csv', cleandir + 'gestures-door-door.csv'], removegestures)
		
		#output gesture summaries
		summarise_gestures([cleandir + 'gestures-payment-watch.csv', cleandir + 'gestures-payment-ring.csv', cleandir + 'gestures-door-watch.csv', cleandir + 'gestures-door-ring.csv', cleandir + 'gestures-door-door.csv'])
		
		#output timestamps for debugging
		p_outfilename = cleandir + 'timestamps-payment.csv'
		p_outfile = open(p_outfilename, 'w')
		p_outfile.write('ORIGINAL_TIMESTAMP,GESTURE,NUMBER_OF_WATCH_SAMPLES,NUMBER_OF_RING_SAMPLES\n')
		p_outfile.write('\n'.join([o[0] + ',' + o[1] + ',' + str(o[2]) + ',' + str(o[3]) for o in p_data]))
		p_outfile.close()
		print('OUTPUT: ' + p_outfilename)
		
		d_outfilename = cleandir + 'timestamps-door.csv'
		d_outfile = open(d_outfilename, 'w')
		d_outfile.write('ORIGINAL_TIMESTAMP,GESTURE,NUMBER_OF_WATCH_SAMPLES,NUMBER_OF_RING_SAMPLES,NUMBER_OF_DOOR_SAMPLES\n')
		d_outfile.write('\n'.join([o[0] + ',' + o[1] + ',' + str(o[2]) + ',' + str(o[3]) + ',' + str(o[4]) for o in d_data]))
		d_outfile.close()
		print('OUTPUT: ' + d_outfilename)