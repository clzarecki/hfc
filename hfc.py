import csv
import datetime
import graphviz
import numpy as np
import random
import re
from sklearn import svm, tree

all_features = ['GWNO', 'EVENT_ID_CNTY', 'EVENT_ID_NO_CNTY', 'EVENT_DATE',
		'YEAR', 'TIME_PRECISION', 'EVENT_TYPE', 'ACTOR1', 'ALLY_ACTOR_1',
		'INTER1', 'ACTOR2', 'ALLY_ACTOR_2', 'INTER2', 'INTERACTION', 'COUNTRY',
		'ADMIN1', 'ADMIN2', 'ADMIN3', 'LOCATION', 'LATITUDE', 'LONGITUDE',
		'GEO_PRECISION', 'SOURCE', 'NOTES', 'FATALITIES']

final_features = ['EVENT_DATE', 'YEAR', 'TIME_PRECISION', 'EVENT_TYPE',
		'INTER1', 'INTER2',
		'INTERACTION', 'COUNTRY', 'LATITUDE', 'LONGITUDE', 'GEO_PRECISION',
		'FATALITIES']

ignored_features = ['GWNO', 'EVENT_ID_CNTY', 'EVENT_ID_NO_CNTY',
		'ACTOR1', 'ALLY_ACTOR_1',
		'ACTOR2', 'ALLY_ACTOR_2',
		'ADMIN1', 'ADMIN2', 'ADMIN3', 'LOCATION', 
		'SOURCE', 'NOTES'
		#, 'COUNTRY'
		]

def get_feature_names(filename):
	"""Get feature names from a csv file
	Arguments:
	filename ­­ the path/name of the csv file
	Returns:
	A list of strings as feature names
	"""
	with open(filename, "rb") as f:
		reader = csv.reader(f)
		i = reader.next()
		rest = [row for row in reader]
	return i

def date_str_to_int(date_str):
	"""Convert a date string to days since 1/1/1997
	Arguments:
	date_str ­­ date as a string in MM/DD/YYYY format
	Returns:
	an integer of the number of days since 1/1/1997
	"""
	date_format = '%m/%d/%Y'
	date = datetime.datetime.strptime(date_str, date_format)
	num_days = (date - datetime.datetime(1997,1,1)).days
	return num_days
	
def int_to_date(num_days):
	"""Convert num of days since 1/1/1997 to a date
	Arguments:
	num_days ­­ number of days since 1/1/1997 as an integer
	Returns:
	The converted date object
	"""
	date = datetime.datetime(1997,1,1) + datetime.timedelta(days=num_days)
	return date

def num_months_between_dates(date1, date2):
	"""Calculates the number of months between two dates as an absolute value
	Arguments:
	date1 -- the first date object
	date2 -- the second date object
	Returns:
	The number of months as an integer
	"""
	return np.absolute((date1.year - date2.year) * 12 + (date1.month - date2.month))

def num_events_per_month(events):
	"""Calculate the average number of events per month from a list of events
	Arguments:
	events ­­ a list of events, in order by date
	Returns:
	The average number of events per month, as a float
	"""
	first_event_date = int_to_date(events[0]["EVENT_DATE"])
	last_event_date = int_to_date(events[-1]["EVENT_DATE"])
	num_months = num_months_between_dates(last_event_date, first_event_date)
	num_events = len(events)
	if num_months == 0:
		num_months = 1
	return num_events / float(num_months)
	
def get_features_and_possible_values(filename):
	"""Get possible values for a set of features in a csv file. The features to
	get values for are in the final_features list. The EVENT_DATE feature is
	converted to an integer.
	Arguments:
	filename ­­ the path/name of the csv file
	Returns:
	A dictionary of lists, with key as feature name and value as a list of
	strings or integers, in sorted order
	"""
	#feature_names = get_feature_names(filename)
	feature_names = final_features
	feat_vals = {}
	for feat in feature_names:
		feat_vals[feat] = set()
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			for feat in feature_names:
				val = row[feat]
				if feat == "EVENT_DATE":
					val = date_str_to_int(val)
				feat_vals[feat].add(val)
	
	# convert set of values to sorted list
	for feature in feature_names:
		feat_vals[feature] = sorted(list(feat_vals[feature]))
	
	return feature_names, feat_vals

def feature_is_numeric(possible_values):
	"""Determines if a feature is numeric, depending on feature values
	Arguments:
	possible_values ­­ a dictionary of feature name to list of values a feature
	can take
	Returns:
	A dictionary of feature name to boolean, True if all possible feature
	values are castable as float, False otherwise
	"""
	numeric = {}
	for name, values in possible_values.iteritems():
		feat_values = np.array(values)
		try:
			feat_values.astype(np.float)
			numeric[name] = True
		except ValueError:
			numeric[name] = False
	return numeric

def max_min_vals(possible_values, feature_numeric):
	"""Finds the maximum and minimum value for each numeric feature
	Arguments:
	possible_values ­­ a dictionary of feature name to list of values a feature
	can take
	feature_numeric -- a dictionary of feature name to boolean, whether or not
	a feature is numeric
	Returns:
	A dictionary of feature name to float, the maximum feature value
	A dictionary of feature name to float, the minimum feature value
	"""
	max_vals = {}
	min_vals = {}
	for feat_name, is_numeric in feature_numeric.iteritems():
		if is_numeric:
			feat_values = np.array(possible_values[feat_name]).astype(np.float)
			max_val = max(feat_values)
			min_val = min(feat_values)
			max_vals[feat_name] = max_val
			min_vals[feat_name] = min_val
	return max_vals, min_vals
	
def normalize_feature_names(feature_names, feature_is_numeric, possible_values):
	"""Get list of features names, normalized for numeric classifiers. Numeric
	features just get the feature name. Discrete-valued features are expanded
	to a list of one-hot features, formatted as FEAT_NAME=VALUE
	Arguments:
	feature_names ­­ a list of feature names as strings
	feature_is_numeric -- a dictionary of string feature names to boolean
	values, whether or not a given feature is numeric
	possible_values -- a dictionary of string feature names to list of sorted
	possible feature values
	Returns:
	A list of feature names, normalized to include one-hot representations of
	discrete features
	"""
	col_names = []
	for feat in feature_names:
		if feat in ignored_features:
			continue
		if feature_is_numeric[feat]:
			col_names.append(feat)
		else:
			for val in possible_values[feat]:
				col_names.append(feat + "=" + val)
	return col_names

def normalize_feature(value, feature_is_numeric, possible_values, min, max):
	"""Normalizes a feature value to be between 0 and 1
	Arguments:
	value ­­ the given feature value
	feature_is_numeric -- True if the feature from the given value is numeric
	possible values -- a sorted list of possble values from the feature the
	value came from
	min -- float for the minimum value from the feature the given value came
	from
	max -- float for the maximum value from the feature the given value came
	from
	Returns:
	Float for the normalized feature value, 0 to 1 normalized between the
	minimum and the maximum value for numeric features, otherwise a one-hot
	list of the represented discrete value
	"""
	if feature_is_numeric:
		float_val = float(value)
		float_max = float(max)
		float_min = float(min)
		range = float_max - float_min
		normalized = (float_val - float_min) / range
		return [normalized]
	else:
		num_values = len(possible_values)
		index = possible_values.index(value) # already sorted
		expanded_feature = [0] * num_values
		expanded_feature[index] = 1
		return expanded_feature

def get_row_normalized(row, features, features_numeric, possible_values,
		min_vals, max_vals):
	"""Normalizes a row so all values are between 0 and 1
	Arguments:
	row ­­ raw values from the event
	features -- a list of features to include
	features_numeric -- a dictionary of string feature names to boolean
	values, whether or not a given feature is numeric
	possible values -- a dictionary of string feature names to list of sorted
	possible feature values
	min_values -- a dictionary of string feature names to minimum value float
	max -- a dictionary of string feature names to maximum value float
	Returns:
	A list of 0 to 1 normalized floats for the event, each feature either being
	0 to 1 normalized between the minimum and the maximum value for numeric
	features, or a one-hot list of the represented discrete value
	"""
	row_normalized_features = []
	for feature in features:
		if feature in ignored_features:
			continue
		feat_normalized = normalize_feature(row[feature],
				features_numeric[feature], possible_values[feature],
				min_vals.get(feature), max_vals.get(feature))
		row_normalized_features.extend(feat_normalized)
	return row_normalized_features

def get_normalized_col_names(filename):
	"""Get list of features names from a csv file, normalized for numeric
	classifiers. Numeric features just get the feature name. Discrete-valued
	features are expanded to a list of one-hot features, formatted as
	FEAT_NAME=VALUE
	Arguments:
	filename -- the csv file/path as a string
	Returns:
	A list of feature names, normalized to include one-hot representations of
	discrete features
	"""
	feat_names, pos_feat_vals = get_features_and_possible_values(filename)
	feat_numeric = feature_is_numeric(pos_feat_vals)
	max_vals, min_vals = max_min_vals(pos_feat_vals, feat_numeric)
	
	col_names = normalize_feature_names(feat_names, feat_numeric, pos_feat_vals)
	return col_names

def row_vals_to_float(row):
	"""Convert a dictionary's string values to float. If a string isn't able to
	be converted to float, it is retained as-is, otherwise the float value is
	used instead.
	Arguments:
	row -- dictionary of string feature name to string values
	Returns:
	The converted dictionary of feature name to values, each being either a
	string or float
	"""
	float_row = {}
	float_match = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$').match
	for key,val in row.iteritems():
		if float_match(val):
			float_row[key] = float(val)
		else:
			float_row[key] = val
	return float_row

def get_data(filename):
	"""Get data from a csv file, each value converted to float if possible.
	Arguments:
	filename -- the file/path to the csv file
	Returns:
	A list of rows, each row being a dictionary of feature name keys to string
	or float values from the file
	"""
	data = []
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		min = 3000
		for row in reader:
			row["EVENT_DATE"] = str(date_str_to_int(row["EVENT_DATE"]))
			data.append(row_vals_to_float(row))
	return data

def split_by_field(data, field1, field2):
	"""Split data into a combination of two different fields.
	Arguments:
	data -- A list of rows, each row being a dictionary of feature name keys to
	string or float values
	field1 -- the first field to split on
	field2 -- the second field to split on
	Returns:
	The separated data, as a dictionary. Each key is a combination of field
	values, contatenated together. The values are a list of events, each event
	being a dictionary of feature name to value.
	"""
	split_data = {}
	for row in data:
		key = row[field1] + row[field2]
		if key in split_data:
			split_data[key].append(row)
		else:
			split_data[key] = [row]
	return split_data

def compare_by_date(row1, row2):
	"""Compare two events by EVENT_DATE field name
	Arguments:
	row1 -- the first event, with the EVENT_DATE field being a numerical value
	row2 -- the second event, with the EVENT_DATE field being a numerical value
	Returns:
	A negative value if row1 is before row2, 0 if the rows have equal dates, a
	positive value if row1 is after row2
	"""
	row1_date = row1['EVENT_DATE']
	row2_date = row2['EVENT_DATE']
	return int(row1_date) - int(row2_date)

def get_features(data, index, num_previous):
	"""Get a list of consecutive events concatenated together, to be used in a
	predictive classifier for the next event after the last one
	Arguments:
	data -- a list of events sorted by date, each event being a list of 0/1
	normalized values
	index -- the index of the event after the last event to include in the list
	num_previous -- the number of events to concatenate together
	Returns:
	A list of 0/1 normalized values with length=(len(data[0]) * num_previous).
	The list is a concatenated list of num_previous events before the given
	index.
	"""
	if num_previous > index:
		return
	
	feature_vals = []
	
	for add_index in range(index - num_previous, index):
		feature_vals += data[add_index]
	
	return feature_vals

def num_percent_str(num, total):
	"""Format a number as the number and a percentage of the total, rounded to
	two decimal places and separated by a tab.
	Arguments:
	num: the number
	total: the total number possible
	Returns:
	A formatted string as the number, a tab, then the percentage of the total
	"""
	return str(num) + "\t" + str("%.2f" % (float(num) * 100 / total)) + "%"

def find_between(s, first, last):
	"""In a string, get the value between the first instance of two values. For
	example, if the string is "Hello World!", the first value is "l", and the
	last value is "d", the returned value would be "lo Worl"
	Arguments:
	s -- the string to find the substring in
	first -- the value immediately preceding the substring
	last -- the value immediately following the substring
	Returns:
	The substring as described
	"""
	try:
		start = s.index( first ) + len( first )
		end = s.index( last, start )
		return s[start:end]
	except ValueError:
		return ""

def node_json(nodenum, treedata):
	"""Generate the JSON string representation of a tree, with "name" being the
	value of the node, and "children" being a JSON list of the node's children.
	Arguments:
	nodenum -- the index of the root
	treedata -- an array representation of a tree, each entry being a tuple of
	three values: the node value, the index of the left child, the index of the
	right child
	Returns:
	JSON string of the tree
	"""
	nodedata = treedata[nodenum]
	has_discrete_question = False
	if nodedata[0].find("<=") < 0:
		has_discrete_question = True
	json = '"name": "' + nodedata[0] + '"'
	if nodedata[1] == None:
		# leaf node
		json = '"name": "' + nodedata[0] + ' Fatalities"'
		return json
	child1json = node_json(nodedata[1], treedata)
	child2json = node_json(nodedata[2], treedata)
	if has_discrete_question:
		json += ',"children": [{' + child2json + '},{' + child1json + '}]'
	else:
		json += ',"children": [{' + child1json + '},{' + child2json + '}]'
	return json

def get_tree_json(decision_tree_data):
	"""Generate an array representation of a tree from a graphviz string
	representation of that tree.
	Arguments:
	decision_tree_data -- a string of the graphviz representation of a tree
	Returns:
	the array representation of a tree, each entry being a tuple of three
	values: the node value, the index of the left child, the index of the right
	child
	"""
	# build tree
	treelines = decision_tree_data.split('\n')
	nodes = {}
	for line in treelines:
		firstchar = line[0]
		if not firstchar.isdigit():
			continue
		lineparts = line.split()
		nodenum = int(lineparts[0])
		charAfterSpace = lineparts[1][0]
		if charAfterSpace == '[':
			# get node question and value
			
			info = find_between(line, "[", "]")
			info = find_between(info, '"', '"')
			infovals = info.split("\\n")
			valinfo = infovals[-1]
			if len(infovals) == 3:
				# internal node
				question = infovals[0]
				pointfiveindex = question.find("<= 0.5")
				if pointfiveindex >= 0:
					question = question[:pointfiveindex].strip()
				nodes[nodenum] = (question, None, None)
			else:
				# leaf node
				val = valinfo[valinfo.index("=") + 2:]
				nodes[nodenum] = (val, None, None)
		if charAfterSpace == "-":
			child = int(lineparts[2])
			parentnode = nodes[nodenum]
			if parentnode[1] == None:
				nodes[nodenum] = (parentnode[0], child, None)
			else:
				nodes[nodenum] = (parentnode[0], parentnode[1], child)
	return "{" + node_json(0, nodes) + "}"

def main():
	filename = "ACLED-All-Africa-File_20170101-to-20170923_csv.csv"
	
	print("Getting data from file...")
	data = get_data(filename)
	
	# get a histogram of number of fatalities for events
# 	fatalities_freq = {}
# 	for event in data:
# 		fatalities = event["FATALITIES"]
# 		if fatalities not in fatalities_freq:
# 			fatalities_freq[fatalities] = 0.0
# 		fatalities_freq[fatalities] += 1
# 	print fatalities_freq
	
	data = split_by_field(data, 'COUNTRY', 'EVENT_TYPE')
	
	# sort by date
	for country, rows in data.iteritems():
		data[country] = sorted(rows, compare_by_date)
	
	print("Normalizing data...")
	feat_names, pos_feat_vals = get_features_and_possible_values(filename)
	feat_numeric = feature_is_numeric(pos_feat_vals)
	max_vals, min_vals = max_min_vals(pos_feat_vals, feat_numeric)
	
	column_names = get_normalized_col_names(filename)
	data_normalized = {}
	for country, rows in data.iteritems():
		country_rows_normalized = []
		for row in rows:
			row_normalized = get_row_normalized(row, feat_names, feat_numeric,
					pos_feat_vals, min_vals, max_vals)
			country_rows_normalized.append(row_normalized)
		data_normalized[country] = country_rows_normalized
	
	"""
	Choose the number of previous events to use as features in the classifier
	"""
# 	for num_previous in [1,2,3,4,5,10]:
	for num_previous in [5]:
		print("Preparing train/test data...")
		X = []
		Y = []
		for country, rows in data_normalized.iteritems():
			events_per_month = num_events_per_month(data[country])
			for index in range(num_previous, len(rows)):
				num_fatalities = data[country][index]['FATALITIES']
				"""
				Uncomment the following lines to ignore events with number of
				fatalities between 0 and 5 exclusive
				"""
# 				if num_fatalities > 0 and num_fatalities < 5:
# 					continue
				training_point = get_features(rows, index, num_previous)
				"""
				Choose between the number of fatalities or whether any
				fatalities occurred as a label
				"""
# 				label = min(1, num_fatalities)
				label = np.round(num_fatalities * events_per_month)
				X.append(training_point)
				Y.append(label)
		
		# prepare column names
		final_columns = []
		for num in range(num_previous):
			for name in column_names:
				final_columns.append("prev#" + str(num+1) + " " + name)
		
		# randomly create train and test set
		XY = zip(X,Y)
		random.seed(0)
		random.shuffle(XY)
		split = int(np.floor(0.8 * len(X)))
		XY_train = XY[:split]
		XY_test = XY[split:]
		train_data, train_labels = zip(*XY_train)
		test_data, test_labels = zip(*XY_test)

		print("Training classifier...")
		svmclf = svm.LinearSVC()
		svmclf.fit(train_data, train_labels)

		print("Building tree from SVM classifier...")
		"""
		Choose whether to use a classifier or regressor, dependent on the label
		being a boolean or real value
		"""
# 		dt = tree.DecisionTreeClassifier(max_depth=3)
		dt = tree.DecisionTreeRegressor(max_depth=3)
		dt = dt.fit(train_data, svmclf.predict(train_data))
		dot_data = tree.export_graphviz(dt, out_file=None,feature_names=final_columns,class_names=["Fatalities=0","Fatalities>=1"],impurity=False)
		graph = graphviz.Source(dot_data)
		graph.render(str(num_previous) + "prev")
		
		print get_tree_json(dot_data)
		
		"""
		Testing code for classification (determining whether the next event
		will have any fatalities
		Note: currently only works with 0/1 classification testing
		"""
# 		print("Testing...")
# 		svm_test_pred = svmclf.predict(test_data)
# 		svm_test_pred = np.array(svm_test_pred)
# 		dt_test_pred = dt.predict(test_data)
# 		dt_test_pred = np.array(dt_test_pred)
# 		test_labels = np.array(test_labels)
# 		svm_num_equal = np.sum(svm_test_pred == test_labels)
# 		dt_num_equal = np.sum(dt_test_pred == test_labels)
# 	
# 		print("Results for using previous " + str(num_previous) + " events:")
# 		print("Num training\t" + num_percent_str(len(train_data), len(X)))
# 		print("Num test\t" + num_percent_str(len(test_data), len(X)))
# 		print("Negative total\t" + num_percent_str(np.sum(np.array(Y) == 0), len(X)))
# 		print("Positive total\t" + num_percent_str(np.sum(np.array(Y) == 1), len(X)))
# 		print("Negative test\t" + num_percent_str(np.sum(np.array(test_labels) == 0), len(test_labels)))
# 		print("Positive test\t" + num_percent_str(np.sum(np.array(test_labels) == 1), len(test_labels)))
# 	
# 		print "SVM accuracy\t" + num_percent_str(svm_num_equal,len(test_labels))
# 		print "Tree accuracy\t" + num_percent_str(dt_num_equal,len(test_labels))
# 	
# 		# precision and recall
# 		truepos = len(np.intersect1d(np.where(svm_test_pred == 1),np.where(test_labels == 1)))
# 		falsepos = len(np.intersect1d(np.where(svm_test_pred == 1),np.where(test_labels == 0)))
# 		trueneg = len(np.intersect1d(np.where(svm_test_pred == 0),np.where(test_labels == 0)))
# 		falseneg = len(np.intersect1d(np.where(svm_test_pred == 0),np.where(test_labels == 1)))
# 		precision = float(truepos) / (truepos + falsepos)
# 		recall = float(truepos) / (truepos + falseneg)
# 		f1 = 2 * precision * recall / (precision + recall)
# 		print("Precision\t" + ("%.4f" % precision))
# 		print("Recall\t\t" + ("%.4f" % recall))
# 		print("F1\t\t" + ("%.4f" % f1))

if __name__ == '__main__':
	main()
