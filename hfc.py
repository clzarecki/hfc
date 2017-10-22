import csv
import datetime
import math
import numpy as np
import re
from sklearn import svm
import time

all_features = ['GWNO', 'EVENT_ID_CNTY', 'EVENT_ID_NO_CNTY', 'EVENT_DATE',
		'YEAR', 'TIME_PRECISION', 'EVENT_TYPE', 'ACTOR1', 'ALLY_ACTOR_1',
		'INTER1', 'ACTOR2', 'ALLY_ACTOR_2', 'INTER2', 'INTERACTION', 'COUNTRY',
		'ADMIN1', 'ADMIN2', 'ADMIN3', 'LOCATION', 'LATITUDE', 'LONGITUDE',
		'GEO_PRECISION', 'SOURCE', 'NOTES', 'FATALITIES']

final_features = ['GWNO', 'EVENT_DATE', 'YEAR', 'TIME_PRECISION', 'EVENT_TYPE',
		'ACTOR1', 'ALLY_ACTOR_1', 'INTER1', 'ACTOR2', 'ALLY_ACTOR_2', 'INTER2',
		'INTERACTION', 'COUNTRY', 'LATITUDE', 'LONGITUDE', 'GEO_PRECISION',
		'FATALITIES']
final_features2 = ['EVENT_DATE', 'YEAR', 'TIME_PRECISION', 'EVENT_TYPE',
		'INTER1', 'INTER2',
		'INTERACTION', 'COUNTRY', 'LATITUDE', 'LONGITUDE', 'GEO_PRECISION',
		'FATALITIES']

def get_feature_names(filename):
	with open(filename, "rb") as f:
		reader = csv.reader(f)
		i = reader.next()
		rest = [row for row in reader]
	return i

def date_str_to_int(date_str):
	date_format = '%m/%d/%Y'
	date = datetime.datetime.strptime(date_str, date_format)
	year = date.year
	month = date.month
	day = date.day
	date = (year - 1990) * 365 + month * 30 + day
	return date

def get_features_and_possible_values(filename):
	#feature_names = get_feature_names(filename)
	feature_names = final_features2
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
	col_names = []
	for feat in feature_names:
		if feature_is_numeric[feat]:
			col_names.append(feat)
		else:
			for val in possible_values[feat]:
				col_names.append(feat + "=" + val)
	return col_names

def normalize_feature(value, feature_is_numeric, possible_values, min, max):
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
	row_normalized_features = []
	for feature in features:
		feat_normalized = normalize_feature(row[feature],
				features_numeric[feature], possible_values[feature],
				min_vals.get(feature), max_vals.get(feature))
		row_normalized_features.extend(feat_normalized)
	return row_normalized_features

def get_normalized_col_names(filename):
	feat_names, pos_feat_vals = get_features_and_possible_values(filename)
	feat_numeric = feature_is_numeric(pos_feat_vals)
	max_vals, min_vals = max_min_vals(pos_feat_vals, feat_numeric)
	
	col_names = normalize_feature_names(feat_names, feat_numeric, pos_feat_vals)
	return col_names

def row_vals_to_float(row):
	float_row = {}
	float_match = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$').match
	for key,val in row.iteritems():
		if float_match(val):
			float_row[key] = float(val)
		else:
			float_row[key] = val
	return float_row

def get_data(filename):
	data = []
	with open(filename) as csvfile:
		reader = csv.DictReader(csvfile)
		min = 3000
		for row in reader:
			row["EVENT_DATE"] = str(date_str_to_int(row["EVENT_DATE"]))
			data.append(row_vals_to_float(row))
	return data

def split_by_field(data, field_name):
	split_data = {}
	for row in data:
		field_val = row[field_name]
		if field_val in split_data:
			split_data[field_val].append(row)
		else:
			split_data[field_val] = [row]
	return split_data

def compare_by_date(row1, row2):
	row1_date = row1['EVENT_DATE']
	row2_date = row2['EVENT_DATE']
	return int(row1_date) - int(row2_date)

def get_features(data, index, num_previous):
	if num_previous > index:
		return
	
	feature_vals = []
	
	for add_index in range(index - num_previous, index + 1):
		feature_vals += data[add_index]
	
	return feature_vals

def split_by_attr(data, labels, attr_index, split_value):
	below_data = []
	below_labels = []
	above_data = []
	above_labels = []
	
	for point, label in zip(data, labels):
		if point[attr_index] < split_value:
			below_data.append(point)
			below_labels.append(label)
		else:
			above_data.append(point)
			above_labels.append(label)
	
	return below_data, below_labels, above_data, above_labels

def entropy(labels):
	# get counts
	label_choices = list(set(labels))
	total_count = len(labels)
	label_counts = [0] * len(label_choices)
	for label in labels:
		label_counts[label_choices.index(label)] += 1
	
	# calculate entropy
	entropy = 0
	for count in label_counts:
		prob = float(count) / total_count
		entropy -= prob * math.log(prob, 2)
	
	return entropy

def find_best_attribute(training_data, svm_classifier, invalid_attrs):
	# get label for training points using svm
	labels = svm_classifier.predict(training_data)

	best_gain = 0
	best_attr = -1
	for attr_index in range(len(training_data[0])):
		if attr_index in invalid_attrs:
			continue
		
		# find best information gain
		part1_data, part1_labels, part2_data, part2_labels = split_by_attr(
				training_data, labels, attr_index, 0.5)
		gain = entropy(labels) - ((len(part1_labels) / len(labels)) * entropy(part1_labels) + (len(part2_labels) / len(labels)) + entropy(part2_labels))
		
		if gain > best_gain:
			best_gain = gain
			best_attr = attr_index
	
	return best_attr

def main():
	filename = "ACLED-All-Africa-File_20170101-to-20170923_csv.csv"
	
	print("Getting data from file...")
	data = get_data(filename)
	data = split_by_field(data, 'COUNTRY')
	
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
	
	print("Preparing training data...")
	training_data = []
	training_labels = []
	num_previous = 1
	for country, rows in data_normalized.iteritems():
		for index in range(num_previous, len(rows)):
			training_point = get_features(rows, index, num_previous)
			label = min(1, data[country][index]['FATALITIES'])
			training_data.append(training_point)
			training_labels.append(label)

	print("Training classifier...")
	clf = svm.LinearSVC()
	clf.fit(training_data, training_labels)

	print("Building tree from SVM classifier...")
	invalid_attrs = set(range(18,65))
	num_attrs = len(column_names)
	for n in range(num_previous):
		invalid_attrs |= set(range(num_attrs * (n+1) + 18, num_attrs * (n+1) + 65))
	invalid_attrs |= set(range(num_previous * num_attrs, (num_previous + 1) * num_attrs))
	
	attr = find_best_attribute(training_data, clf, invalid_attrs)
	print(str(attr // num_attrs) + " " + column_names[attr % num_attrs])

if __name__ == '__main__':
	main()
