import sys
import csv

metadata_fn = 'metadata.csv'
titles_fn = 'positive.txt'

if len(sys.argv) > 1:
	titles_fn = sys.argv[1]

missed_count = 0
metadata = []
with open(metadata_fn, 'r', encoding = 'utf-8') as f:
	csv.field_size_limit(999999999)
	for row in csv.reader(f, delimiter='\n', quotechar='#'):
		try:
			beforedoi = row[0].split("/")[0]
			if beforedoi.find('"') > -1:
				chunk = beforedoi.split('"')
				firstchunk = chunk[0].split(',')
				if firstchunk[1] == '': #no sha
					firstchunk[1] = "None"
				title = chunk[1]
				title.lstrip(" ")
				title.rstrip(" ")
				firstchunk.append(title)
				metadata.append(firstchunk)
			else:
				chunk = beforedoi.split(",")
				if chunk[1] == '': # no sha
					chunk[1] = "None"
				chunk[3].lstrip(" ")
				chunk[3].rstrip(" ")
				metadata.append(chunk)

			metadata[len(metadata)-1].remove('')
		except:
			missed_count += 1

# print(metadata[len(metadata)-1])
print(missed_count)
	
with open(titles_fn, 'r', encoding="utf-8") as f:
	titles = f.read().split('\n')

for row in metadata:
	try:
		if (row[3].find("acute lung injury in a mouse model of pulmonary fibrosis") > -1):
			print(row)
	except:
		pass

hashes = []
for title in titles:
	title.lstrip(" ")
	title.rstrip(" ")
	found = False
	for row in metadata:
		if len(row) > 3:
			# if(row[3].find("acute lung injury in a mouse model of pulmonary fibrosis") > -1):
			# 	print(row[3])
			# 	print(title)
			# 	print(row[1])
			if row[3].find(title[:30]) > -1:
				#hashes.append(row[1] + ','+title) #debug
				hashes.append(row[1])
				found = True
				break
	if not found:
		#hashes.append(','+title) #debug
		hashes.append("None")
		
titles_out_fn = titles_fn[:-4] + '_out.txt'

with open(titles_out_fn, 'w') as f:
	f.write('\n'.join(hashes))