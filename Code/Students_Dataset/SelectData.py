import os
os.system('mkdir -p test train')
os.chdir('SMAI2013StudentsDataset')

roll_nos = [];
for files in os.listdir("."):
	roll_nos.append(files[:9])


from itertools import groupby
freq = [len(list(group)) for key, group in groupby(roll_nos)]
print freq

selected = []
selected_freq = []
s = 0;
for i in range(len(freq)):
	s += freq[i]
	if(freq[i] >= 5):
		selected.append(roll_nos[s-1])
		selected_freq.append(freq[i])

for i in range(len(selected)):
	os.system("cp  `ls| grep " + str(selected[i]) + " | head -1` ../test/ ")
	os.system("cp  `ls| grep " + str(selected[i]) + " | tail -"+ str(selected_freq[i] - 1)+"` ../train/ ")
	
