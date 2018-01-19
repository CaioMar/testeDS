from bs4 import BeautifulSoup
import requests
from collections import Counter
from collections import defaultdict
import matplotlib
matplotlib.rc('font', family='TakaoPGothic')
from matplotlib import pyplot as plt
import string
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler as SS
import sys

def merge_two_dicts(x,y):
	for a, _ in y.items():
		x[a] += y[a]
#		if a in x:
#			x[a] += y[a]
#		else:
#			x[a] = y[a]	
	return x

with open("hiragana","r") as hira:
	hiragana = Counter(ch
			   for line in hira
			   for ch in line
			   if ch)

chapters = []
resultant = defaultdict(int)

comeco = int(sys.argv[1])
fim = int(sys.argv[2])

for x in range(comeco,fim):
	print (x)
	chapters = chapters + [requests.get("http://ncode.syosetu.com/n2267be/" + str(x) +"/")]
	chapters[x-comeco].encoding = 'utf-8'
	chapters[x-comeco] = chapters[x-comeco].text
	chapters[x-comeco] = Counter(ch
			          for line in chapters[x-comeco]
			          for ch in line
			          if ch)
	for character in string.printable:
		chapters[x-comeco].pop(character,None)
	for y, _ in hiragana.items():
		chapters[x-comeco].pop(y, None)

	chapters[x-comeco].pop('\u3000', None)
	resultant = merge_two_dicts(resultant, chapters[x-comeco])	
#	chapters[x-comeco] = chapters[x-comeco].items()
#	print (chapters[x-comeco])
print ("Loop has finished")
resultant = resultant.items()
resultant = sorted(resultant, key = lambda x: x[1], reverse = True)
print ("\n**************************\n")
resultant= [x for x, _ in resultant]
#resultant.sort(reverse = True)
print (resultant)
vectors = []
for d, _ in enumerate(chapters):
	vectors = vectors + [[]]
	for j in resultant:
		if j in chapters[d]:
			vectors[d]=vectors[d]+[chapters[d][j]]
		else:
			vectors[d]=vectors[d]+[0]
vectors=np.array(vectors)
print (vectors)
vectors = SS().fit_transform(vectors)
print (vectors)
pca = PCA(n_components=2)
pca.fit(vectors)
print (pca.components_)
scores = pca.transform(vectors)

labels=[]
for x in range(comeco,fim):
	labels += [x]

plt.scatter(scores[:,0],scores[:,1])

for i, l in enumerate(labels):
	plt.annotate(l,
		xy = ( scores[i,0], scores[i,1] ),
		xytext = (3,-3),
		textcoords = 'offset points')

plt.title('PCA dos kanjis de cada capítulo do Re:zero')
plt.xlabel("PC_1")
plt.ylabel("PC_2")
plt.show()

