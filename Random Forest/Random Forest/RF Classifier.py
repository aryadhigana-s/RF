import csv

def bagi(array):
    new = []
    for i in range(len(array)):
        new.append(array[i][-1])
        del array[i][-1]
    return new

results = []
with open("nilai batas (edit).csv",newline= '') as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_ALL)
    for row in reader:
        results.append(row)

for i in range(len(results)):
    [x for xs in results[i] for x in xs.split(',')]
    print(results[i])

dataTrain = results[0:79]
dataTest = results[79:]


label_dataTrain = bagi(dataTrain)
label_dataTest = bagi(dataTest)
#print(len(dataTrain))
#print(len(dataTest))