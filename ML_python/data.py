import csv
# read the csv file about wine
white_list = list()
red_list = list()
with open('winequality-white.csv', 'rb') as WhiteFile:
    reader = csv.reader(WhiteFile)
    count = 0
    for row in reader:
        if count == 0:
            # split the feature by ';'
            Features = row[0].replace('"', '').split(';')
        else:
            Whitedict = dict()
            index = 0
            # store the white dict in list
            while index < len(Features):
                # split the value by ';'
                Whitedict[Features[index]] = float(row[0].split(';')[index])
                index += 1
            white_list.append(Whitedict)
        count += 1

with open('winequality-red.csv', 'rb') as RedFile:
    reader = csv.reader(RedFile)
    count = 0
    for row in reader:
        if count == 0:
            # split the feature by ';'
            Features = row[0].replace('"', '').split(';')
        else:
            Reddict = dict()
            index = 0
            # store the red dict in list
            while index < len(Features):
                # split the value by ';'
                Reddict[Features[index]] = float(row[0].split(';')[index])
                index += 1
            red_list.append(Reddict)
        count += 1

