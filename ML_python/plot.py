import matplotlib.pyplot as plt
import data
# read data from data.py
white_list = data.white_list
red_list = data.red_list
# white wine plot
target_number = dict()
# count samples number for each quality
for sample in white_list:
    target_value = sample['quality']
    if target_number. has_key(target_value):
        target_number[target_value] += 1
    else:
        target_number[target_value] = 1
x = target_number.keys()
y = target_number.values()
plt.bar(x, y, align="center")
plt.xlabel('Target')
plt.ylabel('Number of White samples')
plt.title('White Wine')
plt.show()
# red wine plot
target_number = dict()
# count samples number for each quality
for sample in red_list:
    target_value = sample['quality']
    if target_number. has_key(target_value):
        target_number[target_value] += 1
    else:
        target_number[target_value] = 1
x = target_number.keys()
y = target_number.values()
plt.bar(x, y, align="center")
plt.xlabel('Target')
plt.ylabel('Number of Red samples')
plt.title('Red Wine')
plt.show()