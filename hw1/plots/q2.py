import numpy as np
import matplotlib.pyplot as plt

N = 6
men_means = (3.86, 0.25, 0.21, 0.19, 0.17, 0.16)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, men_means, width, color='r')

# add some text for labels, title and axes ticks
ax.set_ylabel('Loss')
ax.set_title('Loss vs Learning iterations')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('1', '2', '3', '4', '5', '6'))

#ax.legend(rects1[0], 'Men')



plt.show()
