from surprise import Reader
from surprise import Dataset
import matplotlib.pyplot as plt
import csv

distribution = [0,0,0,0,0]
with open('./ml-latest-parsed.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        rating = row[2]
        if rating == "1.0":
            distribution[0] += 1
        if rating == "2.0":
            distribution[1] += 1
        if rating == "3.0":
            distribution[2] += 1
        if rating == "4.0":
            distribution[3] += 1
        if rating == "5.0":
            distribution[4] += 4

plt.bar([1,2,3,4,5],distribution)
plt.xlabel("Movie ratings")
plt.ylabel("Ratings count")
plt.title("Movie ratings vs. # of counts")
plt.show()


