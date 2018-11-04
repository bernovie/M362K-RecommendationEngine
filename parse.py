original = open('/Users/berny/Desktop/ml-latest/ratings.csv', "r")
parsed = open('ml-latest-parsed.csv', "w+")
new_users = open('./ml-latest-small/Movie_Ratings_from_Survey.csv')

# Constants
# 736
NUMBER_OF_RATINGS_PER_USER = 1000
NUMBER_OF_USERS = 5000

parsed.write(original.readline())
prev_line = original.readline().split(",")
line = original.readline().split(",")
parsed.write(",".join(prev_line))
rating = 0
ratings_count = 0
# Parse the 22 Million dataset to the number of users we want
for user in range(0, NUMBER_OF_USERS):
    while (line[0] == prev_line[0] and rating < NUMBER_OF_RATINGS_PER_USER ):
        parsed.write(",".join(line))
        prev_line = line
        line = original.readline().split(",")
        rating += 1

    ratings_count += rating

    if not line[0] == prev_line[0] and not user == NUMBER_OF_USERS - 1 :
        parsed.write(",".join(line))
        prev_line = line
        line = original.readline().split(",")
        rating = 1

    if rating >= NUMBER_OF_RATINGS_PER_USER:
        prev_line = line
        line = original.readline().split(",")
        while(line[0] == prev_line[0]):
            prev_line = line
            line = original.readline().split(",")
        if not user == NUMBER_OF_USERS - 1:
            parsed.write(",".join(line))
            prev_line = line
            line = original.readline().split(",")
            rating = 1

# Add the new users from the survey
prev_line = new_users.readline()
prev_line = prev_line.split(",")
line = new_users.readline()
line = line.split(",")
new_user_count = NUMBER_OF_USERS + 1
while("".join(line) != ""):
    while (line[0] == prev_line[0]):
        prev_line[0] = str(new_user_count)
        prev_line = ",".join(prev_line)
        parsed.write(prev_line)
        prev_line = line
        line = (new_users.readline()).split(",")
        ratings_count += 1

    if not line[0] == prev_line[0]:
        prev_line[0] = str(new_user_count)
        ratings_count += 1
        parsed.write(",".join(prev_line))
        prev_line = line
        line = new_users.readline().split(",")
    new_user_count += 1

print("Number of Ratings: ", ratings_count + 1)
# Average #movies rated by user (# of ratings/# users):
print("Data density : ", round((ratings_count+1)/(NUMBER_OF_USERS+44), 3))
# New Data set has 464,515 tags distributed over 3,500 users
parsed.close()
new_users.close()
original.close()
