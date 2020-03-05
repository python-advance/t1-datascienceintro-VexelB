import pandas

parced_csv = pandas.read_csv('train.csv', sep=',', index_col='PassengerId', quotechar='"')

def sexcount():
    a = parced_csv[parced_csv['Sex'] == "male"]
    b = parced_csv[parced_csv['Sex'] == "female"]
    return len(a), len(b)

def portcount():
    a, b, c = parced_csv["Embarked"].value_counts()
    return a, b, c

def surviveperc():
    a = parced_csv["Survived"].value_counts()[0]
    return a, int((a/len(parced_csv)*100)*10**3)/10**3

def classsystem():
    a, b, c = parced_csv["Pclass"].value_counts()
    return b, c, a

def sibsptoParch():
    return parced_csv["SibSp"].corr(parced_csv["Parch"], method = 'pearson')

def agetosurv():
    return parced_csv["Age"].corr(parced_csv["Survived"], method = 'pearson')

def sextosurv():
    return parced_csv["Sex"].corr(parced_csv["Survived"], method = "pearson") # надо пеервести sex в бул

def classtosurv():
    return parced_csv["Pclass"].corr(parced_csv["Survived"], method = "pearson")

def passage():
    avg_age = parced_csv['Age'].mean(skipna=True) 
    median_age = parced_csv['Age'].median(skipna=True)
    return avg_age, median_age

def avgprice():
    avg_fare = parced_csv['Fare'].mean(skipna=True) 
    median_fare = parced_csv['Fare'].median(skipna=True)
    return avg_fare, median_fare

def manname():
    top_male_name = pandas.Series([i.split(', ')[1] for i in parced_csv['Name']]).value_counts().index[0]
    return top_male_name

def manfemnames():
    male_sorted_names = pandas.Series([i.split(', ')[1] for i in parced_csv[(parced_csv['Age'] > 15) & (parced_csv['Sex'] == 'male')]['Name']]).value_counts() 
    female_sorted_names = pandas.Series([i.split(', ')[1] for i in parced_csv[(parced_csv['Age'] > 15) & (parced_csv['Sex'] == 'female')]['Name']]).value_counts() 
    return male_sorted_names.index[0], female_sorted_names.index[0]
# print(sexcount())
# print(portcount())
# print(surviveperc())
# print(classsystem())
# print(sibsptoParch())
# print(agetosurv())
# print(sextosurv())
# print(classtosurv())
# print(passage())
# print(avgprice())
# print(manname())
# print(manfemnames())
