import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# CSV file load karo
data = pd.read_csv("spam.csv", encoding="latin-1")

# Sirf 2 columns chahiye
data = data[["v1", "v2"]]

# Columns ka naam theek karo
data.columns = ["label", "message"]

# Kitne spam kitne ham
print(data["label"].value_counts())

# Messages aur labels alag karo
x = data["message"]
y = data["label"]

# 80% train 20% test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Text ko numbers ki table mein badla — taaki AI samajh sake wo naive bayes wala concept use kar sake
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)


# Model banaya
model = MultinomialNB()
#fit matlab model ko training data se sikhana
model.fit(x_train_cv, y_train)

#test the model
predictions = model.predict(x_test_cv)
print(predictions)
#eder pehli baar mery model ne khud se predict kiya hai ki konsa message spam hai aur konsa ham, ab hum dekhte hain ki model kitna sahi predict kar raha hai

#accuracy_score se hum dekh sakte hain ki model kitna sahi predict kar raha hai
print(accuracy_score(y_test, predictions))

message = ["my name is wajid and i am a software engineer"]
message_cv = cv.transform(message)
result = model.predict(message_cv)
print(result)