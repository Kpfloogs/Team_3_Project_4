# Team_3_Project_4 | Stroke Predictor
We plan to analyze a kaggle datasource around health factors that may cause a stroke. In our analysis we hope to uncover medical factors that lead to strokes, such as gender, is the patient has had heart disease, hypertension, etc. Then creating a user input menu of dropdowns that when using our supervised learning model will determine from the users inputs if they are likely to have a stroke.

![noStroke Risk](https://user-images.githubusercontent.com/113635771/226497313-c4a66004-0a2c-44d3-87ce-0d7a7697a383.png)

## Project Team
- Kathleen Pflugi
- Lauren Cadogan
- Megan Blazevic
- Satya Sakuntala NagaSravya Mattapalli

## Data & Delivery

### Data Set
Kaggle Data Set | Stroke Prediction Dataset https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

### Technology Used
- GitHub
- Pandas
- Plotly
- Jupyter Notebook
- Seaborn
- Matplotlib
- sklearn (train_test_split; StandardScaler, transform, OneHOT Encoder, confusion matrix, LogisticRegression, RandomForestClassifier)
- D3
- Pickle
- JSON
- Flask
- SQlite
- JavaScript
- HTML with CSS page
- Render (hosted our model on Render)

### Machine Learning
We testing our data set using supervised machine learning. First we split our data into our features and target arrays. The split data was then processed into training and testingdataset, utilizing a random_state of 78. StandardScaler instantance as then created, fit, and scaled utilizing the transform fucntion.
```
# Split our preprocessed data into our features and target arrays
   y = stroke_df["stroke"]
   X = stroke_df.drop(["stroke"],1)

# Split the preprocessed data into a training and testing dataset
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

# Create a StandardScaler instances
   scaler = StandardScaler()

# Fit the StandardScaler
   X_scaler = scaler.fit(X_train)

# Scale the data
   X_train_scaled = X_scaler.transform(X_train)
   X_test_scaled = X_scaler.transform(X_test)
   ```
Our data was then inputed into a Logistic Regression model and a Random Forest model and the data was fit tested within each model.

## Analysis
Our logistic regression model was the most accurate with a 94.9% accuracy rating. Our logistic regression model preformed only slightly better (.24%) better than the random forest modeling.

### *Logistic Regression*
```
# Train a Logistic Regression model and print the model score
# Create the model
   from sklearn.linear_model import LogisticRegression
   model = LogisticRegression()

# Fit the model to the training data. 
   model.fit(X_train, y_train)

   print(f"Training Data Score: {model.score(X_train, y_train)}")
   print(f"Testing Data Score: {model.score(X_test, y_test)}")

   print(f'Actual:\t\t{list(y_test[:10])}')
   print(f'Predicted:\t{list(model.predict(X_test[:10]))}')
   ```
Our data trained at a 96% training data score; with a testing data score of 95%

### *Random Forest*
```
# Train a Random Forest Classifier model and print the model score
# Import a Random Forests classifier
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler().fit(X_train)
   X_train_scaled = scaler.transform(X_train)
   X_test_scaled = scaler.transform(X_test)

   clf = RandomForestClassifier(random_state=1).fit(X_train_scaled, y_train)
   y_pred = clf.predict(X_test_scaled)

   print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
   print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')
   ```
Our data trained at 100% with a testing stcore of 94.62%.

### Visualizations
When looking at our data, you can see the coorelation between each individual risk factor (machine learning features) and their stroke risk. From this map you can draw the conclusion that you even though you may have some risk factors; their is not a strong coorelation between any 2 of the factors and having a higher change of a stroke.
![heatmap](https://user-images.githubusercontent.com/113635771/226497396-26fd1ab3-ab61-419b-b0a8-4f281f76d7ea.png)

Scatter plots were also created to show the relationship between BMI with the average glucose level; and the second plot for BMI compared with the patients age. ![bmiGlucose](https://user-images.githubusercontent.com/113635771/226497441-1d6e328e-2811-4203-8381-222708471ae5.png)
![bmiAge](https://user-images.githubusercontent.com/113635771/226497450-ea794584-ffb6-44e8-9781-f321a527d8f0.png)

### Web Page
Our data can be found at https://team-3-project-4.onrender.com/

Our site contains 3 seperate pages, the home screen that shows the above visualizations. A data page that houses a table of the data of the 4,000+ patients that were in our data set. The 3rd page is the predictor page.

![strokePredictor](https://user-images.githubusercontent.com/113635771/226497502-9d158019-09a6-4227-9744-2717841120f0.png)

This predictor page is the go to page that the user will input via drop downs their information. Upon completion of entereing your informaiton and pressing the submit button; the values makes a call out to our API; and check the values against our predictor model; and send back a result in the form of a pop up box letting the user know if they are at risk or having a stroke or not.
![noStroke Risk](https://user-images.githubusercontent.com/113635771/226497524-b7d336d2-a6a8-41ee-b11c-fad20d3cd0ca.png)  ![stokeRisk](https://user-images.githubusercontent.com/113635771/226497533-e5aef78d-04fb-48bc-9fd2-d672a42e4e30.png)



### Future Adjustments
If we had more time, I would suggest that we update the graphs, to match the user inputs for their stroke.

Our data set also included 3 different factors that were not medical; residency type (urban, rural), marital status; and work type (private, government, self-employeed). It would be interesting to dive more into those factors to see if what changes, if any would be made to our machine learning model.

There are many different machine learning models for us to choose from. We choose the models that we understood the best. Future adjustments could be made to look at additional models, to see if there are better outcomes; but our accuracy rating of 95% is fairly accurate.

It would be intersting to test our data against another stroke data set to see if our model predictions were similar to other; and how our model held up with additional data.

### Shout Outs, License and Disclamimer
- This is NOT medical advice. If you are concerned for your well-being please seek medical care, especially if you are experiencing any stroke like symptoms, please call 9-1-1 and seek emergency medical care.
- Stroke definition and signs of a stroke referenced in the powerpoint came from the Centers of Disease (CDC) https://www.cdc.gov/stroke/index.htm
- Brain image for PowerPoint https://www.freepik.com/free-photo/3d-brain-with-lightening_6214247.htm#query=brain%20stroke&position=27&from_view=keyword&track=ais
- BMI has a call out website https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/bmicalc.htm to help determine if a user doesn't readily know that data set.
- © All work done by Project 4-Team 3 for the UMN Data Analytics Bootcamp. 2023 edX Boot Camps LLC. Confidential and Proprietary. All Rights Reserved.
