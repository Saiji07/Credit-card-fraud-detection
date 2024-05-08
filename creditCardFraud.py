#no:1
# Create a directory named .kaggle in the user's home directory (~)
!mkdir ~/.kaggle

# Copy the kaggle.json file to the .kaggle directory
!cp kaggle.json ~/.kaggle/

# Set appropriate permissions for the kaggle.json file to ensure security
!chmod 600 ~/.kaggle/kaggle.json

# Download the dataset 'creditcardfraud' from Kaggle using the Kaggle API
!kaggle datasets download -d mlg-ulb/creditcardfraud

# Unzip the downloaded dataset file
!unzip creditcardfraud.zip

#no:2
import pandas as pd
df=pd.read_csv('creditcard.csv')
df

#no:3
df.describe()
# robustly scalling


#4
df['Class'].value_counts()
#5
# Generate histograms for each column in the DataFrame df
# df: DataFrame object containing the data to visualize
df.hist(
    # Number of bins to use for the histograms
    bins=30,
    # Size of the figure (width, height) in inches
    figsize=(30, 30)
)
#6
# Importing the necessary module for RobustScaler
from sklearn.preprocessing import RobustScaler

# Creating a copy of the DataFrame df to work with
new_df = df.copy()

# Robust scaling the 'Amount' column to handle outliers and override the old 'Amount' column
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))

# Generating a histogram to visualize the scaled 'Amount' column, useful for understanding data distribution
new_df['Amount'].hist()

# Scaling the 'Time' column to range between 0 and 1
# This step helps standardize the 'Time' feature, making it more suitable for machine learning algorithms
time = new_df['Time']
new_df['Time'] = (time - time.min()) / (time.max() - time.min())

# Printing the modified DataFrame
new_df
#7
# Shuffle the rows of the DataFrame 'new_df' using .sample() method
# frac=1 means shuffling all rows, random_state=1 ensures reproducibility of shuffling
new_df = new_df.sample(frac=1, random_state=1)

# Display the shuffled DataFrame 'new_df'
new_df
#8
new_df['Amount'].describe()
#9
train,test,val=new_df[:240000],new_df[240000:262000],new_df[262000:]
train['Class'].value_counts(),test['Class'].value_counts(), val['Class'].value_counts
#10
# Convert the DataFrame 'train' to a NumPy array and assign it to 'train_np'
# This converts the data in 'train' DataFrame to a format suitable for numerical computations
train_np,test_np,val_np= train.to_numpy(),test.to_numpy(),val.to_numpy()
train_np.shape,test_np.shape,val_np.shape

#11
# Extract features (independent variables) from the training data
# For 'x_train', select all rows and all columns except the last column
x_train,y_train = train_np[:, :-1],train_np[:, -1]

# Extract the target variable (dependent variable) from the training data
# For 'y_train', select all rows and only the last column

# Extract features from the testing data
# Similar to 'x_train', select all rows and all columns except the last column
x_test,y_test = test_np[:, :-1],test_np[:, -1]

# Extract the target variable from the testing data
# Similar to 'y_train', select all rows and only the last column

# Extract features from the validation data
# Similar to 'x_train', select all rows and all columns except the last column
x_val, y_val = val_np[:, :-1], val_np[:, -1]

# Extract the target variable from the validation data
# Similar to 'y_train', select all rows and only the last column

# Print the shapes of the extracted data arrays
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)

#12
from sklearn.linear_model import LogisticRegression
# Instantiate a Logistic Regression model
logistic_model = LogisticRegression()

# Train the Logistic Regression model using the training data
logistic_model.fit(x_train, y_train)

# Evaluate the accuracy of the Logistic Regression model on the training data
logistic_model.score(x_train, y_train)
#13
#fraud as positive class and not fraud as negative class
from sklearn.metrics import classification_report
# Generate a classification report for the validation dataset
# 'y_val' contains the actual target labels from the validation dataset
# 'logistic_model.predict(x_val)' predicts the labels for the validation features 'x_val'
# 'target_names' parameter specifies the names of the classes for better readability in the report
print(classification_report(y_val, logistic_model.predict(x_val), target_names=['Not Fraud', 'Fraud']))
#14
# Importing necessary modules from TensorFlow Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# Creating a Sequential model
shallow_nn = Sequential()

# Adding an input layer to the model with the shape matching the number of features in the input data
shallow_nn.add(InputLayer((x_train.shape[1],)))

# Adding a Dense (fully connected) layer with 2 units and 'relu' activation function
shallow_nn.add(Dense(2, activation='relu'))

# Adding Batch Normalization layer to normalize the activations of the previous layer
shallow_nn.add(BatchNormalization())

# Adding a Dense output layer with 1 unit and 'sigmoid' activation function
# 'sigmoid' activation is commonly used for binary classification where the output represents the probability of the positive class
shallow_nn.add(Dense(1, activation='sigmoid')) #If less than 0.5 than not fraud,and >0.5 than fraud

# Defining a ModelCheckpoint callback to save the best model during training
# 'save_best_only=True' ensures that only the best model is saved based on the validation loss
checkpoint = ModelCheckpoint('shallow_nn', save_best_only=True)

# Compiling the model
# 'adam' optimizer is commonly used for training neural networks
# 'binary_crossentropy' loss function is suitable for binary classification problems
# 'accuracy' metric is used to evaluate the performance of the model during training
shallow_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#15
shallow_nn.summary()
#give summary
#16
shallow_nn.fit(
    x_train,          # Features of the training data
    y_train,          # Target labels of the training data
    validation_data=(x_val, y_val),  # Validation data (features and target labels)
    epochs=5,         # Number of epochs (iterations over the entire training dataset)
    callbacks=checkpoint  # List of callbacks (in this case, ModelCheckpoint for saving the best model)
)
#17
def neural_net_predictions(model, x):
    # Make predictions using the neural network model on the input data
    predictions = model.predict(x)

    # Flatten the predictions array and compare each element with 0.5 to classify as 0 or 1
    # If prediction > 0.5, classify as 1 (Fraud), otherwise classify as 0 (Not Fraud)
    binary_predictions = (predictions.flatten() > 0.5).astype(int)

    return binary_predictions

# Call the neural_net_predictions function with the shallow neural network model (shallow_nn) and validation data (x_val)
predictions = neural_net_predictions(shallow_nn, x_val)

# Print the predictions
print(predictions)
#18
# Print a classification report for the validation data
# 'y_val' contains the actual target labels from the validation dataset
# 'neural_net_predictions(shallow_nn, x_val)' predicts the labels for the validation features 'x_val'
# 'target_names' parameter specifies the names of the classes for better readability in the report
print(classification_report(y_val, neural_net_predictions(shallow_nn, x_val), target_names=['Not Fraud', 'Fraud']))
#19
# Instantiate a RandomForestClassifier model with specified parameters
# 'max_depth=2' sets the maximum depth of the trees in the forest to 2
# 'n_jobs=-1' allows the random forest to use all available CPU cores for parallel processing
rf = RandomForestClassifier(max_depth=2, n_jobs=-1)

# Train the RandomForestClassifier model on the training data
rf.fit(x_train, y_train)

# Print a classification report for the validation data
# 'y_val' contains the actual target labels from the validation dataset
# 'rf.predict(x_val)' predicts the labels for the validation features 'x_val'
# 'target_names' parameter specifies the names of the classes for better readability in the report
report = classification_report(y_val, rf.predict(x_val), target_names=['Not Fraud', 'Fraud'])

# Print the classification report
print(report)
#20

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Instantiate a GradientBoostingClassifier model with specified parameters
# 'n_estimators=50' sets the number of boosting stages (trees) to 50
# 'learning_rate=1.0' controls the contribution of each tree in the sequence
# 'max_depth=1' sets the maximum depth of the individual trees to 1
# 'random_state=0' ensures reproducibility of results
gbc = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=1, random_state=0)

# Train the GradientBoostingClassifier model on the training data
gbc.fit(x_train, y_train)

# Print a classification report for the validation data
# 'y_val' contains the actual target labels from the validation dataset
# 'gbc.predict(x_val)' predicts the labels for the validation features 'x_val'
# 'target_names' parameter specifies the names of the classes for better readability in the report
report = classification_report(y_val, gbc.predict(x_val), target_names=['Not Fraud', 'Fraud'])

# Print the classification report
print(report)
#21
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Instantiate a LinearSVC model with class_weight set to 'balanced'
# 'class_weight='balanced'' automatically adjusts weights inversely proportional to class frequencies
svc = LinearSVC(class_weight='balanced')

# Train the LinearSVC model on the training data
svc.fit(x_train, y_train)

# Print a classification report for the validation data
# 'y_val' contains the actual target labels from the validation dataset
# 'svc.predict(x_val)' predicts the labels for the validation features 'x_val'
# 'target_names' parameter specifies the names of the classes for better readability in the report
report = classification_report(y_val, svc.predict(x_val), target_names=['Not Fraud', 'Fraud'])

# Print the classification report
print(report)
#22
# 22,23,24,25  are used to make dataset balanced so can try on it
train.head()
#23
# Querying the DataFrame to filter instances labeled as "Not Fraud" (Class == 0)
not_frauds = new_df.query('Class == 0')

# Querying the DataFrame to filter instances labeled as "Fraud" (Class == 1)
frauds = new_df.query('Class == 1')

# Counting the number of instances labeled as "Not Fraud" and "Fraud" respectively
# The value_counts() method is used to count the occurrences of each unique value in the 'Class' column
# It returns a Series where the index represents the unique values and the values represent their counts
not_fraud_counts = not_frauds['Class'].value_counts()
fraud_counts = frauds['Class'].value_counts()

# Printing the counts of "Not Fraud" and "Fraud" instances
print("Not Fraud Counts:", not_fraud_counts)
print("Fraud Counts:", fraud_counts)
#24
# Concatenating an equal number of instances labeled as "Fraud" and "Not Fraud"
# The 'frauds' DataFrame contains instances labeled as "Fraud"
# The 'not_frauds.sample(len(frauds), random_state=1)' part randomly samples the same number of instances as 'frauds'
# The 'random_state=1' parameter ensures reproducibility of sampling
balanced_df = pd.concat([frauds, not_frauds.sample(len(frauds), random_state=1)])

# Counting the number of instances labeled as "Fraud" and "Not Fraud" in the balanced DataFrame
# The value_counts() method is used to count the occurrences of each unique value in the 'Class' column
# It returns a Series where the index represents the unique values and the values represent their counts
balanced_class_counts = balanced_df['Class'].value_counts()

# Printing the counts of "Fraud" and "Not Fraud" instances in the balanced DataFrame
print("Balanced Class Counts:", balanced_class_counts)
#25
balanced_df = balanced_df.sample(frac=1, random_state=1)
balanced_df
#26
# Convert the balanced DataFrame to a NumPy array
balanced_df_np = balanced_df.to_numpy()

# Split the NumPy array into training, testing, and validation sets
# Extract features and target labels for each set
# The slicing indices are used to partition the data
# The 'astype(int)' method is used to convert the target labels to integers
x_train_b, y_train_b = balanced_df_np[:700, :-1], balanced_df_np[:700, -1].astype(int)
x_test_b, y_test_b = balanced_df_np[700:842, :-1], balanced_df_np[700:842, -1].astype(int)
x_val_b, y_val_b = balanced_df_np[842:, :-1], balanced_df_np[842:, -1].astype(int)

# Print the shapes of the extracted arrays for verification
print("x_train_b shape:", x_train_b.shape)
print("y_train_b shape:", y_train_b.shape)
print("x_test_b shape:", x_test_b.shape)
print("y_test_b shape:", y_test_b.shape)
print("x_val_b shape:", x_val_b.shape)
print("y_val_b shape:", y_val_b.shape)
#27
# Create pandas Series objects from the target labels of the training, testing, and validation sets
# These Series objects will allow us to easily count the occurrences of each unique label
y_train_b_series = pd.Series(y_train_b)
y_test_b_series = pd.Series(y_test_b)
y_val_b_series = pd.Series(y_val_b)

# Count the occurrences of each unique label in the training, testing, and validation sets
# The value_counts() method is used to count the occurrences of each unique value in the Series
train_class_counts = y_train_b_series.value_counts()
test_class_counts = y_test_b_series.value_counts()
val_class_counts = y_val_b_series.value_counts()

# Print the counts of each unique label in the training, testing, and validation sets
print("Training Set Class Counts:\n", train_class_counts)
print("\nTesting Set Class Counts:\n", test_class_counts)
print("\nValidation Set Class Counts:\n", val_class_counts)
#28
#applying logistic model on balanced data set
logistic_model_b = LogisticRegression()
logistic_model_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, logistic_model_b.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
#29
#applying shallow_nn model on balanced data set in which relu is 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# Create a Sequential model for the balanced dataset
shallow_nn_b = Sequential()

# Add an input layer to the model with the shape matching the number of features in the balanced training data
shallow_nn_b.add(InputLayer((x_train.shape[1],)))

# Add a Dense (fully connected) layer with 2 units and 'relu' activation function
shallow_nn_b.add(Dense(2, activation='relu'))

# Add Batch Normalization layer to normalize the activations of the previous layer
shallow_nn_b.add(BatchNormalization())

# Add a Dense output layer with 1 unit and 'sigmoid' activation function
# 'sigmoid' activation is commonly used for binary classification where the output represents the probability of the positive class
shallow_nn_b.add(Dense(1, activation='sigmoid'))

# Define a ModelCheckpoint callback to save the best model during training
# 'save_best_only=True' ensures that only the best model based on the validation loss is saved
checkpoint = ModelCheckpoint('shallow_nn_b', save_best_only=True)

# Compile the model
# 'adam' optimizer is commonly used for training neural networks
# 'binary_crossentropy' loss function is suitable for binary classification problems
# 'accuracy' metric is used to evaluate the performance of the model during training
shallow_nn_b.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model to the balanced training data, validating it using the balanced validation data
# 'epochs=40' specifies the number of epochs (iterations over the entire training dataset)
# 'callbacks=checkpoint' saves the best model during training
shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_val_b, y_val_b), epochs=40, callbacks=[checkpoint])
#30
shallow_nn_b.fit(x_train_b, y_train_b, validation_data=(x_val_b, y_val_b), epochs=40, callbacks=checkpoint)
#31
print(classification_report(y_val_b, neural_net_predictions(shallow_nn_b1, x_val_b), target_names=['Not Fraud', 'Fraud']))
#32
##applying RandomForestClassifier on balanced data set
rf_b = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, rf.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
#33
##applying GradientBoostingClassifier on balanced data set
gbc_b = GradientBoostingClassifier(n_estimators=50, learning_rate=1.0, max_depth=2, random_state=0)
gbc_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, gbc.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
#34
##applying LinearSVC on balanced data set
svc_b = LinearSVC(class_weight='balanced')
svc_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, svc.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
#35
svc_b = LinearSVC(class_weight='balanced')
svc_b.fit(x_train_b, y_train_b)
print(classification_report(y_val_b, svc.predict(x_val_b), target_names=['Not Fraud', 'Fraud']))
#36
print(classification_report(y_test_b, neural_net_predictions(shallow_nn_b, x_test_b), target_names=['Not Fraud', 'Fraud']))




