from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the dataset
dataset = pd.read_csv("jobPlacement.csv")
dataset.sample(10)

# Encode categorical columns
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status' ] 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dataset[categorical_cols] = dataset[categorical_cols].apply(lambda col: le.fit_transform(col)) 
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()

array_hot_encoded = ohe.fit_transform(dataset[categorical_cols])

data_hot_encoded = pd.DataFrame(array_hot_encoded, index=dataset.index)

data_other_cols = dataset.drop(columns=categorical_cols)

data_out = pd.concat([data_hot_encoded, data_other_cols], axis=1)

dataset.sample(10)

# Split dataset into input (x) and output (y)
x = dataset.iloc[:, 1:13].values
y = dataset.iloc[:, -2].values

# Train-test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)

# Perform Linear Regression
from imblearn.combine import SMOTETomek
smk = SMOTETomek(random_state = 42)
x,y = smk.fit_resample(x,y)

# Perform Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict the output
y_pred = regressor.predict(x_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



app = Flask(__name__, template_folder='templates')


@app.route('/api/results', methods=['GET'])
def get_results():
    # Perform predictions and calculate MSE, R-squared
    # Return the results as JSON
    return jsonify({
        'mse': mse,
        'r2': r2
    })

@app.route('/api/plot', methods=['GET'])
def get_plot():
    # Generate the plot and save it as a temporary file
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted values - Linear Regression")
    plot_filename = 'temp_plot.png'
    plt.savefig(plot_filename)

    # Render the plot in HTML template
    return render_template('plot.html', plot_filename=plot_filename)

if __name__ == '__main__':
    app.run(debug=True)
