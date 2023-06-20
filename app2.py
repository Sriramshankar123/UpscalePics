from flask import Flask, request, jsonify, render_template
import pickle
#from sklearn.ensemble import StackingRegressor as reg
#import xgboost as xgb
import logging
import numpy as np
import pandas as pd

df = pd.read_csv('processed_data.csv')
X = df.drop('price' , axis = 'columns')
print(X.head())
print(X.columns)

# load the stacked regressor model
with open('lr_clf.pkl', 'rb') as f:
    model = pickle.load(f)

# create a Flask app
app = Flask(__name__)

# define the home page
@app.route('/')
def home():
    return render_template('index2.html')

def format_number(number):
    number_str = str(int(number))
    num_digits = len(number_str)

    result = ""
    for i in range(num_digits):
        result = number_str[num_digits - 1 - i] + result
        if i % 2 == 1 and i != num_digits - 1:
            result = "," + result

    return result

# define the prediction function
@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    bath = float(data['bath'])
 
    bhk = int(data['bhk'])
    total_sqft = float(data['total_sqft'])
    area_type = float(data['area_type'])
    location= str(data['location'])
    #location='Thanisandra'
    print(data)
    # bath = 3.0
    # bhk = 3
    # total_sqft = 2500.0
    # area_type = 1.0
    
    logging.debug(f"bath={bath}, bhk={bhk}, total_sqft={total_sqft}, area_type={area_type},location={location}")
    
    # perform any necessary data processing  on the input fields
    # ...def predict_price(location ,area_type, sqft , bath , bhk):
    # cols=['area_type', 'total_sqft', 'bath', 'bhk', 'Electronic City', 'Hebbal',
    #    'Kanakpura Road', 'Marathahalli', 'Raja Rajeshwari Nagar',
    #    'Sarjapur  Road', 'Thanisandra', 'Uttarahalli', 'Whitefield',
    #    'Yelahanka', 'other']
    
    #loc_index = np.where(cols == location)[0][0]
    if (location in X.columns):
        loc_index = np.where(X.columns == location)[0] 
    else :
        loc_index=(len(X.columns)-1)
   
    x = np.zeros(len(X.columns))
    x[0] = area_type
    x[1] = total_sqft
    x[2] = bath
    x[3] = bhk
    if loc_index>= 0:
        x[loc_index] = 1
    
    
    #xgb_input = xgb.DMatrix(x.reshape(1, -1))
    x = np.array(x)

    pred = model.predict(x.reshape(1, -1))

    # create a feature vector from the input fields
    features = [area_type, total_sqft, bath, bhk]

    # make the prediction using the stacked regressor model
    # pred = model.predict([features])
    pred=pred*100000
    th=str(int(pred%1000))
    
    if len(th)==2:
        th='0'+th
    if len(th)==1:
        th='00'+th
    above_th=pred/1000
    formatted_price = format_number(above_th)

# Add the currency symbol and display the formatted price
    result = {'predicted_price': f"{formatted_price+','+th} INR"}
    if(pred<10000 or total_sqft/bhk<250 or bath>2*bhk):
        result={'predicted_price':"Error!! Parameters out of bound"}
    #print(result['predicted_price'])
    # format the prediction as a JSON response
    #result = {'predicted_price': int(float(pred)*100000)}
    #result = {'predicted_price': "{:,} INR".format(int(float(pred)*100000))}

    return jsonify(result)

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)