from flask import Flask, flash, request, redirect, url_for, render_template
import pandas as pd
import joblib


# Configuring Flask
app = Flask(__name__)
@app.route('/')
def home():
    return render_template('flight.html')


########################### Result Functions ########################################


@app.route('/result', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        airline = request.form['airline']
        flight = request.form['flight']
        source_city = request.form['source_city']
        departure_time = request.form['departure_time']
        stops = request.form['stops']
        arrival_time = request.form['arrival_time']
        destination_city = request.form['destination_city']
        Class = request.form['class']
        duration = request.form['duration']
        days_left = request.form['days_left']
        model = joblib.load('flight_price_prediction_rf_model.joblib')

        # Load label encoders
        label_encoders = {}
        for column in ['airline', 'flight', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']:
            le = joblib.load(f'{column}_label_encoder.joblib')
            label_encoders[column] = le

        # Load the scaler
        scaler = joblib.load('standard_scaler.pkl')

        # Define a single input row
        single_input_values = {
            'airline': airline,
            'flight': flight,
            'source_city': source_city,
            'departure_time': departure_time,
            'stops': stops,
            'arrival_time': arrival_time,
            'destination_city': destination_city,
            'class': Class,
            'duration': duration,
            'days_left': days_left
        }

        # Create a DataFrame from the single input
        single_input_df = pd.DataFrame([single_input_values])

        # Preprocess categorical features for prediction
        for column, le in label_encoders.items():
            single_input_df[column] = le.transform(single_input_df[column])

        # Standardize numerical features
        single_input_scaled = scaler.transform(single_input_df)

        # Make predictions
        predicted_price = model.predict(single_input_scaled)
        return render_template('result.html', r=int(predicted_price[0]), airline=airline, flight=flight, source_city=source_city, departure_time=departure_time, stops=stops, arrival_time=arrival_time, destination_city=destination_city, Class=Class, duration=duration, days_left=days_left)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
