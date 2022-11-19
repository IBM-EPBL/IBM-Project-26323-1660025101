

import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

app = Flask(__name__, template_folder='template')
filename: str = 'resale_model.sav'
model_rand = pickle.load(open(filename, 'rb'))


@app.route('/')
def index():
    return render_template('resaleintro.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    return render_template('resalepredict.html')


@app.route('/y_predict', methods=['GET', 'POST'])
def y_predict():
    regyear = int(request.form['regyear'])
    powerps = float(request.form['powerps'])
    kms = float(request.form['kms'])
    regmonth = request.form.get('regmonth')
    gearbox = request.form['gearbox']
    damage = request.form['dam']
    model = request.form.get('modeltype')
    brand = request.form.get('brand')
    fuelType = request.form.get('fuel')
    vehicletype = request.form.get('vehicletype')
    new_row = {'yearOfRegistration': regyear, 'powerPS': powerps, 'kilometer': kms,
               'monthOfRegistration': regmonth, 'gearbox': gearbox, 'notRepairedDamage': damage,
               'model': model, 'brand': brand, 'fuelType': fuelType,
               'vehicleType': vehicletype}
    print(new_row)
    new_df = pd.DataFrame(columns=['vehicleType', 'yearOfRegistration', 'gearbox',
                                   'powerPS', 'model', 'kilometer', 'monthOfRegistration', 'fuelType',
                                   'brand', 'notRepairedDamage'])
    new_df = new_df.append(new_row, ignore_index=True)
    labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']
    mapper = {}
    np_load_old = np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    for i in labels:
        mapper[i] = LabelEncoder()
        mapper[i].classes_ = np.load(str('classes' + i + '.npy'))
        tr = mapper[i].fit_transform(new_df[i])
        new_df.loc[:, i] = pd.Series(tr, index=new_df.index)
    labeled = new_df[['yearOfRegistration', 'powerPS', 'kilometer', 'monthOfRegistration']
                     + [x for x in labels]]
    X = labeled.values
    print(X)
    y_prediction = model_rand.predict(X)
    print(y_prediction)
    return render_template('resalepredict.html', output=y_prediction[0])


if __name__ == '_main_':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=False)
