import os
import json
import pickle
from sklearn.externals import joblib
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, PostgresqlDatabase, Model, IntegerField,
    FloatField, TextField, IntegrityError
)
from playhouse.shortcuts import model_to_dict


########################################
# Begin database stuff

if 'DATABASE_URL' in os.environ:
    db_url = os.environ['DATABASE_URL']
    dbname = db_url.split('@')[1].split('/')[1]
    user = db_url.split('@')[0].split(':')[1].lstrip('//')
    password = db_url.split('@')[0].split(':')[2]
    host = db_url.split('@')[1].split('/')[0].split(':')[0]
    port = db_url.split('@')[1].split('/')[0].split(':')[1]
    DB = PostgresqlDatabase(
        dbname,
        user=user,
        password=password,
        host=host,
        port=port,
    )
else:
    DB = SqliteDatabase('predictions.db')


class Prediction(Model):
    observation_id = IntegerField(unique=True)
    observation = TextField()
    proba = TextField()
    # is_contraband = TextField()
    true_class = IntegerField(null=True)

    class Meta:
        database = DB


DB.create_tables([Prediction], safe=True)

# End database stuff
########################################

########################################
# Unpickle the previously-trained model


with open('columns.json') as fh:
    columns = json.load(fh)

pipeline = joblib.load('pipeline.pickle')

with open('dtypes.pickle', 'rb') as fh:
    dtypes = pickle.load(fh)


# End model un-pickling
########################################


########################################
# Begin webserver stuff

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # flask provides a deserialization convenience function called
    # get_json that will work if the mimetype is application/json
    obs_dict = request.get_json()
    _id = obs_dict['id']
    observation = obs_dict['observation']
    # now do what we already learned in the notebooks about how to transform
    # a single observation into a dataframe that will work with a pipeline
    obs = pd.DataFrame([observation], columns=columns).astype(dtypes)
    # now get ourselves an actual prediction of the positive class
    proba = pipeline.predict(obs)
    proba=str(proba[0])
    response = {'proba': proba}
    p = Prediction(
        observation_id=_id,
        proba=proba,
        observation=request.data
    )

    try:
        p.save()
    except IntegrityError:
        error_msg = 'Observation ID: "{}" already exists'.format(_id)
        response['error'] = error_msg
        print(error_msg)
        DB.rollback()
    return jsonify(response)


@app.route('/update', methods=['POST'])
def update():
    obs = request.get_json()
    try:
        p = Prediction.get(Prediction.observation_id == obs['id'])
        p.true_class = obs['true_class']
        p.save()
        return jsonify(model_to_dict(p))
    except Prediction.DoesNotExist:
        error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
        return jsonify({'error': error_msg})


@app.route('/list-db-contents')
def list_db_contents():
    return jsonify([
        model_to_dict(obs) for obs in Prediction.select()
    ])

# cols=['Department Name',
#  'InterventionLocationName',
#  'InterventionReasonCode',
#  'ResidentIndicator',
#  'SearchAuthorizationCode',
#  'StatuteReason',
#  'SubjectAge',
#  'SubjectEthnicityCode',
#  'SubjectRaceCode',
#  'SubjectSexCode',
#  'TownResidentIndicator']
# new_obs_str = '{ "ResidentIndicator": true,"Department Name": "new Haven", "SearchAuthorizationCode": "C", "StatuteReason": "Speed Related", "SubjectRaceCode": "B","InterventionReasonCode": "V",   "SubjectSexCode": "F", "SubjectEthnicityCode": "M", "SubjectAge": 26, "InterventionLocationName": "New Haven","TownResidentIndicator":true }'
# new_obs_dict = json.loads(new_obs_str)
# print('type {}'.format(type(new_obs_dict)))
# obs = pd.DataFrame([new_obs_dict], columns=cols)

# # Now you need to make sure that the types are correct so that the
# # pipeline steps will have things as expected.
# # obs = obs.astype(X_train.dtypes)
# ppp= pipeline.predict_proba(obs)
# with open('ppp.pickle', 'wb') as fh:
#     pickle.dump(ppp, fh)

# ppp = pickle.load(open('/Users/valentynakoshelnyk/Documents/GitHub/batch3-workspace/heroku/ppp.pickle', 'rb'))

# End webserver stuff
########################################



if __name__ == "__main__":
    app.run(debug=True, port=5000)

# #############################################################################
#     # DROPPING COLUMNS
# #     cols_drop = ['InterventionDateTime',
# #  'ReportingOfficerIdentificationID']

# #     obs = obs.drop(cols_drop, axis=1)
        
#     # now get ourselves an actual prediction of the positive class
#     # proba = pipeline.predict_proba(obs)[0, 1]
#     #obs_pred = pipe.predict_proba(obs.drop(cols_drop, axis=1))
#     obs_pred = pipeline.predict_proba(obs)
#     threshold = 0.5

#     def convert_is_contraband(obs_pred):
#         if obs_pred[0][1] >= threshold:
#             return "True"
#         else:
#             return "False"

#     # response = {'proba': proba}

#     is_contraband = convert_is_contraband(obs_pred)

#     response = {'ContrabandIndicator': is_contraband}

#     p = Prediction(
#         observation_id=_id,
#         # proba=proba,
#         is_contraband=is_contraband,
#         observation=request.data
#     )
#     try:
#         p.save()
#     except IntegrityError:
#         error_msg = 'Observation ID: "{}" already exists'.format(_id)
#         response['error'] = error_msg
#         print(error_msg)
#         DB.rollback()
#     return jsonify(response)


# @app.route('/update', methods=['POST'])
# def update():
#     obs = request.get_json()
#     try:
#         p = Prediction.get(Prediction.observation_id == obs['id'])
#         p.true_class = obs['true_class']
#         p.save()
#         return jsonify(model_to_dict(p))
#     except Prediction.DoesNotExist:
#         error_msg = 'Observation ID: "{}" does not exist'.format(obs['id'])
#         return jsonify({'error': error_msg})



# @app.route('/list-db-contents')
# def list_db_contents():
#     return jsonify([
#         model_to_dict(obs) for obs in Prediction.select()
#     ])


# # End webserver stuff
# ########################################

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

