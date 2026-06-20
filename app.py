from flask import Flask, render_template, request
import numpy as np
import pickle
import os
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "heart_model.pkl"), "rb") as f:
    model_data = pickle.load(f)
xgb = model_data["xgb"]
rf = model_data["rf"]
threshold = model_data["threshold"]
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.form.to_dict()
        feature_order = [
            "AgeCategory","DiffWalking","Stroke","PhysicalHealth","Diabetic",
            "KidneyDisease","Smoking","PhysicalActivity","SkinCancer","Sex",
            "BMI","Race","AlcoholDrinking","Asthma","MentalHealth"
        ]

        features = []
        age_map = {
            "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3,
            "40-44": 4, "45-49": 5, "50-54": 6, "55-59": 7,
            "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11,
            "80 or older": 12
        }

        yes_no_map = {
            "Yes": 1,
            "No": 0
        }

        sex_map = {
            "Male": 1,
            "Female": 0
        }

        race_map = {
            "White": 0,
            "Black": 1,
            "Asian": 2,
            "American Indian/Alaskan Native": 3,
            "Other": 4
        }

        diabetic_map = {
            "Yes": 1,
            "No": 0,
            "Borderline": 2
        }

        for feature in feature_order:
            value = input_data.get(feature)


            if feature == "AgeCategory":
                value = age_map[value]

            elif feature in [
                "DiffWalking","Stroke","KidneyDisease","Smoking",
                "PhysicalActivity","SkinCancer","AlcoholDrinking","Asthma"
            ]:
                value = yes_no_map[value]


            elif feature == "Sex":
                value = sex_map[value]


            elif feature == "Race":
                value = race_map[value]


            elif feature == "Diabetic":
                value = diabetic_map[value]

            value = float(value)

            features.append(value)

        final_features = np.array([features])

        xgb_prob = xgb.predict_proba(final_features)[:, 1]
        rf_prob  = rf.predict_proba(final_features)[:, 1]

        hybrid_prob = (0.70 * xgb_prob) + (0.30 * rf_prob)

        prediction = (hybrid_prob > threshold).astype(int)[0]
        probability = hybrid_prob[0]

        # Result
        if prediction == 1:
            result = "⚠️ High Risk of Heart Disease"
        else:
            result = "✅ Low Risk of Heart Disease"

        return render_template(
            "result.html",
            prediction_text=result,
            probability=round(probability * 100, 2)
        )

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)