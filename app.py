from flask import Flask, request, jsonify
import joblib 
import numpy as np

app=Flask(__name__)
#load model
model=joblib.load('model_pipeline.pkl')
@app.route('/')
def home():
    return "ML API is running!"

@app.route('/test')
def test():
    import numpy as np
    result = model.predict(np.array([[5,6]]))
    return str(result)

@app.route('/predict',methods=['POST'])
def predict():
    try:
        
        data=request.get_json()
        hours=data['hours_studied']
        sleep=data['sleep_hours']
        features=np.array([[hours, sleep]])
        prediction=model.predict(features)
        return jsonify({
            "prediction":int(prediction[0]),
            "message":"Pass" if prediction[0]==1 else "Fail"
            
        })
    except Exception as e:
        return jsonify({"error":str(e)})
    
if __name__ == '__main__':
    app.run()
    
