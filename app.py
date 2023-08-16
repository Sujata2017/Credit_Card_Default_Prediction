from flask import Flask,render_template,request,jsonify
import numpy as np
import pandas as pd
from src.logger import logging
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

## Route for a home page

#@app.route('/')
#def index():
#    return render_template('index.html') 


@app.route('/predictdata',methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    else:
        data = CustomData(
            #LIMIT_BAL = int(request.form.get("LIMIT_BAL")),
            SEX = int(request.form.get("SEX")),
            #SEX = request.form.get("SEX"),
            EDUCATION = int(request.form.get("EDUCATION")),
            #EDUCATION = request.form.get("EDUCATION"),
            #MARRIAGE = int(request.form.get("MARRIAGE")),
            #MARRIAGE = request.form.get("MARRIAGE"),
            AGE = int(request.form.get("AGE")),
            PAY_0 = int(request.form.get("PAY_0")),
            PAY_2 = int(request.form.get("PAY_2")),
            PAY_3 = int(request.form.get("PAY_3")),
            PAY_4 = int(request.form.get("PAY_4")),
            PAY_5 = int(request.form.get("PAY_5")),
            PAY_6 = int(request.form.get("PAY_6"))
            #BILL_AMT1 = int(request.form.get("BILL_AMT1")),
            #BILL_AMT2 = int(request.form.get("BILL_AMT2")),
            #BILL_AMT3 = int(request.form.get("BILL_AMT3")),
            #BILL_AMT4 = int(request.form.get("BILL_AMT4")),
            #BILL_AMT5 = int(request.form.get("BILL_AMT5")),
            #BILL_AMT6 = int(request.form.get("BILL_AMT6")), 
            #PAY_AMT1 = int(request.form.get("PAY_AMT1")),
            #PAY_AMT2 = int(request.form.get("PAY_AMT2")),
            #PAY_AMT3 = int(request.form.get("PAY_AMT3")),
            #PAY_AMT4 = int(request.form.get("PAY_AMT4")),
            #PAY_AMT5 = int(request.form.get("PAY_AMT5")),
            #PAY_AMT6 = int(request.form.get("PAY_AMT6"))
            )
        

        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results=predict_pipeline.predict(pred_df)

        #return render_template("home.html",results=results[0])
        #return render_template("home.html",results=results)
        #logging.info("printing the result")
        #print(results)

        pred = results
        

        if pred == 1:
        #if pred > 0:    
            return render_template("home.html",final_result = "0 - Non-Defaulter, 1 - Defaulter. The Credit Card Holder Will Be Defaulter in Next Month:{}".format(pred))
        elif pred == 0:
        #else:    
            return render_template("home.html",final_result = "0 - Non-Defaulter, 1 - Defaulter. The Credit Card Holder Will Not Be Defaulter in Next Month:{}".format(pred))


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)    


