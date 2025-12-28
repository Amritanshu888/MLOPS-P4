from fastapi import FastAPI  ## FastAPI is specifically used to create API's, it basically works like Flask but has much more good functionalities in it
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from src.textSummarizer.pipeline.prediction_pipeline import PredictionPipeline

text:str = "What is Text Summarization?"

app = FastAPI()

@app.get("/",tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
async def training():
    try:
        os.system("python main.py")  ## As this gets executed my training entirely starts(Better execute when u have a GPU)
        return Response("Training successfull !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
@app.post("/predict")
async def predict_route(text):
    try:
        obj = PredictionPipeline()
        text = obj.predict(text) ## Whatever text we are giving
        return text ## Then it will just return the text
    except Exception as e:
        raise e

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)        

## To execute : In terminal write the command : python app.py
## After running we can execute the API's from the swagger ui(using try it out)
## In the API's while executing u can provide a small story in the text to get the summary.