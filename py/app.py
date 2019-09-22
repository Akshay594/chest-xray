from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.config import Config
import uvicorn
import os
from io import BytesIO
from fastai.basic_train import load_learner
from fastai import *
from fastai.vision import *
import urllib
import aiohttp

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette(debug=True)

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['*'], allow_methods=['*'])

### EDIT CODE BELOW ###

answer_question_1 = """ 
when the training loss is way higher than the validation loss = "Under-fitting"
When the validation loss is way higher than training loss = "Over-fitting"
"""

answer_question_2 = """ 
Gradient descent measures the local gradient of the error function with regards to the parameter vector θ, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!
"""

answer_question_3 = """ 
Regression analysis helps to predict the continuous values.
"""



model_34 = load_learner('./models', file='Final_model.pkl')


@app.route("/api/answers_to_hw", methods=["GET"])
async def answers_to_hw(request):
    return JSONResponse([answer_question_1, answer_question_2, answer_question_3])

@app.route("/api/class_list", methods=["GET"])
async def class_list(request):
    return JSONResponse(model_34.data.classes)

@app.route("/api/classify", methods=["POST"])
async def classify_url(request):
    body = await request.json()
    bytes = await get_bytes(body["url"])
    img = open_image(BytesIO(bytes))
    _,_,losses = model_34.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(model_34.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

### EDIT CODE ABOVE ###

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ['PORT']))
