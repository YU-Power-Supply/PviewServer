import uvicorn
from fastapi import FastAPI, File

import cv2
import numpy as np
import json

from app.pview_core import Oilly, PIH, Pore, SkinTone
from tensorflow.keras.models import load_model
oilmodel = load_model("/home/ubuntu/PviewServer/app/pview_core/weights/oil_model/oilly_model_weight_1203.h5")
pihmodel = load_model("/home/ubuntu/PviewServer/app/pview_core/weights/pih_model/pih_model_weight_230221.h5")

app = FastAPI()

@app.post("/run_ml")
async def register_user(file: bytes = File(...)):
    byte_file = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(byte_file, cv2.IMREAD_COLOR)
    skindict = {
                "pore_detect" : str(Pore.detect_pore(img)),
                "skin_tone" : str(SkinTone.detect_skintone(img)),
                "pih" : str(PIH.detect_pih(img, pihmodel)),
                "oilly" : str(Oilly.oil_detector(img, oilmodel))}

    return json.dumps(skindict)


if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=5000)