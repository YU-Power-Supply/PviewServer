import uvicorn
from fastapi import FastAPI, File

import cv2
import numpy as np
import json

from app.pview_core import Oilly, PIH, Pore, SkinTone
import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def load_model(self):
        if self.model is None:
            self.model = tf.keras.models.load_model(self.model_path)

        return self.model

import gc
from apscheduler.schedulers.background import BackgroundScheduler
def cleanup_memory():
    gc.collect()

# Create scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(cleanup_memory, 'interval', seconds=5)
scheduler.start()

oilmodel = ModelLoader("/home/ubuntu/PviewServer/app/pview_core/weights/oil_model/oilly_model_weight_1203.h5").load_model()
pihmodel = ModelLoader("/home/ubuntu/PviewServer/app/pview_core/weights/pih_model/pih_model_weight_230221.h5").load_model()

app = FastAPI()

@app.post("/run_ml")
async def global_detect(file: bytes = File(...)):
    byte_file = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(byte_file, cv2.IMREAD_COLOR)
    skindict = {
                "pore_detect" : str(Pore.detect_pore(img)),
                "skin_tone" : str(SkinTone.detect_skintone(img)),
                "pih" : str(PIH.detect_pih(img, pihmodel)),
                "oilly" : str(Oilly.oil_detector(img, oilmodel))}

    del byte_file
    del img

    return json.dumps(skindict)


if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=5000)