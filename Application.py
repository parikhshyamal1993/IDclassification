from fastapi import FastAPI, File, Form, UploadFile, HTTPException 
#from pydantic import BaseModels , HttpUrl
from fastapi.responses import JSONResponse
from loguru import logger
import platform
from source import utils , inference
import uuid
import time
import datetime
import uvicorn , os
 
idxTolable = {0 :"pan" , 1:"addhar",2:"passport"}

logger.add("./logs/file.log",format="{time:YYYY-MM-DD at HH:mm:ss} | {level} |{message}")

uname= platform.uname()


app = FastAPI()



@app.get("/health",status_code=200,description="Status report of api")

def health():
    return {
        "server":uname.node,
        "service":"ID Card Classification",
        "status": "alive"
    }
@app.post("/classify",status_code=201)

def classification(
    file:UploadFile = File(...),
    token:str=Form()
    ):
    timeStamp = datetime.datetime.now()
    docID = uuid.uuid4()
    print("uniques",docID)
    if utils.FileHandler(file.filename)==True:
        try:
            print("in processing")
            contents = file.file.read()
            file_name = str(docID)+"_"+file.filename
            with open(file_name , "wb") as files:
                files.write(contents)
            startTime = time.time()
            input_tensor = inference.transform_image(file_name)
            prediction_idx = inference.get_prediction(input_tensor)
            label = idxTolable[prediction_idx]
            print(label)
            endTime = time.time()
            total_time = endTime-startTime
            os.remove(file_name)
            return {"Time Stamp" : "{}".format(timeStamp),
                    "Document":"{}".format(label),
                    "Processing Time":"{}".format(total_time),
                    }
        except Exception as e :
            logger.error("Exception handler",e)
            os.remove(file_name)
            raise HTTPException(status_code=401,
                                details = "{}".format(e),)
        
if __name__=="__main__":
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=7000,
        workers=1
    )
    
