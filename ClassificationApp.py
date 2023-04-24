import streamlit as st
from PIL import Image
import requests
import os , io
EndPoint = "http://0.0.0.0:7000/classify"

def LoadImagFile():
    #docID = uuid.uuid4()
    docID = "pan"
    file_name = "./{}_temp.jpg".format(str(docID))
    upload_file = st.file_uploader(label="Please upload an Image file ")
    if upload_file is not None:
        image_data = upload_file.getvalue()
        st.image(image_data)
        ImageData = Image.open(io.BytesIO(image_data))
        ImageData.save(file_name)
        return file_name
    else:
        return None

def RunInference(ImagePath):
    print("image path" , ImagePath)
    file = {'file':open(ImagePath,'rb')}
    data = {"token":"en"}

    getData = requests.post(EndPoint,files=file,data=data,verify=False,timeout=300)

    jsonData = getData.json()
    print(jsonData)
    os.remove(ImagePath)
    return jsonData

def main():
    st.title("Classification Application")
    filename = LoadImagFile()
    result = st.button('Run on image')
    if result:
        st.write("Processing Image")
        output = RunInference(filename)
        
        st.write("Time : "+output["Time Stamp"])
        st.write("Document Type : "+output["Document"])
        st.write("Processing Time : "+output["Processing Time"])


if __name__=="__main__":
    main()