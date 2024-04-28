from flask import Flask, render_template, request
from flask_cors import CORS
from model import extract_text_from_pdf,display_res
import pandas as pd

df_str = ''
df = pd.DataFrame()
text=''
app = Flask(__name__)
CORS(app)
    
@app.route("/",methods=["GET","POST"])
def index():
    
    return render_template("index.html")

@app.route("/upload_csv", methods=["POST"])
def upload_and_read_csv():

    global df_str
    global df
    if "pdfFile" not in request.files:
        return "No PDF file uploaded", 400
    # print(type(request))
    # print(request.files["pdfFile"])
    csv_file = request.files["pdfFile"]
    df = pd.read_csv(csv_file.stream)
    df = df.head()
    if csv_file.filename == "":
        return "No selected file", 400

    # text = extract_text_from_pdf(pdf_content)
    df_str = df.to_string(index=False)
    # print(df_str)
    # print(df.to_string())
    if not df.empty:
        value="Successfully Uploaded"
        return render_template("index.html",value=value)
    else:
        value = "Try to Re-Upload File"
        return render_template("index.html",value=value)

@app.route("/upload", methods=["POST"])
def upload_and_read_pdf():
    global text
    
    if "pdfFile" not in request.files:
        return "No PDF file uploaded", 400
    # print(type(request))
    # print(request.files["pdfFile"])
    pdf_file = request.files["pdfFile"]
    print(type(pdf_file))
    if pdf_file.filename == "":
        return "No selected file", 400

    pdf_content = pdf_file.read()
    text = extract_text_from_pdf(pdf_content)
    

    if text:
        value="Successfully Uploaded"
        return render_template("index.html",value=value)
    else:
        value = "Try to Re-Upload File"
        return render_template("index.html",value=value)

@app.route("/display", methods=["GET","POST"])
def display_sentence():
    if request.form.get('action1') == "Send":
        query = request.form['text']
        result = display_res(query,df_str)
        # print("Result",result)
        return render_template("index.html",value=result)
    

if __name__ == "__main__":
    app.run(port=8000,debug=True)


