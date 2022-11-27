import pickle
import pandas as pd
import dash
import dash_html_components as html 
import webbrowser
from dash.dependencies import Input,Output
import dash_core_components as dcc
app=dash.Dash()
project_name=None
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")
    #open browser automatically/opening a tab in browser in automation
    
def globalization():
    global df
    df=pd.read_csv('balanced_reviews.csv')
    df=df[df["overall"]!=3]
    j=0
    global a
    a={}
    df2=df.iloc[:,:-1].sample(10)
    for i in range(0,len(df2)):
        while j<len(df2):
            a[df2.iloc[i,1]]=df2.iloc[j,0]
            j+=1
            break
        
def create_app_ui():
    #actually creating a ui of my app
    main_layout=html.Div(
        [
        html.H2(id='Main_title',children=app.title,style={"text-align":"center","font-size":"250%","font-family":"helvetica"}),
        dcc.Textarea(id="textarea_review",
                     placeholder="Enter the Review here.....",
                     style={"width":"100%","height":200,"background-color":"grey","font-size":"100%","font-family":"verdana","color":"yellow","border":"5 px solid black"}
                     ),
             dcc.Dropdown(id="review",
                      options=[{"label":i,"value":j}for i,j in a.items()],
                      placeholder="Select Review"),
        html.Button(id="button_review",children="find_review",n_clicks=0),
        html.H1(id="result",children=None,style={"color":"red"})
        ]
        )
    #Div is a container
    return main_layout
@app.callback(
    Output("button_review","value"),
    [
    Input("textarea_review","value")
    ]
    )
def update_app_ui(textarea_value):
    if textarea_value:
            print("Data Type=",str(type(textarea_value)))
            print("Value=",str(textarea_value))
            result_list=check_review(textarea_value)
            textarea_value=None
            if result_list[0]==0:
                result="Negative"     
            elif result_list[0]==1:
                result="Positive"
            return result
    else:
        return None
             
@app.callback(
    Output("result","children"),
    [
    Input("button_review","value"),
    Input("button_review","n_clicks")
    ]
    )
def update_button_review(value,clicks):
    if clicks>0 and value: 
        if value=="Positive":
            return value
        elif value=="Negative":
            return value
        
@app.callback(
    Output("button_review","n_clicks"),
    [
     Input("button_review","value")
     ]
    )
def n_click(value):
    if not value:
        return 0
        
def load_model():
    #loading pickle and csv file in memory
    print("Busy in loading the model in memory")
    global df
    df=pd.read_csv("balanced_reviews.csv")
    file=open("pickle_model.pkl","rb")
    global pickle_model
    pickle_model=pickle.load(file)
    global vocab
    file=open("feature.pkl","rb")
    vocab=pickle.load(file)
def check_review(reviewText):
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    transformer=TfidfTransformer()
    loaded_vec=CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl","rb")))
    reviewText=transformer.fit_transform(loaded_vec.fit_transform([reviewText]))
    return pickle_model.predict(reviewText)
        
def main():
    #controll the flow of project
    #calling of all the function
    print("Welcome to My Project")
    load_model() 
    globalization()
    open_browser()
    global project_name
    project_name="Analysis of reviews"
    global app
    app.title=project_name
    #title of project
    app.layout=create_app_ui
    #it gives me a blank webpage control the ui of a webpage
    app.run_server()
    #start the server running infinte times
    print("this wiil not be executed")
    app=None
    project_name=None
    
if __name__=="__main__":
    main()