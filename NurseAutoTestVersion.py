#Core pkgs
import streamlit as st
st.set_page_config(page_title= 'Nurse Call Data ML App',page_icon='👨‍💻',layout = 'wide')
#EDA pkgs
import pandas as pd
import numpy as np

#Data visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
#%matplotlib inline

matplotlib.use('Agg')

#ML pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy 
from sklearn.metrics import confusion_matrix #to show the confusion matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeRegressor
import os


#Arima pckgs
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import scipy.interpolate as interpolate
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA


from datetime import datetime, timedelta

#utils
import base64
import time
timestr = time.strftime("%Y%m%d.%H%M%S")

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',FutureWarning)
import itertools
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
from IPython.display import HTML

#fxtn for  file
def csv_downloader(data):
	csvfile=data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = "new_text_file_{}_.csv".format(timestr)
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href, unsafe_allow_html=True)


def save_uploaded_file(uploadedfile):
	with open(os.path.join("E:/HOLIDAY/New folder/4th Year/Dashboard final GUI/",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
	return st.success("Saved file: {} in E:/HOLIDAY/New folder/4th Year/Dashboard final GUI".format(uploadedfile.name))

# Class
class FileDownloader(object):
	"""docstring for FileDownloader
	>>> download = FileDownloader(data,filename,file_ext).download()
	"""
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,timestr,self.file_ext)
		st.markdown("#### To Download Your Predictions File ###")
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Click Here!!</a>'
		st.markdown(href,unsafe_allow_html=True)





#@st.cache(suppress_st_warning=True) 



#@st.cache(suppress_st_warning=True)  # 👈 Changed this
def main():
    #st.title("My Auto Nurse Call Data ML App")
    #st.title("IBRAHIM ALI ABDI P15/37272/2016")
    st.markdown("<h1 style='text-align: center; color: yellow;'>My Auto Nurse Call Data Analytics Tool</h1>", unsafe_allow_html=True)

    activities = ["EDA","Plot Visualization","Model Building","About","Help"]
    choice = st.sidebar.selectbox("Select Activity",activities)

    if st.checkbox("Help"):
       #st.header("Email")
       st.markdown("<h5 style='text-align: center; color: blue;'>How to Use</h3>", unsafe_allow_html=True)
       st.markdown("<p style='text-align: center; color: white;'>✅In the upper left corner click on the '>' symbol to pull the sidebar and select analytics activities.<br>✅Below you should first import the required dataset that was logged by the real-time monitor desktop application </p>", unsafe_allow_html=True)
       #st.info("alibra@students.uonbi.ac.ke")
       st.markdown(" <i style='text-align: center; color: white;'>✅The procedures are top-down and step-wise</i>", unsafe_allow_html=True)
       st.markdown(" <i style='text-align: center; color: white;'>✅In the event of an Error i.e 'UnboundLocalError: local variable 'kpi2' referenced before assignment',<br> Ensure you have imported the right dataset i.e., sample2.csv data</i>", unsafe_allow_html=True)
       #st.text("In the upper left corner click on the '>' symbol to pull the sidebar and select analytics activities ")
       #st.text("Below you should first import the required dataset that was logged by desktop application real-time monitor ")
       st.text("Maintained by IBRAHIM ALI. You can reach me through ✅alibra@students.uonbi.ac.ke or alibra71@gmail.com")

    if choice =='EDA':
       
       #st.subheader("Exploratory Data Analysis")
       st.markdown("<h3 style='text-align: center; color: yellow;'>Exploratory Data Analysis</h3>", unsafe_allow_html=True)

       data = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
       if data is not None:
        # Making a list of missing value types
        #missing_values = ["n/a", "na", "--","0"]
        # Replace using median 

       	df = pd.read_csv(data)
       	#calculates and replaces missing values with median
       	median = df['BloodOxygen'].median()
        df['BloodOxygen'].fillna(median, inplace=True)

        median1 = df['HeartRate'].median()
        df['HeartRate'].fillna(median1, inplace=True)

        median2 = df['BodyTemp'].median()
        df['BodyTemp'].fillna(median2, inplace=True)

       	#Below removes the unmwanted columns in the logged data csv file
       	# to_drop = ['UNIX Timestamp (Milliseconds since 1970-01-01)','Sample Number (100 samples per second)']
       	# df.drop(to_drop, inplace=True, axis=1)
        st.markdown("<h6 style='text-align: center; color: yellow;'>Below is the dataset you imported</h6>", unsafe_allow_html=True)
       	st.dataframe(df)

        #st.markdown("## Exploring The Dataset Above")
        st.markdown("<h3 style='text-align: center; color: yellow;'>Exploring Your Dataset Above</h3>", unsafe_allow_html=True)

        # kpi 1 
        kpi1, kpi2 = st.beta_columns(2)
        with kpi1:
          #st.markdown("**Features**")
          st.markdown("<h4 style='text-align: center; color: yellow;'>Shape of Data</h4>", unsafe_allow_html=True)
          st.write(df.shape)
          st.info("Above values shows the number of rows,columns in the Dataset ")
          #st.info("rows,columns in the Dataset") 

          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

        with kpi2:
          #st.markdown("**Attributes**")
          st.markdown("<h4 style='text-align: center; color: yellow;'>Attributes/Columns in the dataset above</h4>", unsafe_allow_html=True)
          all_columns = df.columns.to_list()
          st.write(all_columns)
          st.info("Above shows the attributes/features in the Dataset ")
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

        

          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)


        #st.markdown("## Select Features of interest To Display")
        st.markdown("<h2 style='text-align: center; color: yellow;'>Select Features of interest To Display</h2>", unsafe_allow_html=True)
        selected_columns = st.multiselect("Select Columns you want to check for Correlations",all_columns)
        new_df = df[selected_columns]
        st.dataframe(new_df)

        #st.markdown("<hr/>",unsafe_allow_html=True)


        st.markdown("<hr/>",unsafe_allow_html=True) #for the next row 


        #st.markdown("## Select Features of interest To Display")
        st.markdown("<h2 style='text-align: center; color: yellow;'>Below is the Summary Statistics of the Dataset</h2>", unsafe_allow_html=True)
        st.info("Statistics such as mean, standard deviations and quartiles of where most instances are distributed in the dataset logged by the wearable are as below")
        st.write(df.describe())


        st.markdown("<hr/>",unsafe_allow_html=True)

        


    elif choice =='Plot Visualization':
        #st.subheader("Data Visualization")
        st.markdown("<h3 style='text-align: center; color: yellow;'>Visualization of Dataset</h3>", unsafe_allow_html=True)

        datas = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
        if datas is not None:
       	 # Replace using median 
       	 df = pd.read_csv(datas)
       	 df.columns = ["SampleNumber","UNIXTimestamp","RecordedInstanceTime","HeartRate","BloodOxygen","BodyTemp"]
         #df.index = pd.DatetimeIndex(df.index).to_period('S')
       	 #calculates and replaces missing values with median
       	 median = df['BloodOxygen'].median()
         df['BloodOxygen'].fillna(median, inplace=True)

         median1 = df['HeartRate'].median()
         df['HeartRate'].fillna(median1, inplace=True)

         median2 = df['BodyTemp'].median()
         df['BodyTemp'].fillna(median2, inplace=True)


       	 #Below removes the unmwanted columns in the logged data csv file
       	 #to_drop = ['UNIXTimestamp','SampleNumber']
       	 #to_drop = ['UNIXTimestamp','SampleNumber','RecordedInstanceTime']
       	 #to_drop = ['UNIXTimestamp']
       	 #to_drop1=['SampleNumber']
       	 #df.drop(to_drop, inplace=True, axis=1)
       	 st.dataframe(df)
       	 to_drop = ['UNIXTimestamp','SampleNumber','RecordedInstanceTime']
       	 df.drop(to_drop, inplace=True, axis=1)
         #st.markdown("## KPI First Row")
         st.markdown("<h2 style='text-align: center; color: yellow;'>Key Performance Indicators</h2>", unsafe_allow_html=True)
         st.markdown("<p style='text-align: center; color: white;'>Below shows how features are related and distributed.<br>Hover above the charts and select the expand button to zoom into the charts</p>", unsafe_allow_html=True)
         # kpi 1 
         kpi1, kpi2, kpi3 = st.beta_columns(3)
         with kpi1:
          #st.markdown("**Correlation KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Feature Correlation KPI</h4>", unsafe_allow_html=True)
          #number1 = 111 
          st.set_option('deprecation.showPyplotGlobalUse', False)
          st.write(sns.heatmap(df.corr(), annot=True,cmap="YlGnBu"))
          st.pyplot()          
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number1}</h1>", unsafe_allow_html=True)

        with kpi2:
          #st.markdown("**Second KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Feature Distribution KPI</h4>", unsafe_allow_html=True)
          number2 = 222 
          all_columns = df.columns.to_list()
          columns_to_plot = st.selectbox("Select Only 1 Target Column to Render Where Most Activity Occurred",all_columns)
          pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct="%.f%%")
         
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)


        with kpi3:
          #st.markdown("**Plot Selected KPI**")
          st.markdown("<h4 style='text-align: center; color: White;'>Plot of Feature Distribution KPI</h4>", unsafe_allow_html=True)
          number3 = 333 
          st.write(pie_plot)
          st.pyplot()           
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number3}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)

        st.markdown("<h2 style='text-align: center; color: yellow;'>Custom Visual Render Charts</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;'>Please select your chart of choice.<br>You can only interact with one chart at a time as rendering all charts simultaneously consumes heavy computer memory</p>", unsafe_allow_html=True)
        #datac = st.beta_columns(1)
        #with datac:
        number2 = 222
        all_columns_names = df.columns.tolist()
        type_of_plot = st.selectbox("Select Type of Chart Plot",["area","bar","line","hist","box","kde"])
        selected_columns_names = st.multiselect("Select One Column To Plot",all_columns_names)
        if st.button("Generate Plot"):
          st.success("I'm now Generating your Customizable Plot of type {} for {}".format(type_of_plot,selected_columns_names))

          if type_of_plot == 'area':
            cust_data = df[selected_columns_names]
            st.markdown("<h2 style='text-align: center; color: yellow;'>Your Rendered Chart</h2>", unsafe_allow_html=True)
            st.area_chart(cust_data)
            st.markdown("<hr/>",unsafe_allow_html=True)

          elif type_of_plot == 'bar':
            cust_data = df[selected_columns_names]
            st.markdown("<h2 style='text-align: center; color: yellow;'>Your Rendered Chart</h2>", unsafe_allow_html=True)
            st.bar_chart(cust_data)
            st.markdown("<hr/>",unsafe_allow_html=True)

          elif type_of_plot == 'line':
            cust_data =  df[selected_columns_names]
            st.markdown("<h2 style='text-align: center; color: yellow;'>Your Rendered Chart</h2>", unsafe_allow_html=True)
            st.line_chart(cust_data)
            st.markdown("<hr/>",unsafe_allow_html=True)

          elif type_of_plot:
            cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
            st.markdown("<h2 style='text-align: center; color: yellow;'>Your Rendered Chart</h2>", unsafe_allow_html=True)
            st.write(cust_plot)
            st.pyplot()
            st.markdown("<hr/>",unsafe_allow_html=True)

        
        #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)

        st.markdown("<hr/>",unsafe_allow_html=True)






 







    
    elif choice =='Model Building':
       st.subheader("Building Machine Learning Model")
       datas = st.file_uploader("Please Upload Your Sensors Dataset",type=["csv","txt","xls"]) 
       if datas is not None:
       	 # Making a list of missing value types
        #missing_values = ["n/a", "na", "--","0"]
        # Replace using median 

       	data = pd.read_csv(datas)
        #data['RecordedInstanceTime'] = pd.DatetimeIndex(data['RecordedInstanceTime']).to_period('S')
        #data.set_index(data['RecordedInstanceTime'],inplace=True)
        #data.drop('RecordedInstanceTime',axis=1,inplace=True)



        #median = data['BloodOxygen'].median(

        #data['BloodOxygen'].fillna(median, inplace=True)
        #data['HeartRate'].fillna(method='ffill', inplace=True)
        data['HeartRate']=data['HeartRate'].replace(0,data['HeartRate'].mode()).astype(int)
        #data['HeartRate']=data['HeartRate'].replace(2,data['HeartRate'].mean()).astype(int) #to replace value 2
        data['HeartRate']=data['HeartRate'].replace(200,data['HeartRate'].median()).astype(int) #to replace value 200
        data['HeartRate']=data['HeartRate'].replace(204,data['HeartRate'].median()).astype(int) #to replace value 204
        data['HeartRate']=data['HeartRate'].replace(179,data['HeartRate'].mean()).astype(int)#to replace value 179 
        data['HeartRate']=data['HeartRate'].replace(134,data['HeartRate'].mean()).astype(int)
        #data['HeartRate']=data['HeartRate'].replace(6,data['HeartRate'].mean()).astype(int) #to replace value 2
        data['BloodOxygen']=data['BloodOxygen'].replace(0,data['BloodOxygen'].mode()).astype(int)
        data['BodyTemp']=data['BodyTemp'].replace(0,data['BodyTemp'].mean()).astype(int)

        st.markdown("<h2 style='text-align: center; color: yellow;'>Key Performance Indicators</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;'>Below shows how features are related and distributed.<br>Hover above the charts and select the expand button to zoom into the charts</p>", unsafe_allow_html=True)
        # kpi 1 
        kpi1, kpi2 = st.beta_columns(2)
        with kpi1:
          st.markdown("<h4 style='text-align: center; color: yellow;'>Feature Engineering</h4>", unsafe_allow_html=True)
          data = data.dropna(axis=0)
          data.set_index(data['RecordedInstanceTime'],inplace=True)
          data.drop('RecordedInstanceTime',axis=1,inplace=True)
          data.columns = ["SampleNumber","UNIXTimestamp","HeartRate","BloodOxygen","BodyTemp"]
          to_drop = ['UNIXTimestamp']
          data.drop(to_drop, inplace=True, axis=1)
          to_drop1 = ['SampleNumber']
          data.drop(to_drop1, inplace=True, axis=1)
          all_columns_names = data.columns.tolist()
          #type_of_plot = st.selectbox("Select Type of Chart Plot",["area","bar","line","hist","box","kde"])
          selected_columns_names = st.selectbox("Select Your Target Feature To perform ML",all_columns_names)
          #pie_plot = data[columns_to_plot].value_counts().plot.pie(autopct="%.f%%")
          st.write(data[selected_columns_names])

        with kpi2:
          st.markdown("<h4 style='text-align: center; color: yellow;'>Your selected Feature Distribution</h4>", unsafe_allow_html=True)
          st.line_chart(data[selected_columns_names])
          #st.markdown(f"<h1 style='text-align: center; color: red;'>{number2}</h1>", unsafe_allow_html=True)
        st.markdown("<hr/>",unsafe_allow_html=True)


        #st.markdown("## Select Features of interest To Display")


        st.markdown("<h2 style='text-align: center; color: yellow;'>Below is Your target Dataset Feature I'll perform Machine Learning on.</h2>", unsafe_allow_html=True)
        kp1, kp2 = st.beta_columns(2)
        with kp1:
          st.markdown("<h4 style='text-align: center; color: yellow;'>Feature Engineering Skeleton</h4>", unsafe_allow_html=True)
          #data = data.dropna(axis=0)
          series_value = data[selected_columns_names].values
          data_mean= data.rolling(window=0).mean()
          value = pd.DataFrame(series_value)
          #create a baseline model, assumption the recent history is the best reflection of the future
          data_df = pd.concat([value,value.shift(0)],axis=1)  
          data_df.columns = ['Actual_Heart_Rate','Forecast_Heart_Rate']
          st.write(data_df.head())
          st.info("Note the figure above shows where Predicted values will be printed against actual values.Now simulated with only actuals")
          data_error = mean_squared_error(data_df.Actual_Heart_Rate,data_df.Forecast_Heart_Rate)
          st.text("mean_squared_error of the feature is:")
          data_error
          #np.sqrt(data_error)
          #st.write(series_value)

        with kp2:
          st.markdown("<h4 style='text-align: left; color: yellow;'>Target Feature Statistics</h4>", unsafe_allow_html=True)
          st.write(data[selected_columns_names].describe())
        st.markdown("<hr/>",unsafe_allow_html=True)

        st.markdown("<h4 style='text-align: center; color: yellow;'>Target Feature Statistics</h4>", unsafe_allow_html=True)

        st.markdown("<p style='text-align: center; color: green;'>Below shows two charts, Autocorrelation and Partial Autocorrelation.<br>The charts are very useful when determining model prediction parameters on the y-axes.<br> Autocorrelation is used to identify parameter Q while the other to identify D value used in the ARIMA model.<br>ARIMA model computes through Autoregressive (p) Integrated (d) and Moving Average (q)</p>", unsafe_allow_html=True)
        rel1,rel2 = st.beta_columns(2)
        with rel1:
          st.markdown("<h4 style='text-align: center; color: yellow;'>Correlation in Instances</h4>", unsafe_allow_html=True)
          st.write(plot_acf(data[selected_columns_names]))

        with rel2:
          st.markdown("<h4 style='text-align: center; color: yellow;'>Partial Correlation in Instances</h4>", unsafe_allow_html=True)
          st.write(plot_pacf(data[selected_columns_names]))
          st.markdown("<hr/>",unsafe_allow_html=True)


        #first create testing data
        #will now create the model p=2,3 d=0 & q=3,4
        data_train =data[selected_columns_names][0:75597] #this is 70% of the sample size i.e., sample2.csv specifically
        data_test =data[selected_columns_names][75597:94496] #roughly 30% of sample
        data_model = ARIMA(data_train, order=(0,1,1))
        data_model_fit = data_model.fit()

        st.markdown("<h2 style='text-align: center; color: yellow;'>Predictions</h>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: white;'>Below shows two charts, Autocorrelation and Partial Autocorrelation.<br>The charts are very useful when determining model prediction parameters on the y-axes.<br> Autocorrelation is used to identify parameter Q while the other to identify D value used in the ARIMA model.<br>ARIMA model computes through Autoregressive (p) Integrated (d) and Moving Average (q)</p>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: yellow;'>Prediction Model Statistics of Your Selected Target</h4>", unsafe_allow_html=True)
        LRresult = (data_model_fit.summary().tables[0])
        st.table(LRresult)
        st.markdown("<hr/>",unsafe_allow_html=True)


        st.markdown("<h4 style='text-align: center; color: yellow;'>Predictions vs Actuals</h4>", unsafe_allow_html=True)
        #st.markdown("<p style='text-align: center; color: green;'>Below shows two charts, Autocorrelation and Partial Autocorrelation.<br>The charts are very useful when determining model prediction parameters on the y-axes.<br> Autocorrelation is used to identify parameter Q while the other to identify D value used in the ARIMA model.<br>ARIMA model computes through Autoregressive (p) Integrated (d) and Moving Average (q)</p>", unsafe_allow_html=True)
        data_forecast = data_model_fit.forecast(steps =18899)[0]
        Forecast_HeartRate=pd.DataFrame(data_forecast, columns=['Prediction_HeartRate'])
        #data[selected_columns_names].reset_index(level=['RecordedInstanceTime'])  #converts the index into a column
        data1=pd.DataFrame(data_test)
        data1.insert(1, "PredictedFeatureValue",data_forecast, True)
        #data1.reset_index(level=['RecordedInstanceTime']) 
        #data1.reset_index(drop=False, inplace=True) 

        rel3,rel4 = st.beta_columns(2)

        with rel3:
          st.markdown("<h4 style='text-align: left; color: yellow;'>The output</h4>", unsafe_allow_html=True)
          st.write(data1)

        with rel4:
          st.set_option('deprecation.showPyplotGlobalUse', False)
          #data1.reset_index(level=['RecordedInstanceTime']) 
          #data1.reset_index(drop=False, inplace=True)
          st.write(data1.plot())
          st.pyplot()

        st.markdown("<hr/>",unsafe_allow_html=True)


        st.markdown("<h4 style='text-align: center; color: yellow;'>Your Predictions File is Ready For Download </h4>", unsafe_allow_html=True)

        output = pd.DataFrame(data1)
        output.to_csv('Patient Vitals Predictions.csv', index = True)
        download = FileDownloader(output.to_csv(),file_ext='csv').download()

        st.markdown("<hr/>",unsafe_allow_html=True)
        


        




        








    
    elif choice =='About':
       st.subheader("About My Project ✨")
       #st.header("🌟Nurse Call Sensor Data ML Web Application")
       st.markdown("<h3 style='text-align: center; color: yellow;'>🌟Nurse Call Sensor Data ML Web Application🌟</h3>", unsafe_allow_html=True)
       st.markdown("<p style='text-align: center; color: white;'>📚This application realizes the completion of the functions of the other components used i.e., wearable vitals monitor, manual transmitter and a central receiver to address the scope and functionality of the nurse call alert system project.<br>📚Data logged by the real-time monitor application at central nurse desk can be analysed here to get an understanding of a long horizontal data inorder to assist with decision processes by health practitioners<br>📚The application predicts values in the dataset through ARIMA model that behaves very well on a long-stretch of data</p>", unsafe_allow_html=True)
       st.sidebar.header("About App")
       st.sidebar.info("An EDA App for Exploring Receiver Device Data logs")
       st.sidebar.header("Final Year Project")
       #st.sidebar.markdown("[Common ML Dataset Repo]("")")
       st.sidebar.header("Email")
       st.sidebar.info("alibra@students.uonbi.ac.ke")
       st.sidebar.text("Built with Streamlit Python Framework")
       st.sidebar.info("Maintained by IBRAHIM ALI")
       st.balloons()


if __name__=='__main__':
	main()


