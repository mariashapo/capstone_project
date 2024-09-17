Predicting Gentrification
==============================

**Research Aim:**

Gentrification refers to a demographic transformation of a neighbourhood in which lower-income groups are displaced by the influx of residents from a higher social class. Gentrification has been studied extensively in recent years due to its increasing occurrence in major cities, with notable examples of Hackney in London or Wicker Park in Chicago. However, while gentrification can contribute to local economic development and revitalisation, it often results in the displacement of long-term residents, cultural erosion, and rising living costs, making it a highly contested and complex issue. 

Due to its increasing occurrence in major cities, gentrification has been studied extensively in the quest to achieve inclusive and sustainable urban development. However, in most cases, gentrification is evaluated after it occurs. To address this issue, researchers have been developing methodologies that predict gentrification before it happens. To contribute to this discourse, this project creates a machine-learning workflow to analyse building permits in Chicago. By predicting gentrification and understanding the factors that contribute to it, policymakers and communities can take proactive measures to mitigate its negative impacts and promote equitable and inclusive urban development. 

**Datasets:**

This project will be exploring 2 type of datasets: Chicago Planning Application Dataset (2006 to present, updated daily) and US Census Income Datasets (one for each year from 2010 to 2021).  


Link to raw datasets: https://drive.google.com/drive/folders/1e5dHubh1Ij9XcA1TvWWu_dT2S4CIFYhT?usp=share_link <br />
Original dataset was found on Chicago Data Portal https://data.cityofchicago.org/Buildings/Building-Permits/ydr8-5enu <br />
Data to evaluate each geographical 'unit' was found on https://data.census.gov/ <br />

Project Organization
==============================
In this folder, you will find the following:
	

	data:
		clean:
		-income_cleaned
		-permits_cleaned

		interim:
		-pickled files

		raw:
		#has to be downloaded from Google Drive
		- Building_Permits.csv

			income:
			#has to be downloaded from Google Drive
			- ACSST5Y2010.S1901-Data



	notebooks:
	- 1. PERMITS_CLEANING+EDA
	- 2. INCOME_CLEANING+EDA
	- 3. MERGE+EDA+MODEL_PREP
	- 4. CLASSIFICATION_MODEL
	- functions.py
		images:
			- Plotly_1
			- Plotly_2
	- setup		
	- README.txt
==============================

