import requests
import pandas as pd

import streamlit as st


def query_chemfh(df: pd.DataFrame):
	url = "http://121.40.210.46:8110/api/admet"
	n = 2000

	param = {'SMILES': []}
	smiles_list = df['SMILES'].tolist()
	data = {"SMILES": smiles_list}

	for _, sublist in enumerate(divide_list(smiles_list, n)):
		param['SMILES'] = sublist

		response = requests.post(url, json=param)

		if response.status_code == 200:
			data = response.json()['data']
			result = transform(data)
			return result


def filter_by_properties(df, properties_to_filter):
	for prop, should_filter in properties_to_filter.items():
		if should_filter:
			df = df[df[prop] < 0.5]   # Assumes a threshold that needs adjusting based on the property's significance
	return df


def divide_list(lst, n):
	for i in range(0, len(lst), n):
		yield lst[i:i + n]


def transform(data):
	resultList = []
	for mol in data['data']:
		if not mol['data']:
			# Invalid SMILES
			tmp = {'smiles': mol['smiles']}
		else:
			tmp = dict({'smiles': mol['smiles']})
			for _, admet in mol['data'].items():
				for endpoint in admet:
					# endpoint is a dict
					tmp[endpoint['name']] = endpoint['value']
		resultList.append(tmp)
	return pd.DataFrame(resultList).fillna('Invalid SMILES')
