import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_object_dtype
from rdkit.Chem import PandasTools

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined" rel="stylesheet">
""",
			unsafe_allow_html=True)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
	"""
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
	modify = st.toggle("Add filters")

	if not modify:
		return df

	df = df.copy()

	# Try to convert datetimes into a standard format (datetime, no timezone)
	for col in df.columns:
		if is_object_dtype(df[col]):
			try:
				df[col] = pd.to_datetime(df[col])
			except Exception:
				pass

		if is_datetime64_any_dtype(df[col]):
			df[col] = df[col].dt.tz_localize(None)

	modification_container = st.container()

	with modification_container:
		to_filter_columns = st.multiselect("Filter data by", df.columns)
		for column in to_filter_columns:
			left, right = st.columns((1, 20))
			# Treat columns with < 10 unique values as categorical
			if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
				user_cat_input = right.multiselect(f"Values for {column}",
							df[column].unique(),
							default=list(df[column].unique()),
							)
				df = df[df[column].isin(user_cat_input)]
			elif is_numeric_dtype(df[column]):
				_min = float(df[column].min())
				_max = float(df[column].max())
				step = (_max-_min) / 10
				user_num_input = right.slider(f"Values for {column}",
						min_value=_min,
						max_value=_max,
						value=(_min, _max),
						step=step,
						)
				df = df[df[column].between(*user_num_input)]
			elif is_datetime64_any_dtype(df[column]):
				user_date_input = right.date_input(f"Values for {column}",
							value=(df[column].min(), df[column].max(),
							),
							)
				if len(user_date_input) == 2:
					user_date_input = tuple(map(pd.to_datetime, user_date_input))
					start_date, end_date = user_date_input
					df = df.loc[df[column].between(start_date, end_date)]
			else:
				user_text_input = right.text_input(f"Substring or regex in {column}", )
				if user_text_input:
					df = df[df[column].astype(str).str.contains(user_text_input)]

	return df


@st.cache_data
def display_dataframe(df: pd.DataFrame, placement: str = 'center') -> pd.DataFrame:
	if placement == 'center':
		container = st.container()
		for i in range(0, len(df), 50):
			container.dataframe(df[i:i + 50])
	elif placement == 'left':
		col1, col2 = st.columns(2)
		container = col1.container()
		for i in range(0, len(df), 50):
			container.dataframe(df[i:i + 50])
	elif placement == 'right':
		col1, col2 = st.columns(2)
		container = col2.container()
		for i in range(0, len(df), 50):
			container.dataframe(df[i:i + 50])
	else:
		st.error("Invalid placement argument for displaying dataframe. Use 'center', 'left', or 'right'.")
	return df


def save_dataframe_to_sdf(dataframe, file_path, molecule_column='Molecule', id_column='ID'):
	"""Helper function to save DataFrame to SDF format."""
	PandasTools.WriteSDF(dataframe,
			file_path,
			molColName=molecule_column,
			idName=id_column,
			properties=list(dataframe.columns))
