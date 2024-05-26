import streamlit as st
import plotly.express as px


def plot_histograms(df, properties):
	for prop, selected in properties.items():
		if selected:
			fig = px.histogram(df, x=prop, nbins=30, title=prop, color_discrete_sequence=['orange'])
			fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
			st.plotly_chart(fig, use_container_width=True)
