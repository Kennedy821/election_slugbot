import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gp
import h3
from shapely.geometry import Polygon
import os

def splice_geometry_into_smaller_chunks(geometry_file):
    import numpy as np

    temp_gdf = gp.GeoDataFrame([],geometry=geometry, crs=4326)
    
    minx, miny, maxx, maxy = temp_gdf.total_bounds
    n_cells = 10
    x = np.linspace(minx, maxx, n_cells)
    y = np.linspace(miny, maxy, n_cells)
    cells = []
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            cells.append(Polygon([(x[i], y[j]), (x[i+1], y[j]), (x[i+1], y[j+1]), (x[i], y[j+1])]))
    grid = gp.GeoDataFrame(cells, columns=['geometry'], crs=gdf.crs)

    split_geometries = gp.overlay(temp_gdf, grid, how='intersection')

    return split_geometries

def flatten(nested_list):
    """
    Flatten a list of lists.

    Parameters:
    nested_list (list): A list of lists to be flattened.

    Returns:
    list: A single, flattened list.
    """
    return [item for sublist in nested_list for item in sublist]


def polygon_to_h3(gdf, resolution=8):
    """
    Convert a GeoDataFrame of polygons to H3 hexagons at the specified resolution.

    Parameters:
    gdf (gp.GeoDataFrame): GeoDataFrame containing polygon geometries.
    resolution (int): H3 resolution level (default is 11).

    Returns:
    gp.GeoDataFrame: GeoDataFrame containing H3 hexagon geometries.
    """
    def polygon_to_h3_indices(polygon, resolution):
        # Get the H3 indices for the polygon
        h3_indices = h3.polyfill(polygon.__geo_interface__, resolution, geo_json_conformant=True)
        return list(h3_indices)
    
    h3_indices_list = []
    for polygon in gdf.geometry:
        try:

            if isinstance(polygon, Polygon):
                h3_indices_list.extend(polygon_to_h3_indices(polygon, resolution))
            else:
                for sub_polygon in polygon:
                    h3_indices_list.extend(polygon_to_h3_indices(sub_polygon, resolution))
                    
        except Exception as e:
            print(f"{polygon} could not be processed initially")

            splice_polygon = splice_geometry_into_smaller_chunks(polygon)

            if isinstance(splice_polygon, Polygon):
                h3_indices_list.extend(polygon_to_h3_indices(splice_polygon, resolution))
            else:
                for sub_polygon in splice_polygon:
                    h3_indices_list.extend(polygon_to_h3_indices(sub_polygon, resolution))
            print(f"{polygon} was subsequently processed as a spliced polygon")
            
            pass
        
    # Create a new GeoDataFrame from the H3 indices
    hexagons = [Polygon(h3.h3_to_geo_boundary(h, geo_json=True)) for h in h3_indices_list]
    
    h3_gdf = gp.GeoDataFrame(h3_indices_list,geometry=hexagons, crs=gdf.crs).rename(columns={0:f"h3_index"})
    
    return h3_gdf



st.set_page_config(layout="wide")

logo,gap, header = st.columns([1,1,9])
with logo:
    # Display the image
    st.image("recolored_image.png", width=75)

    # Custom CSS to position the image
    st.markdown(
        """
        <style>
        [data-testid="stImage"] {
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 1;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
with header:
    st.title("UK 2024: Election Predictions")
st.header("GPT4o")

# Dictionary to map party to colors
party_colors = {
    "Conservative": "royalblue",
    "Labour": "darkred",
    "Lib Dems": "gold",
    "Greens": "green",
    "Reform": "purple",
    "SNP": "yellow",
    "Other": "teal",
    "Marginal": "black"
}


# Create DataFrame
df = pd.read_excel(f"{os.getcwd()}/results/FINAL RESULTS Election Slugbot base model.xlsx", sheet_name="gpt4o")

# standardise the names

df.loc[df.standardised_party_winner=="Conservative","standardised_party_winner"] = "Conservative"
df.loc[df.standardised_party_winner=="Labour","standardised_party_winner"] = "Labour"
df.loc[df.standardised_party_winner=="Liberal Democrat","standardised_party_winner"] = "Lib Dems"
df.loc[df.standardised_party_winner=="SNP","standardised_party_winner"] = "SNP"
df.loc[df.standardised_party_winner=="Green","standardised_party_winner"] = "Greens"
df.loc[~df.standardised_party_winner.isin(["Conservative","Labour","Lib Dems","SNP","Green"]),"standardised_party_winner"] = "Other"


aggregated_df = df.groupby(df.columns[-1]).nunique()[[df.columns[0]]].reset_index()
aggregated_df["total_seats"] = aggregated_df[aggregated_df.columns[1]].sum()
aggregated_df["pct"] = aggregated_df[aggregated_df.columns[1]] / aggregated_df["total_seats"]

# load the geometry file
gdf = gp.read_file("uk_pcon_map_w_2024_election_predictions.geojson")[["pcon_name","geometry"]].to_crs(4326)

gdf = gdf.set_index("pcon_name").explode().reset_index()
# gdf
h3_gdf = polygon_to_h3(gdf.set_index("pcon_name"))
gdf = h3_gdf.sjoin(gdf, predicate="intersects")


gdf = gdf.merge(df[["pcon_name","standardised_party_winner"]], on="pcon_name", how="left").rename(columns={"pcon_name":"Constituency","standardised_party_winner":"Party"})
gdf["Latitude"] = gdf.geometry.centroid.y
gdf["Longitude"] = gdf.geometry.centroid.x
gdf["Party"] = gdf["Party"].str.replace("too hard to call","Marginal")
# gdf = gdf
# st.dataframe(gdf.drop(columns="geometry"))
# gdf["h3_index"] = gdf.apply(lambda x: h3.geo_to_h3(x.Latitude, x.Longitude, 8), axis=1)



# seat_df = pd.DataFrame(seats)
seat_df = gdf.drop(columns="geometry").sort_values("Party")
seat_counts = gdf[['Party','Constituency']].drop_duplicates()["Party"].value_counts().to_dict()
# Page Title

# # Stacked Horizontal Bar Chart
# st.subheader("Party Percentage")
# fig = go.Figure()

# fig.add_trace(go.Bar(
#     y=aggregated_df[aggregated_df.columns[0]],
#     x=aggregated_df['pct'],
#     orientation='h',
#     marker=dict(color=['blue', 'green', 'red']),
#     name='Percentage'
# ))

# fig.update_layout(barmode='stack')
# st.plotly_chart(fig)


# # Marimekko Chart
# st.subheader("Seats Distribution")

# # ***Example DataFrame:**

# data = {'Party Name': ['Party A', 'Party B', 'Party C', 'Party D'],
#         'Seats Won': [35, 28, 17, 10]}
# df = pd.DataFrame(data)

# # **3. Marimekko Chart Creation**

# # Customize the plot
# fig_mekko = plt.figure(figsize=(10, 6))  # Adjust figure size as needed
# ax = plt.gca()
# # Create the Marimekko chart
# sns.barplot(x='Party Name', y='Seats Won', data=df,
#             palette="Set2",  # Choose a color palette
#             orient='v',  # Vertical bars
#             linewidth=0.5)  # Add a slight line for separation

# # Add labels and title
# plt.xlabel('Political Parties')
# plt.ylabel('Seats Won')
# plt.title('Political Party Seat Distribution')

# # Show the plot
# st.plotly_chart(fig_mekko)


st.markdown("## Predicted seat count")

kpi_columns = st.columns(len(seat_counts))

for i, (party, count) in enumerate(seat_counts.items()):
    if party not in ["SNP","Lib Dems"]:
        kpi_columns[i].markdown(
            f"""
            <div style="background-color: {party_colors[party]}; padding: 8px; border-radius: 9px; text-align: center; color: white;">
                <h5 style="color: white;text-align: center;">{party}</h5>
                <p style="text-align: center;">Seats:{count}</p>
            </div>
            """, unsafe_allow_html=True
        )
    else:
        kpi_columns[i].markdown(
            f"""
            <div style="background-color: {party_colors[party]}; padding: 8px; border-radius: 9px; text-align: center; color: black;">
                <h5 style="color: black;text-align: center;">{party}</h5>
                <p style="text-align: center;">Seats:{count}</p>
            </div>
            """, unsafe_allow_html=True
        )
        


# # Map with Pydeck using HexagonLayer
# st.header("Electoral Predictions Map")
# fig = plt.figure(figsize=(3,10))
# ax = plt.gca()




# # Plot each party with their respective color
# for party, color in party_colors.items():
#     if party not in ["SNP","Lib Dems"]:
#         try:
#             gdf[gdf["Party"] == party].plot(color=color, ax=ax, alpha=0.8, edgecolor="w", linewidth=0.1)
#         except Exception as e:
#             print(e)
#             pass
#     else:
#         try:
#             gdf[gdf["Party"] == party].plot(color=color, ax=ax, alpha=0.8, edgecolor="k", linewidth=0.1)
#         except Exception as e:
#             print(e)
#             pass

# # Create custom legend
# legend_labels = list(party_colors.keys())
# legend_colors = [plt.Line2D([0], [0], marker='o', color='grey', markerfacecolor=party_colors[party], markersize=10)
#                  for party in legend_labels]

# ax.legend(legend_colors, legend_labels, title="Parties", fontsize='small')
# # ax.legend(legend_colors, legend_labels, title="Parties", loc='lower right', bbox_to_anchor=(1, 0), fontsize='small')
# plt.axis("off")
# st.pyplot(fig)




# Define party colors
party_colors = {
    'Conservative': [0, 135, 220],
    'Labour': [220, 36, 31],
    'Liberal Democrats': [253, 187, 48],
    'SNP': [255, 242, 0],
    'Green Party': [0, 116, 63],
    'Plaid Cymru': [120, 190, 32],
    'Reform': [152, 0, 150]
}

# Add colors based on party
gdf['color'] = gdf['Party'].map({
    'Conservative': [0, 135, 220],
    'Labour': [220, 36, 31],
    'Liberal Democrats': [253, 187, 48],
    'SNP': [255, 242, 0],
    'Green Party': [0, 116, 63],
    'Plaid Cymru': [120, 190, 32],
    'Reform': [152, 0, 150]
})
# Map with Pydeck using PolygonLayer
viz_gdf = gdf[["Party","Constituency","h3_index"]].convert_dtypes()

# for the non-major locations we'll zoom up to h3 level 7 
# viz_gdf.loc[viz_gdf.Party.isin(["Other","Marginal","SNP"]),"h3_index"] = viz_gdf.loc[viz_gdf.Party.isin(["Other","Marginal","SNP"])]["h3_index"].apply(lambda x: h3.h3_to_parent(x,7))
viz_gdf = viz_gdf.drop_duplicates()
viz_gdf = viz_gdf.merge(df[["pcon_name","slugbot_predictions"]], left_on="Constituency", right_on="pcon_name", how="left").drop(columns="pcon_name").rename(columns={"slugbot_predictions":"prediction"})
st.markdown("## Predicted electoral map")


deck = pdk.Deck(
    map_provider="mapbox",
    map_style=pdk.map_styles.MAPBOX_LIGHT,

    # map_style=pdk.map_styles.SATELLITE

    initial_view_state = pdk.ViewState(
    latitude=gdf.geometry.centroid.y.mean(),
    longitude=-gdf.geometry.centroid.x.mean()-5,
    zoom=5,
    pitch=0,
    bearing=0,
    height=1000,width='100%'
    ),

    layers = [
        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Labour"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[255, 6, 20]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),

        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Conservative"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[0, 135, 220]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),


        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Lib Dems"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[253, 187, 48]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),

        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Reform"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[152, 0, 150]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),


        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="SNP"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[255, 242, 0]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),

        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Other"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[0, 128, 128]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),

        pdk.Layer(
            'H3HexagonLayer',
            viz_gdf[viz_gdf.Party=="Marginal"],
            # get_polygon='geometry.coordinates',
            pickable=True,
            stroked=False,
            filled=True,
            extruded=False,
            get_hexagon="h3_index",
            get_fill_color="[0, 0, 0]",
            get_line_color=[255, 255, 255],
            line_width_min_pixels=0,
            opacity = 0.4
        ),
    ],


    tooltip={
        # "html": "<b>Hex cell:</b> {h3_index} <br/> Constituency: {Constituency} <br/>Winner: {Party}"
        "html": "Constituency: {Constituency} <br/>Winner: {Party} <br/>Summary: {prediction}"

    }
    

)
st.pydeck_chart(deck)
