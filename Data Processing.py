# Importing Data Analysis Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading Data
df = pd.read_csv("data/airlines_flights_data.csv")
df.drop("index", axis=1, inplace=True)

print(df.head())
print(df.info())
print(df.describe().T)
print(df.isnull().sum())

# Showing all the Airlines with their number of flights in Horizontal Bar Graph
df["airline"].value_counts(ascending=True).plot.barh(color=["lightgreen", "lightblue"])
plt.title("Airlines wiht Frequencies")
plt.xlabel("Numer of Flights")
plt.ylabel("Airlines")
plt.savefig("imgs/visual_1.png")
plt.show()

# Show Bar Graphs representing the Departure Time & Arrival Time
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
deprature_time_indcies = df["departure_time"].value_counts().index
arrival_time_indcies = df["arrival_time"].value_counts().index
sns.countplot(data=df, x="departure_time", ax=ax[0], order=deprature_time_indcies, hue="departure_time")
sns.countplot(data=df, x="arrival_time", ax=ax[1], order=arrival_time_indcies, hue="arrival_time")
plt.savefig("imgs/visual_2.png")
plt.show()

# Show Bar Graphs representing the Source City & Destination City
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.countplot(data=df, y="source_city", hue="source_city", ax=ax[0], order=df["source_city"].value_counts(ascending=True).index)
sns.countplot(data=df, y="destination_city", hue="destination_city", ax=ax[1], order=df["destination_city"].value_counts(ascending=True).index)
plt.savefig("imgs/visual_3.png")
plt.show()

# Does price varies with airlines
airlines_price = df.groupby("airline")["price"].agg("mean").sort_values()
print(airlines_price)

# Drawing a Categorical Plot showing the Mean Ticket Price for each Airline
sns.catplot(x="airline", y="price", data=df, kind="bar", hue="class", errorbar=None, palette="rocket")
plt.savefig("imgs/visual_4.png")
plt.show()

# Does ticket price change based on the departure time and arrival time
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=df, x="departure_time", y="price", hue="departure_time", errorbar=None, ax=ax[0])
ax[0].set_xlabel("Departure Time")
ax[0].set_ylabel("Price")

sns.barplot(data=df, x="arrival_time", y="price", hue="arrival_time", errorbar=None, ax=ax[1])
ax[1].set_xlabel("Arrival Time")
ax[1].set_ylabel("Price")

plt.savefig("imgs/visual_5.png")
plt.show()

sns.relplot(x="arrival_time", y="price", data=df, col="departure_time", kind="line")
plt.savefig("imgs/visual_6.png")
plt.show()

# How the price changes with change in Source and Destination
sns.relplot(data=df, x="source_city", y="price", col="destination_city", kind="line")
plt.savefig("imgs/visual_7.png")
plt.show()

# How is the price affected when tickets are bought in just 1 or 2 days before departure?
sns.catplot(data=df, x="days_left", y="price", kind="bar", hue="days_left", aspect=2, errorbar=None)
plt.savefig("imgs/visual_8.png")
plt.show()

sns.relplot(data=df, x="days_left", y="price", kind="line", aspect=1.5)
plt.savefig("imgs/visual_9.png")
plt.show()

# How does the ticket price vary between Economy and Business class
sns.barplot(data=df, x="class", y="price", hue="airline", errorbar=None)
plt.savefig("imgs/visual_10.png")
plt.show()

# What will be the Average Price of Vistara airline for a flight from Delhi to Hyderabad in Business Class ?
mean_price = df[
    (df["airline"] == "Vistara")
    & (df["source_city"] == "Delhi")
    & (df["destination_city"] == "Hyderabad")
    & (df["class"] == "Business")
]["price"].mean().item()

print(mean_price)

# Displaying the distribution of the numberical columns
def display_distribution(data, numerical_cols, figure_name):
    fig, ax = plt.subplots(1, len(numerical_cols), figsize=(16, 6), dpi=95)
    for i, col in enumerate(numerical_cols):
        ax[i].hist(data=data, x=col)
        ax[i].set_xlabel(col, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"imgs/{figure_name}.png")
    plt.show()

numerical_cols = df.select_dtypes("number").columns
display_distribution(data=df, numerical_cols=numerical_cols, figure_name="distributions")

# Displaying Boxplots to show outliers
def display_outilers(data, numerical_cols, figure_name):
    fig, ax = plt.subplots(len(numerical_cols), 1, figsize=(7, 18), dpi=95)
    for i, col in enumerate(numerical_cols):
        ax[i].boxplot(data[col], vert=False)
        ax[i].set_ylabel(col)
    plt.tight_layout()
    plt.savefig(f"imgs/{figure_name}.png")
    plt.show()

numerical_cols = df.select_dtypes("number").columns
display_outilers(data=df, numerical_cols=numerical_cols, figure_name="with_outliers")

# Applying logarithimic transformation to handling the right skewing
right_skewed_data = ["duration", "price"]
df[right_skewed_data] = np.log1p(df[right_skewed_data])

display_distribution(df, right_skewed_data, "distribution_after_log")
display_outilers(df, right_skewed_data, "outliers_after_log")

# Ordinal Encoding for Ordinal Features
df["stops"] = df["stops"].map({"zero": 0, "one": 1, "two_or_more": 2})
df["class"] = df["class"].map({"Economy": 0, "Business": 1})

# Saving data after processing
data_ml_model = df.copy()
data_ml_model.to_csv("data/data_ml_model.csv", index=False)
