import streamlit as st
import pandas as pd
import os
import hopsworks
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta, date
from utils import LSTMModel, prepare_data_for_forecast, forecast
from features import feature_engineering
import base64

# Define constants
INPUT_FOLDER = 'prediction_dataframes'


@st.cache_data
def load_csv_data(filename):
    """
    Load data from a CSV file and cache the result to avoid reloading the file on every rerun.
    """
    file_path = os.path.join(INPUT_FOLDER, filename)
    return pd.read_csv(file_path)


@st.cache_data
def load_hopsworks_data():
    """
    Load data from Hopsworks and cache the result.
    """
    project = hopsworks.login()
    fs = project.get_feature_store()
    bitcoin_fg = fs.get_feature_group(name='bitcoin_analysis', version=1)
    data = bitcoin_fg.select_all()
    feature_view = fs.get_or_create_feature_view(
        name='bitcoin_analysis_training_fv',
        version=1,
        query=data
    )
    df = feature_view.get_batch_data()
    sorted_df = df.sort_values(by='id')
    sorted_df['date'] = pd.to_datetime(sorted_df['date'])
    sorted_df['year'] = sorted_df['date'].dt.year
    sorted_df['month'] = sorted_df['date'].dt.month
    sorted_df['day'] = sorted_df['date'].dt.day
    sorted_df['day_of_week'] = sorted_df['date'].dt.dayofweek

    return sorted_df

@st.cache_data
def load_local_data():
    start_date = date(2016, 1, 1)
    end_date = date.today() - timedelta(days=1)
    df = feature_engineering.prepare_data(start_date=start_date, end_date=end_date)
    df = prepare_data_for_forecast(df)
    return df

def calculate_investment_value(investment_amount, initial_price, final_price):
    """
    Calculate the investment value based on initial and final prices.
    """
    return (investment_amount / initial_price) * final_price


def plot_visualization(option, df):
    if option == 'Bitcoin Price and USD Index':
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bitcoin Price', color=color)
        ax1.plot(df['date'], df['close'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('USD Index', color=color)
        ax2.plot(df['date'], df['close_usd_index'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        plt.title('Bitcoin Price and USD Index Over Time')
        st.pyplot(fig)

    elif option == 'Bitcoin Price, Oil Price, and Gold Price':
        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bitcoin Price', color=color)
        ax1.plot(df['date'], df['close'], color=color, label='Bitcoin Price')
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()
        color = 'tab:green'
        ax2.set_ylabel('Oil Price', color=color)
        ax2.plot(df['date'], df['close_oil'], color=color, label='Oil Price')
        ax2.tick_params(axis='y', labelcolor=color)

        ax3 = ax1.twinx()
        color = 'tab:orange'
        ax3.spines["right"].set_position(("outward", 60))  # Offset the third axis
        ax3.set_ylabel('Gold Price', color=color)
        ax3.plot(df['date'], df['close_gold'], color=color, label='Gold Price')
        ax3.tick_params(axis='y', labelcolor=color)

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        lines_3, labels_3 = ax3.get_legend_handles_labels()
        lines = lines_1 + lines_2 + lines_3
        labels = labels_1 + labels_2 + labels_3
        ax1.legend(lines, labels, loc='upper left')

        fig.tight_layout()
        plt.title('Bitcoin Price, Oil Price, and Gold Price Over Time')
        st.pyplot(fig)

    elif option == 'Boxplot of Bitcoin Closing Prices by Year':
        fig, ax = plt.subplots(figsize=(12, 8))

        # Extract the year from the date column
        df['year'] = pd.to_datetime(df['date']).dt.year

        # Prepare the data for the boxplot
        data_by_year = [group['close'].values for name, group in df.groupby('year')]

        # Plot the boxplots for each year's closing prices
        ax.boxplot(data_by_year, labels=df['year'].unique())
        ax.set_title('Boxplot of Bitcoin Closing Prices by Year')
        ax.set_xlabel('Year')
        ax.set_ylabel('Closing Price (USD)')
        ax.set_xticklabels(df['year'].unique(), rotation=45)
        ax.grid(True)

        st.pyplot(fig)


def load_model_from_disk(input_path):
    import torch
    # import the model from the local directory
    lstm_model = LSTMModel(input_size=11, hidden_size=30, output_size=1)
    lstm_model.load_state_dict(torch.load(input_path))
    lstm_model.eval()

    return lstm_model

def get_today_bitcoin_price():
    btc_price = yf.download('BTC-USD', period='1d')['Close'].values[0]
    return btc_price

def get_bitcoin_price_selected_dates(start_date, end_date):
    return yf.download('BTC-USD', start=start_date, end=end_date)['Close'].values[0]

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url(data:image/jpg;base64,{encoded_string});
             background-size: cover;
             background-position: center;
             background-repeat: no-repeat;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


def main():
    set_background('images/background.jpg')

    st.title('Bitcoin Price Prediction')

    st.write("""
        Welcome to our Bitcoin Price Prediction application. 
        Here, you can explore various forecasts and visualizations related to Bitcoin prices. 
        Use the investment calculator to predict the future value of your investments based on the provided data.
    """)

    # Load data from Hopsworks
    # sorted_df = load_hopsworks_data()

    # TODO: Load data by downloading it instead of hopsworks later
    #  on change it to hopsworks again which is the commented line above
    data = load_local_data()
    lstm_model = load_model_from_disk('model/bitcoin_price_movement_prediction_lstm.pth')

    movement_interpretation = forecast(lstm_model, data, 100, 2)

    # Load CSV data
    #forecast_df = load_csv_data('forecast_7_days.csv')
    #investment_forecast_df = load_csv_data('forecast_30_days.csv')

    st.header('Bitcoin current portfolio calculator')

    # Create a calendar selector
    selected_date = st.date_input("Select a date", value=None, min_value=datetime(2014, 10, 1), max_value=date.today())
    if selected_date:
        btc_owned = st.number_input('Enter the amount of Bitcoin you own', min_value=0.0, step=0.01)
        selected_date = selected_date
        following_day_date = selected_date + timedelta(days=1)
        btc_price_selected_date = get_bitcoin_price_selected_dates(selected_date.strftime('%Y-%m-%d'), following_day_date.strftime('%Y-%m-%d'))
        btc_price_today = get_today_bitcoin_price()

        # Display the selected date
        st.write("The price of the bitcoin at your selected date is: ", btc_price_selected_date)
        st.write("The price of the bitcoin today is: ", btc_price_today)

        if btc_owned > 0:
            # Calculate the value of the investment on the selected date and today
            initial_investment_value = btc_owned * btc_price_selected_date
            current_investment_value = btc_owned * btc_price_today

            # Calculate the gain or loss
            gain_loss = current_investment_value - initial_investment_value
            gain_loss_percentage = (gain_loss / initial_investment_value) * 100

            # Display the investment results
            st.markdown(
                f"You invested **\${initial_investment_value}** on {selected_date.strftime('%Y-%m-%d')}. "
                f"Today, your investment is worth **${current_investment_value}**."
            )

            if gain_loss > 0:
                st.write(f"You have a gain of ${gain_loss:.2f} ({gain_loss_percentage:.2f}%)")
            else:
                st.write(f"You have a loss of ${-gain_loss:.2f} ({gain_loss_percentage:.2f}%)")

            if movement_interpretation:
                st.header('Bitcoin Price Prediction for the Next 2 Days')

                day_1_prediction = movement_interpretation[0]
                day_2_prediction = movement_interpretation[1]

                day_1_emoji = "📈" if day_1_prediction == "increase" else "📉"
                day_2_emoji = "📈" if day_2_prediction == "increase" else "📉"

                st.write(f"Day 1 Prediction: {day_1_prediction.capitalize()} {day_1_emoji}")
                st.write(f"Day 2 Prediction: {day_2_prediction.capitalize()} {day_2_emoji}")

                # Suggestion based on prediction and current gain/loss
                if movement_interpretation[0] == "increase" or movement_interpretation[1] == "increase":
                    if gain_loss > 0:
                        suggestion = "sell"
                    else:
                        suggestion = "hold"
                elif movement_interpretation[0] == "decrease" and movement_interpretation[1] == "decrease":
                    suggestion = "buy"
                else:
                    suggestion = "hold"

                st.subheader(f"Suggestion: It may be a good time to {suggestion}.")
            else:
                st.write("Prediction data is not available.")



if __name__ == "__main__":
    main()