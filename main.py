import streamlit as st, pandas as pd, numpy as np, calendar
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

def validate_date(date_obj):
    """ 
    Automatically corrects and returns a valid datetime object near the provided one if the original date is invalid.
    """
    year, month, day = date_obj.year, date_obj.month, date_obj.day

    # Adjust the month and year if they are out of bounds
    if month < 1:
        month = 1
    elif month > 12:
        month = 12

    # Get the last day of the given month and year
    last_day_of_month = calendar.monthrange(year, month)[1]

    # Adjust the day if it's out of bounds
    if day < 1:
        day = 1
    elif day > last_day_of_month:
        day = last_day_of_month

    return datetime(year, month, day)

# Function to generate simulated restaurant data
def generate_restaurant_data(start_date, end_date):
    """
    Generates simulated restaurant data including dates, foot traffic, sales, and volumes of popular food items.
    
    Assumptions are based on:
    - Average revenue for a new restaurant: $111,860.70 annually 
      (Source: UpMenu, 2023, URL: https://www.upmenu.com)
    - Complex correlation between foot traffic and sales 
      (Source: MarketScale, 2022, URL: https://marketscale.com/industries/food-and-beverage/restaurant-sales-are-up-and-traffic-is-down-but-why-is-that/)
    - Factors affecting revenue include location, menu, customer experience, etc. 
      (Source: BNG Payments, URL: https://bngpayments.net)
    """
    
    # Validate the dates
    start_date = validate_date(start_date)
    end_date = validate_date(end_date)

    # Create df
    date_range = pd.date_range(start=start_date, end=end_date)
    data = {"Date": [], "Foot Traffic": [], "Sales": [], 
            "Popular Item 1 Volume": [], "Popular Item 2 Volume": [], "Popular Item 3 Volume": []}
    
    # Set averages based on industry averages
    annual_revenue = 111860.70
    daily_revenue_average = annual_revenue / 365
    sales_variation = 0.2
    foot_traffic_variation = 0.15

    # Build data
    for date in date_range:
        is_weekend = date.weekday() >= 5
        daily_sales = daily_revenue_average * (1 + np.random.uniform(-sales_variation, sales_variation))
        daily_sales *= 1.2 if is_weekend else 1.0
        foot_traffic = np.random.normal(100, 10) * (daily_sales / daily_revenue_average)
        foot_traffic *= (1 + np.random.uniform(-foot_traffic_variation, foot_traffic_variation))
        foot_traffic = max(10, int(foot_traffic))

        item1_volume = daily_sales * 0.1
        item2_volume = daily_sales * 0.07
        item3_volume = daily_sales * 0.05

        data["Date"].append(date)
        data["Foot Traffic"].append(foot_traffic)
        data["Sales"].append(daily_sales)
        data["Popular Item 1 Volume"].append(item1_volume)
        data["Popular Item 2 Volume"].append(item2_volume)
        data["Popular Item 3 Volume"].append(item3_volume)

    return pd.DataFrame(data)

def calculate_last_7_days_sales(df):
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=7)
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]['Sales'].sum()

def calculate_last_7_days_visitors(df):
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=7)
    return df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]['Foot Traffic'].sum()

def calculate_average_sale_value_last_7_days(df):
    end_date = df['Date'].max()
    start_date = end_date - timedelta(days=7)
    last_7_days_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    total_sales = last_7_days_df['Sales'].sum()
    total_transactions = last_7_days_df['Sales'].count()
    return total_sales / total_transactions if total_transactions else 0

def calculate_returning_customer_rate(df):
    # This function needs specific customer data to be implemented correctly.
    # As a placeholder, it returns 0.0.
    return np.random.randint(50, 95)

# Define a consistent color palette for the dashboard
color_palette = {'background': '#f4f4f8', 'text': '#333333', 'bar': '#ff851b', 'line': '#0074D9', 'area': '#2ECC40', 'scatter': '#85144b'}

# Update the layout of each plot with consistent styling
def format_plot(fig, title, xaxis_title, yaxis_title, color):
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        # plot_bgcolor='rgba(233, 236, 239, 0.5)',  # Light gray background for plot area
        # paper_bgcolor='rgba(233, 236, 239, 1)',   # Slightly darker gray for the surrounding area
        # font=dict(color="black", size=12),
        margin=dict(t=60, l=50, r=50, b=50),  # Adjust margins to prevent clipping
        # title_font=dict(size=16, color=color),  # Stylish title
    )
    return fig

# Apply consistent styling to KPI annotations
def style_kpi_annotation(text, x, y):
    return dict(
        text=text, xref="paper", yref="paper", x=x, y=y, 
        xanchor='left', yanchor='bottom',
        showarrow=False, align="left",
        bgcolor="rgba(255, 255, 255, 0.9)",
        font=dict(family="Arial, sans-serif", size=14, color=color_palette['text'])
    )

def get_predictions(data, col_to_predict, days_to_forecast):
    # Convert 'Date' to datetime and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Select the 'Sales' column
    sales_data = data[col_to_predict]

    # Fit the Holt-Winters model with identified best parameters
    final_model = ExponentialSmoothing(sales_data, trend='mul', seasonal='mul', seasonal_periods=7, use_boxcox=True)    
    fitted_final_model = final_model.fit(optimized=True)

    # Forecast the next 60 days
    forecast = fitted_final_model.forecast(days_to_forecast)

    # Prepare the forecast dataframe
    forecast_dates = pd.date_range(start=sales_data.index[-1], periods=days_to_forecast)
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast.values})
    forecast_df['Forecast'] = [x+np.random.randint(15,30) for x in forecast_df['Forecast']]
    forecast_df.set_index('Date', inplace=True)

    # Combine actual and forecasted data for plotting
    combined_data = pd.concat([sales_data, forecast_df], axis=1)

    plot_months = round(round(days_to_forecast/30) * 2)

    # Filter the combined data for the last 4 months and the forecast
    plot_data = combined_data.last(f'{plot_months}M')

    return plot_data, forecast_df


# Streamlit app initialization
def main():
    st.set_page_config(page_title="Real-Time Data Science Dashboard", page_icon="✅", layout="wide")
    st.title("Sports Bar KPIs and Forecasts")
    
    # Declare sidebar components
    starting_date = st.sidebar.date_input(label="Start Date (through Today)", value=datetime(2019, 1, 1))
    forecast_days = st.sidebar.number_input(label="How many days to forecast ahead (max 120)", value=60, placeholder="Enter a number of days to predict...")
    gen_data = st.sidebar.button("Generate data")
    train_model = st.sidebar.checkbox("Train a machine learning model with the data")

    end_date = datetime.now().date()
    restaurant_data = generate_restaurant_data(starting_date, end_date)

    if gen_data:
    
        # Creating plots
        #line_graph_sales = go.Figure(data=[go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Sales'], mode='lines+markers')])
        #bar_chart_traffic = go.Figure(data=[go.Bar(x=restaurant_data['Date'], y=restaurant_data['Foot Traffic'])])
        scatter_sales_traffic = go.Figure(data=[go.Scatter(x=restaurant_data['Foot Traffic'], y=restaurant_data['Sales'], mode='markers')])
        window_size = 14  # Adjust the window size for smoother trends
        line_graph_sales = go.Figure(data=[
            go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Sales'], mode='lines+markers'),
            go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Sales'].rolling(window=window_size).mean(), mode='lines', name='Trendline', line=dict(color='red'))
        ])
        line_graph_sales.update_layout(title='Sales Over Time with Trendline',
                                    xaxis_title='Date',
                                    yaxis_title='Sales')
        
        bar_chart_traffic = go.Figure(data=[
            go.Bar(x=restaurant_data['Date'], y=restaurant_data['Foot Traffic']),
            go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Foot Traffic'].rolling(window=window_size).mean(), mode='lines', name='Trendline', line=dict(color='red'))
        ])
        bar_chart_traffic.update_layout(title='Foot Traffic Over Time with Trendline',
                                        xaxis_title='Date',
                                        yaxis_title='Foot Traffic')
        
        # Track cumulative sales
        restaurant_data['Cumulative Sales'] = restaurant_data['Sales'].cumsum()
        # area_chart_cumulative_sales = go.Figure(data=[go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Cumulative Sales'], fill='tozeroy')])

        # Sales by year plot
        restaurant_data['Year'] = restaurant_data['Date'].dt.year
        # Group data by year and calculate cumulative sales for each year
        yearly_cumulative_sales = restaurant_data.groupby('Year')['Cumulative Sales'].max()
        bar_chart_cumulative_sales = go.Figure(data=[go.Bar(x=yearly_cumulative_sales.index, y=yearly_cumulative_sales)])
        bar_chart_cumulative_sales = go.Figure(data=[
            go.Bar(x=yearly_cumulative_sales.index, y=yearly_cumulative_sales),
            go.Scatter(x=yearly_cumulative_sales.index, y=yearly_cumulative_sales, mode='lines', name='Trendline', line=dict(color='red'))
        ])

        # Average sale by day
        restaurant_data['Day'] = restaurant_data['Date'].dt.day_name()
        # Group data by day and calculate average sales for each day
        average_sales_by_day = restaurant_data.groupby('Day')['Sales'].mean()
        days_order = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        # Horizontal bar plot with ordered days
        horizontal_bar_chart_average_sales = go.Figure(data=[go.Bar(y=average_sales_by_day.index, x=average_sales_by_day, orientation='h')])
        horizontal_bar_chart_average_sales.update_layout(title='Average Sales Per Day',
                                                        xaxis_title='Average Sales',
                                                        yaxis_title='Day',
                                                        yaxis=dict(categoryorder='array', categoryarray=days_order),
                                                        height=400, width=500)
        
        # Group data by day and calculate average foot traffic for each day
        average_foot_traffic_by_day = restaurant_data.groupby('Day')['Foot Traffic'].mean()
        # Horizontal bar plot with ordered days
        horizontal_bar_chart_average_foot_traffic = go.Figure(data=[go.Bar(y=average_foot_traffic_by_day.index, x=average_foot_traffic_by_day, orientation='h')])
        horizontal_bar_chart_average_foot_traffic.update_layout(title='Average Foot Traffic Per Day',
                                                                xaxis_title='Average Foot Traffic',
                                                                yaxis_title='Day',
                                                                yaxis=dict(categoryorder='array', categoryarray=days_order),
                                                                height=400, width=500)

     
        # Summing up the sales volumes for each popular item
        item_sales = restaurant_data[['Popular Item 1 Volume', 'Popular Item 2 Volume', 'Popular Item 3 Volume']].sum()
        # Creating a pie chart using Plotly
        item_sale_proportions = go.Figure(data=[go.Pie(labels=item_sales.index, values=item_sales.values, hole=.3)])
        item_sale_proportions.update_layout(title_text='Proportions of Sales for Each Item')
        
        # Plotting the sales of three food items over time
        multi_line_food_sales = go.Figure()
        multi_line_food_sales.add_trace(go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Popular Item 1 Volume'], mode='lines', name='Item 1'))
        multi_line_food_sales.add_trace(go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Popular Item 2 Volume'], mode='lines', name='Item 2'))
        multi_line_food_sales.add_trace(go.Scatter(x=restaurant_data['Date'], y=restaurant_data['Popular Item 3 Volume'], mode='lines', name='Item 3'))

        # Apply the format to all plots
        line_graph_sales = format_plot(line_graph_sales, 'Sales Over Time', 'Date', 'Sales', 'blue')
        bar_chart_traffic = format_plot(bar_chart_traffic, 'Foot Traffic Over Time', 'Date', 'Foot Traffic', 'orange')
        scatter_sales_traffic = format_plot(scatter_sales_traffic, 'Sales vs Foot Traffic', 'Foot Traffic', 'Sales', 'green')
        bar_chart_cumulative_sales = format_plot(bar_chart_cumulative_sales, 'Sales by Year', 'Year', 'Sales', 'red')
        # area_chart_cumulative_sales = format_plot(area_chart_cumulative_sales, 'Cumulative Sales Over Time', 'Date', 'Cumulative Sales', 'red')
        multi_line_food_sales = format_plot(multi_line_food_sales, 'Individual Food Sales Over Time', 'Date', 'Sales', 'violet')

        # Calculate KPIs
        last_7_days_sales = round(calculate_last_7_days_sales(restaurant_data))
        last_7_days_visitors = round(calculate_last_7_days_visitors(restaurant_data))
        average_sale_value_last_7_days = round(calculate_average_sale_value_last_7_days(restaurant_data))
        returning_customer_rate = round(calculate_returning_customer_rate(restaurant_data))

        # create three columns
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric(label="Last 7 Days Sales", value=last_7_days_sales, delta=np.random.randint(-50,50))
        kpi2.metric(label="Last 7 Days Visitors", value=last_7_days_visitors, delta=np.random.randint(-10,10))
        kpi3.metric(label="Average Sale Value (Last 7 Days)", value=average_sale_value_last_7_days, delta=np.random.randint(-20,20))
        kpi4.metric(label="Returning Customer Rate (Last 30 Days)", value=returning_customer_rate, delta=returning_customer_rate-np.random.randint(-10,10))

        # Create container for charts
        placeholder = st.empty()
        with placeholder.container():
            plot1, plot2 = st.columns(2)
            with plot1:
                st.markdown("### Sales Over Time")
                st.write(line_graph_sales)
            with plot2:                
                st.markdown("### Foot Traffic Over Time")
                st.write(bar_chart_traffic)
            
            plot3, plot4 = st.columns(2)
            with plot3:
                st.markdown("### Sales vs Traffic")
                st.write(scatter_sales_traffic)
            with plot4:
                st.markdown("### Sales by Year")
                st.write(bar_chart_cumulative_sales)
            
            plot5,plot6 = st.columns(2)
            with plot5:
                st.markdown("### Proportion of Sales Total by Item")
                st.write(item_sale_proportions)
            with plot6:
                st.markdown("### Item Sales Over Time")
                st.write(multi_line_food_sales)

            plot6,plot7 = st.columns(2)
            with plot6:
                st.markdown("### Average Sales by Day")
                st.write(horizontal_bar_chart_average_sales)
            with plot7:
                st.markdown("### Average Foot Traffic by Day")
                st.write(horizontal_bar_chart_average_foot_traffic)

            st.markdown("### Detailed Restaurant Data View")
            st.dataframe(restaurant_data.sort_values(by='Date', ascending=False), use_container_width=True, hide_index=True)

        if train_model:
            # Train model + get predictions
            sales_plot_data, sales_forecasts = get_predictions(restaurant_data.copy(), "Sales", forecast_days)
            traffic_plot_data, traffic_forecasts = get_predictions(restaurant_data.copy(), "Foot Traffic", forecast_days)
            
            # Create the Plotly chart for sales
            sales_forecast_plot = go.Figure()
            # Plot actual sales
            sales_forecast_plot.add_trace(go.Scatter(x=sales_plot_data.index, y=sales_plot_data['Sales'], mode='lines', name='Actual Sales'))
            # Plot forecasted sales
            sales_forecast_plot.add_trace(go.Scatter(x=sales_forecasts.index, y=sales_forecasts['Forecast'], mode='lines', name='Forecasted Sales'))
            # Update layout
            sales_forecast_plot.update_layout(title='Actual Sales and Sales Forecast', xaxis_title='Date', yaxis_title='Sales', template='plotly_dark')

            # Create the Plotly chart for traffic
            traffic_forecast_plot = go.Figure()
            # Plot actual sales
            traffic_forecast_plot.add_trace(go.Scatter(x=traffic_plot_data.index, y=traffic_plot_data['Foot Traffic'], mode='lines', name='Actual Traffic'))
            # Plot forecasted sales
            traffic_forecast_plot.add_trace(go.Scatter(x=traffic_forecasts.index, y=traffic_forecasts['Forecast'], mode='lines', name='Forecasted Traffic'))
            # Update layout
            traffic_forecast_plot.update_layout(title='Actual Traffic and Traffic Forecast', xaxis_title='Date', yaxis_title='Foot Traffic', template='plotly_dark')

            plot7, plot8 = st.columns(2)
            with plot7:
                st.markdown("### Sales Forecast")
                st.write(sales_forecast_plot)
            with plot8:
                st.markdown("### Traffic Forecast")
                st.write(traffic_forecast_plot)

    
    else:
        st.write("Please generate data to view the app")

main()