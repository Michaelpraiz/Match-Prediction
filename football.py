import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import pandas as pd

df=pd.read_csv("matche_stats.csv", index_col=0)

df['xG_diff'] = df['xG'] - df['xGA']
df['Goal_diff'] = df['GF'] - df['GA']
df['Shot_accuracy'] = df['SoT'] / df['Sh']
df['Points'] = df['Result'].map({'W': 3, 'D': 1, 'L': 0})

# Load the trained model
with open('result-scores2.pkl', 'rb') as file:
    rfc, rfr = joblib.load(file)        

# Streamlit app layout
# Set page title
st.set_page_config(page_title="Football Match Stats and Predictions", layout="wide")
st.title("Football Match Statistics and Predictions")

# Display field for team data in the sidebar
with st.sidebar:
    team = st.selectbox('Select Team 1', df["Team"].sort_values().unique(), index=None)
    team2 = st.selectbox('Select Team 2', df["Team"].sort_values().unique(), index=None)
    show_data = st.button("Show Team Data")
    predict_score= st.button("Predict Outcome")

container=st.container()
data1, data2 = container.columns(2)

if show_data:
    with data1:
        # Display team data
        st.write("Team 1 Data:")
        st.write(df.loc[df["Team"]== team])

        # Calculate wins, losses, and draws
        wins = len(df.loc[df["Team"]== team, "Result"][df.loc[df["Team"]== team, "Result"] == 'W'])
        losses = len(df.loc[df["Team"]== team, "Result"][df.loc[df["Team"]== team, "Result"] == 'L'])
        draws = len(df.loc[df["Team"]== team, "Result"][df.loc[df["Team"]== team, "Result"] == 'D'])

        # Create a dataframe for the pie chart
        pie_df = pd.DataFrame({
            "Result": ["Wins", "Losses", "Draws"],
            "Count": [wins, losses, draws]
        })

        # Create the pie chart
        fig = px.pie(pie_df, names="Result", values="Count", color="Result", hole=0.5,
                     color_discrete_map={"Wins": "#3498db", "Losses": "#e74c3c", "Draws": "#f39c12"})

        # Update the layout
        fig.update_layout(title_text=f"Result Distribution for {team}", title_x=0, 
                        width=600, height=600, margin=dict(t=50, b=10, l=10, r=10))

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

        # Create a line chart
        fig = px.line(df.loc[df["Team"].isin([team, team2])], x="Round", y="Poss", color="Team",
                       title=f"Possession for {team} and {team2}", markers=True, color_discrete_map={team: "#3498db", team2: "red"})

        # Update the chart layout
        fig.update_traces(marker=dict(size=7.5, color="white", symbol="star"))
        fig.update_layout(showlegend=True, title_x=0, title_y=0.95, title_font_size=15)

        # Display the chart in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        # Filter the data for the two teams
        team_data = df[df['Team'].isin([team, team2])]

        # Create the scatter plot
        fig = px.scatter(team_data, 
                        x='Sh', 
                        y='SoT', 
                        size='Dist', 
                        color='Team',
                        hover_data=['Round', 'Opponent', 'Result'],
                        title='Shots vs Shots on Target, with Average Shot Distance',
                        labels={'Sh': 'Total Shots', 'SoT': 'Shots on Target', 'Dist': 'Avg. Shot Distance'},
                        size_max=20)

        # Update layout for better readability
        fig.update_layout(
            xaxis_title='Total Shots',
            yaxis_title='Shots on Target',
            legend_title='Team'
        )

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)

        # Advanced  statistics
        st.write("")
        st.write("")
        st.write(f"Advanced Metrics: :green[{team}] VS :red[{team2}]")
        st.write("")
        st.write(f"xG Difference: :green[{df[df['Team'] == team]['xG_diff'].mean():.2f}] VS :red[{df[df['Team'] == team2]['xG_diff'].mean():.2f}]")
        st.write(f"Goal Difference: :green[{df[df['Team'] == team]['Goal_diff'].mean():.2f}] VS :red[{df[df['Team'] == team2]['Goal_diff'].mean():.2f}]")
        st.write(f"Shot Accuracy: :green[{df[df['Team'] == team]['Shot_accuracy'].mean():.2%}] VS :red[{df[df['Team'] == team2]['Shot_accuracy'].mean():.2%}]")
        st.write(f"Average Points per Game: :green[{df[df['Team'] == team]['Points'].mean():.2f}] VS :red[{df[df['Team'] == team2]['Points'].mean():.2f}]")

    with data2:
        # Display team data
        st.write("Team 2 Data:")
        st.write(df.loc[df["Team"]== team2])

        # Calculate wins, losses, and draws
        wins = len(df.loc[df["Team"]== team2, "Result"][df.loc[df["Team"]== team2, "Result"] == 'W'])
        losses = len(df.loc[df["Team"]== team2, "Result"][df.loc[df["Team"]== team2, "Result"] == 'L'])
        draws = len(df.loc[df["Team"]== team2, "Result"][df.loc[df["Team"]== team2, "Result"] == 'D'])

        # Create a dataframe for the pie chart
        pie_df = pd.DataFrame({
            "Result": ["Wins", "Losses", "Draws"],
            "Count": [wins, losses, draws]
        })

        # Create the pie chart
        fig = px.pie(pie_df, names="Result", values="Count", color="Result", hole=0.5,
                    color_discrete_map={"Wins": "#3498db", "Losses": "#e74c3c", "Draws": "#f39c12"})

        # Update the layout
        fig.update_layout(title_text=f"Result Distribution for {team2}", title_x=0, 
                        width=600, height=600, margin=dict(t=50, b=10, l=10, r=10))

        # Show the plot
        st.plotly_chart(fig, use_container_width=True)

        # Filter the data for the two teams
        team_data = df[df['Team'].isin([team, team2])]

        # Create a bar chart using plotly.express
        fig = px.bar(team_data, x='Result', y='SoT', color='Team', barmode='group',
                    title=f"Shot on Target (SoT) for each match outcome between {team} and {team2}",
                    labels={'SoT': 'Shot on Target'})

        # Display the plot
        st.plotly_chart(fig)

        # Get recent results for both teams
        recent_results_team1 = df[df['Team'] == team].sort_values('Date')
        recent_results_team2 = df[df['Team'] == team2].sort_values('Date')

        # Combine the recent results into a single DataFrame
        recent_results_combined = pd.concat([recent_results_team1.assign(Team=team), 
                                            recent_results_team2.assign(Team=team2)])

        # Create a line chart to visualize recent form for both teams
        fig = px.line(recent_results_combined, x='Round', y='Points', color='Team', markers=True,
                    title=f"Recent Form of {team} and {team2}",
                    labels={'Points': 'Points', 'Round': 'Round'})
    
        # Update the chart layout
        fig.update_traces(marker=dict(size=7.5, color="white", symbol="star"))
        fig.update_layout(showlegend=True, title_x=0, title_y=0.95, title_font_size=15)

        st.plotly_chart(fig, use_container_width=True)

        # Create a bar chart using plotly.express
        fig = px.bar(team_data.groupby('Team')[['GF', 'GA']].sum().reset_index(), 
                    x=['GF', 'GA'], y='Team', barmode='group',
                    title=f"Goals For and Against by Team for {team} and {team2}",
                    labels={'value': 'Goals', 'variable': 'Type'}, height=300,  color_discrete_map={'GF': "#3498db", 'GA': "red"})

        # Display the plot
        st.plotly_chart(fig)

if predict_score:
    # Convert 'Round' to numeric if it's not already
    if 'Round' in df.columns:
        df['Round'] = pd.to_numeric(df['Round'], errors='coerce')
    
    # Get average stats for each team, only for numeric columns
    team1_stats = df.loc[df["Team"] == team].select_dtypes(include=[np.number]).mean()
    team2_stats = df.loc[df["Team"] == team2].select_dtypes(include=[np.number]).mean()

    # Prepare input data
    features = ['xGA', 'xG', 'SoT', 'Round', 'Poss', 'Sh', 'Result']
    
    # Check which features are actually available
    available_features = [feature for feature in features if feature in team1_stats.index and feature in team2_stats.index]

    # Prepare match data using only available features
    match_data = np.array([team1_stats[feature] - team2_stats[feature] for feature in available_features])
    match_data = match_data.reshape(1, -1)

    # Make prediction
    prediction = rfc.predict(match_data)
    probabilities = rfc.predict_proba(match_data)

    # Make prediction
    score_prediction = rfr.predict(match_data)   

    # Display predictions
    st.subheader("Prediction :red[Results]")

    def get_head_to_head(team, team2):
        h2h = df[(df['Team'] == team) & (df['Opponent'] == team2) | 
                (df['Team'] == team2) & (df['Opponent'] == team)]
        return h2h

    # Add this to the prediction section
    st.subheader("Head-to-Head History")
    h2h = get_head_to_head(team, team2)
    st.write(h2h[['Team', 'Opponent', 'Result', 'GF', 'GA']])

    # Create a bar chart for head-to-head results
    h2h_results = h2h['Result'].value_counts().reset_index()
    h2h_results.columns = ['Result', 'Count']  # Rename columns

    # Create the bar chart using the DataFrame with customized colors
    fig = px.bar(h2h_results, x='Result', y='Count', color="Result",
                color_discrete_map={"W": "#1f77b4", "L": "#ff7f0e", "D": "#2ca02c"},
                title=f"Head-to-Head Results: {team} vs {team2}")
    st.plotly_chart(fig)

    # Result prediction
    result_map = {0: ":red[Lose]", 1: ":blue[Draw]", 2: ":green[Win]"}
    (f"{team} Is Predicted To {result_map[prediction[0]]} Against {team2}")

    col1, col2, col3= st.columns(3)
    with col1:
        st.metric('LOSS', f'{probabilities[0][0]:.1%}')
    with col2:
        st.metric('DRAW', f'{probabilities[0][1]:.1%}')
    with col3:
        st.metric('WIN', f'{probabilities[0][2]:.1%}') 

    # Display probabilities inside a styled box
    loss_prob = (probabilities[0][0]*100)
    draw_prob = (probabilities[0][1]*100)
    win_prob = (probabilities[0][2]*100)

    score1 = int(score_prediction[0][-1])
    score2 = int(score_prediction[-1][0])

    def adjust_scores(loss_prob, draw_prob, win_prob, score1, score2):
        if draw_prob > win_prob and draw_prob > loss_prob:
            # Force scores to be equal
            score1 = score2
        elif win_prob > draw_prob and win_prob > loss_prob:
            # Force score1 to be greater than score2
            score1 = score2 + 1  # Ensure score1 is at least one more than score2
        elif loss_prob > win_prob and loss_prob > draw_prob:
            # Force score1 to be less than score2
            score1 = score2 - 1  # Ensure score1 is at least one less than score2
        
        return score1, score2

    def enforce_min_scores(win_prob, loss_prob, draw_prob, score1, score2):
        if win_prob > 50:
            score1 = max(score1, 2)  # Ensure score1 is at least 2
        if loss_prob >= 35 or draw_prob > 35:
            score2 = max(score2, 1)  # Ensure score2 is at least 1
        
        return score1, score2

    # Call the functions to adjust and enforce minimum scores
    score1, score2 = adjust_scores(loss_prob, draw_prob, win_prob, score1, score2)
    score1, score2 = enforce_min_scores(win_prob, loss_prob, draw_prob, score1, score2)

    # if draw is below 40 score1 and score2 should be both 0
    if draw_prob < 40 and draw_prob > win_prob and draw_prob > loss_prob:
        score1 = 0
        score2 = 0

    # Score prediction
    # Display the styled score prediction
    st.markdown(
        f"""
        <div style="text-align: left; margin-bottom: 20px;">
            <h4 style="color: #e74c3c; font-size: 18px;">Predicted Score</h4>
            <p style="color: #ffff; font-size: 16px;">
                {team} <span style="color: #3498db;">{score1}</span> - <span style="color: #e74c3c;">{score2}</span> {team2} 
                <span title="Match score probability is 80% and might differ from the actual result." style="color: #888; cursor: help;">(?)</span>
            </p>
        </div>
        """, unsafe_allow_html=True
    )
    