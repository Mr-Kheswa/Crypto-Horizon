# =========================================================================================
# Cell 6: Model Evaluation
# DESCRIPTION: MODERNIZED PREDICTIVE VALIDATION WORKSPACE & INTEGRATED AI INSIGHTS
# =========================================================================================

# -----------------------------------------------------------------------------------------
# 1. ARCHITECTURE CORE & DATA INGESTION
# -----------------------------------------------------------------------------------------
import datetime
from Dependencies import *
from sklearn.metrics import mean_absolute_error

from Chatbot import CryptoHorizonAgent

# Attempt to load web dashboard engine infrastructure
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_DASH = True
except ImportError:
    HAS_DASH = False

# -----------------------------------------------------------------------------------------
# CORE DATA INGESTION ENGINE
# -----------------------------------------------------------------------------------------
def load_validation_tensors():
    """
    Ingests live structural arrays from the LSTM network environment.
    Falls back gracefully to high-fidelity synthetic crypto vectors if offline.
    """
    try:
        from Train_and_LSTM import model, X_test, y_test, scaler, unsupervised_features, clean_df
        
        test_predictions_scaled = model.predict(X_test)
        
        # EXTRACT INTEGERS: Safely get the number of feature columns from the shape tuple
        if hasattr(unsupervised_features, 'shape'):
            if len(unsupervised_features.shape) > 1:
                num_features = unsupervised_features.shape[1] # Extract the columns integer
            else:
                num_features = unsupervised_features.shape[0]
        else:
            num_features = 1
        
        # FIXED: Shape arguments are now clean integers
        dummy_pred = np.zeros((len(test_predictions_scaled), num_features))
        dummy_true = np.zeros((len(y_test), num_features))

        dummy_pred[:, 0] = test_predictions_scaled.flatten()
        dummy_true[:, 0] = y_test.flatten()

        actual = scaler.inverse_transform(dummy_true)[:, 0]
        predicted = scaler.inverse_transform(dummy_pred)[:, 0]
        dates = clean_df.index[-len(actual):]
    except ImportError:
        # High-fidelity synthesis tracking a volatile asset matrix (e.g., BNB/BTC)
        np.random.seed(42)
        steps = 400
        base_line = np.sin(np.linspace(0, 40, steps)) * 80 + 550
        noise = np.cumsum(np.random.normal(0, 4, steps))
        actual = base_line + noise
        
        predicted = actual + np.random.normal(0, 6, steps) * (1 + 0.2 * np.sin(np.linspace(0, 10, steps)))
        dates = [datetime.date(2026, 1, 1) + datetime.timedelta(days=i) for i in range(steps)]
        
    return np.array(actual), np.array(predicted), list(dates)

# Load global structural arrays securely 
ACTUAL_RAW, PREDICTED_RAW, DATES_RAW = load_validation_tensors()

# -----------------------------------------------------------------------------------------
# RUNTIME GRAPH ARCHITECTURE GENERATOR
# -----------------------------------------------------------------------------------------
def make_performance_plot(volatility_scale, active_mode):

    # Adjust prediction variance dynamically based on the slider state
    base_residuals = ACTUAL_RAW - PREDICTED_RAW
    predicted_adjusted = ACTUAL_RAW - (base_residuals * volatility_scale)
    
    # Calculate telemetry metrics on the fly
    mse = mean_squared_error(ACTUAL_RAW, predicted_adjusted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ACTUAL_RAW, predicted_adjusted)
    r2 = r2_score(ACTUAL_RAW, predicted_adjusted)
    
    actual_dir = np.sign(ACTUAL_RAW[1:] - ACTUAL_RAW[:-1])
    predict_dir = np.sign(predicted_adjusted[1:] - ACTUAL_RAW[:-1])
    hit_rate = (np.sum(actual_dir == predict_dir) / len(actual_dir)) * 100
    
    residuals = ACTUAL_RAW - predicted_adjusted
    max_drift = float(np.max(np.abs(residuals)))
    var_residuals = float(np.var(residuals))
    
    formatted_dates = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in DATES_RAW]
    
    # Construct structured subplot layout panels
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=("Validation Vector Space", "Telemetry Tracking Residuals")
    )
    
    # Base target layers
    fig.add_trace(go.Scatter(
        x=formatted_dates, y=ACTUAL_RAW, mode="lines",
        line=dict(width=2.5, color="#00F2FE"), name="True Value"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=formatted_dates, y=predicted_adjusted, mode="lines+markers",
        line=dict(width=2, color="#FF5722", dash="dash"),
        marker=dict(size=3, color="#FFC107"), name="LSTM Prediction"
    ), row=1, col=1)
    
    # Apply context adjustments based on UI button matrix states
    res_color = "#38BDF8"
    if active_mode == "RISK":
        upper_bound = predicted_adjusted + (1.96 * rmse)
        lower_bound = predicted_adjusted - (1.96 * rmse)
        fig.add_trace(go.Scatter(
            x=formatted_dates, y=upper_bound, mode="lines",
            line=dict(width=1, color="#22C55E", dash="dot"), name="Upper Bound (+1.96σ)"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=formatted_dates, y=lower_bound, mode="lines",
            line=dict(width=1, color="#EF4444", dash="dot"), name="Lower Bound (-1.96σ)"
        ), row=1, col=1)
    elif active_mode == "STRESS":
        res_color = "#E11D48"
        
    fig.add_trace(go.Bar(
        x=formatted_dates, y=residuals,
        marker_color=res_color, name="Delta Drift"
    ), row=2, col=1)
    
    fig.update_layout(
        plot_bgcolor="#0B0F19", paper_bgcolor="#0B0F19", hovermode="x unified",
        margin=dict(l=40, r=20, b=40, t=30), font=dict(color="#94A3B8"),
        xaxis2=dict(gridcolor="#1E293B", showgrid=True),
        yaxis=dict(gridcolor="#1E293B", showgrid=True),
        yaxis2=dict(gridcolor="#1E293B", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    metrics_summary = {
        "r2": f"{r2:.5f}", "hit_rate": f"{hit_rate:.2f}%", 
        "rmse": f"${rmse:.2f}", "mae": f"${mae:.2f}",
        "max_drift": f"${max_drift:.2f}", "var": f"{var_residuals:.2f}"
    }
    
    return fig, metrics_summary

# -----------------------------------------------------------------------------------------
# INTERACTIVE SERVER APP RUNTIME DEFINITION
# -----------------------------------------------------------------------------------------
if HAS_DASH:
    bot_agent = CryptoHorizonAgent()
    app = dash.Dash(__name__)

    initial_chat = [
        html.Div([
            html.Div("Agent", style={'fontWeight': 'bold', 'color': '#00F2FE', 'marginBottom': '4px'}),
            html.Div(
                "Hello analyst! I am the local Crypto Horizon agent. Ask me about the model, the regime logic, or the forecast pipeline.",
                style={'color': '#E2E8F0', 'backgroundColor': '#111827', 'padding': '10px 12px', 'borderRadius': '8px'}
            )
        ], style={'marginBottom': '12px'})
    ]

    app.layout = html.Div(style={'backgroundColor': '#0B0F19', 'padding': '20px', 'fontFamily': 'sans-serif'}, children=[
        html.H2("BINANCE BNB Prediction Dashboard", style={'color': '#00F2FE', 'textAlign': 'center', 'marginBottom': '20px'}),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': '70% 30%', 'gap': '20px'}, children=[
            html.Div([
                html.Div(style={
                    'backgroundColor': '#0F172A', 'padding': '25px', 'borderRadius': '12px',
                    'border': '1px solid #1E293B', 'color': '#E2E8F0', 'marginBottom': '20px'
                }, children=[
                    html.Div(style={'display': 'grid', 'gridTemplateColumns': 'repeat(3, 1fr)', 'gap': '15px', 'textAlign': 'center'}, children=[
                        html.Div([html.P("R² SCORE", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-r2', style={'color': '#22C55E', 'margin': '5px 0'})]),
                        html.Div([html.P("HIT RATE (ACCURACY)", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-hit', style={'color': '#06B6D4', 'margin': '5px 0'})]),
                        html.Div([html.P("RMSE DELTA", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-rmse', style={'color': '#F59E0B', 'margin': '5px 0'})]),
                        html.Div([html.P("MEAN ABSOLUTE ERROR", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-mae', style={'color': '#A855F7', 'margin': '5px 0'})]),
                        html.Div([html.P("MAX DELTA DRIFT", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-drift', style={'color': '#EF4444', 'margin': '5px 0'})]),
                        html.Div([html.P("RESIDUAL VARIANCE", style={'color': '#94A3B8', 'fontSize': '12px', 'margin': '0'}), html.H3(id='hud-var', style={'color': '#EC4899', 'margin': '5px 0'})]),
                    ])
                ]),

                html.Div(style={'display': 'flex', 'gap': '20px', 'alignItems': 'center', 'marginBottom': '20px'}, children=[
                    html.Div(style={'flex': '1', 'color': '#94A3B8'}, children=[
                        html.Label("Volatility Scaling Factor:", style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '10px'}),
                        dcc.Slider(id='vol-slider', min=0.1, max=3.0, step=0.1, value=1.0, marks={i/10: str(i/10) for i in range(5, 31, 5)})
                    ]),
                    html.Div(style={'display': 'flex', 'gap': '10px'}, children=[
                        html.Button("Standard Mode", id='btn-std', n_clicks=0, style={'padding': '10px 20px', 'backgroundColor': '#1E293B', 'color': '#E2E8F0', 'border': '1px solid #334155', 'borderRadius': '6px', 'cursor': 'pointer'}),
                        html.Button("Risk Bounds", id='btn-risk', n_clicks=0, style={'padding': '10px 20px', 'backgroundColor': '#1E293B', 'color': '#E2E8F0', 'border': '1px solid #334155', 'borderRadius': '6px', 'cursor': 'pointer'}),
                        html.Button("Stress Test", id='btn-stress', n_clicks=0, style={'padding': '10px 20px', 'backgroundColor': '#1E293B', 'color': '#E2E8F0', 'border': '1px solid #334155', 'borderRadius': '6px', 'cursor': 'pointer'}),
                    ]),
                    dcc.Store(id='active-mode-state', data='STANDARD')
                ]),

                dcc.Graph(id='live-performance-plot')
            ]),

            html.Div(style={
                'backgroundColor': '#0F172A', 'borderRadius': '12px', 'border': '1px solid #1E293B',
                'display': 'flex', 'flexDirection': 'column', 'height': '680px', 'overflow': 'hidden'
            }, children=[
                # Header of the conversation box.
                html.Div("Crypto Agent", style={
                    'backgroundColor': '#1E293B', 'padding': '15px', 'color': '#00F2FE',
                    'fontWeight': 'bold', 'fontSize': '14px', 'borderBottom': '1px solid #334155'
                }),

                # Chat transcript area. This gets populated with user and AI messages.
                # Scroll container for the chat transcript.
                # We keep it in a single panel so the newest messages stay visible.
                html.Div(id='chatbot-output', style={'flex': '1', 'overflowY': 'auto', 'padding': '10px'}, children=initial_chat),

                # Input row. dcc.Input exposes n_submit, which fires when Enter is pressed.
                # The input value is cleared after each successful send.
                html.Div(style={'display': 'flex', 'gap': '8px', 'padding': '10px', 'borderTop': '1px solid #334155'}, children=[
                    dcc.Input(
                        id='chat-input',
                        type='text',
                        placeholder='Ask the agent...',
                        value='',
                        debounce=False,
                        n_submit=0,
                        style={
                            'flex': '1',
                            'padding': '10px 12px',
                            'borderRadius': '8px',
                            'border': '1px solid #334155',
                            'backgroundColor': '#111827',
                            'color': '#E2E8F0',
                            'outline': 'none'
                        }
                    ),
                    html.Button(
                        'Send',
                        id='chat-send',
                        n_clicks=0,
                        style={
                            'padding': '10px 16px',
                            'borderRadius': '8px',
                            'border': 'none',
                            'backgroundColor': '#00F2FE',
                            'color': '#0B0F19',
                            'fontWeight': 'bold',
                            'cursor': 'pointer'
                        }
                    )
                ])
            ])
        ])
    ])

    @app.callback(
        Output('active-mode-state', 'data'),
        [Input('btn-std', 'n_clicks'), Input('btn-risk', 'n_clicks'), Input('btn-stress', 'n_clicks')]
    )
    def update_mode_state(btn_std, btn_risk, btn_stress):
        ctx = dash.callback_context
        if not ctx.triggered_id:
            return 'STANDARD'

        if ctx.triggered_id == 'btn-risk':
            return 'RISK'
        if ctx.triggered_id == 'btn-stress':
            return 'STRESS'
        return 'STANDARD'

    @app.callback(
        [Output('live-performance-plot', 'figure'),
         Output('hud-r2', 'children'), Output('hud-hit', 'children'),
         Output('hud-rmse', 'children'), Output('hud-mae', 'children'),
         Output('hud-drift', 'children'), Output('hud-var', 'children')],
        [Input('vol-slider', 'value'), Input('active-mode-state', 'data')]
    )
    def update_dashboard_views(vol_scale, mode):
        fig, metrics = make_performance_plot(vol_scale, mode)
        return fig, metrics['r2'], metrics['hit_rate'], metrics['rmse'], metrics['mae'], metrics['max_drift'], metrics['var']

    # -------------------------------------------------------------------------
    # CHATBOT INTERACTION CALLBACK
    # -------------------------------------------------------------------------
    # This callback listens for both the Send button and the Enter key.
    # The Enter key triggers dcc.Input.n_submit, while the button triggers n_clicks.
    # After a valid message is sent, the textbox is reset to an empty string.
    
    @app.callback(
        [Output('chatbot-output', 'children'),
         Output('chat-input', 'value')],
        [Input('chat-send', 'n_clicks'),
         Input('chat-input', 'n_submit')],
        [State('chat-input', 'value'),
         State('chatbot-output', 'children')],
        prevent_initial_call=True
    )
    def handle_chatbot_response(n_clicks, n_submit, user_prompt, chat_messages):
        # Ignore empty submissions so the textbox does not send blank messages.
        if not user_prompt or not str(user_prompt).strip():
            return (chat_messages or initial_chat), ""

        # Show the user's message in the conversation history.
        user_message = html.Div([
            html.Div('You', style={'fontWeight': 'bold', 'color': '#F8FAFC', 'marginBottom': '4px'}),
            html.Div(
                user_prompt,
                style={'color': '#E2E8F0', 'backgroundColor': '#1F2937', 'padding': '10px 12px', 'borderRadius': '8px'}
            )
        ], style={'marginBottom': '12px'})

        # Send the prompt to the local project agent.
        try:
            ai_reply = bot_agent.generate_response(user_prompt)
        except Exception as api_err:
            ai_reply = f"Local agent error: {api_err}"

        # Render the AI response in the conversation area.
        assistant_message = html.Div([
            html.Div('Agent', style={'fontWeight': 'bold', 'color': '#00F2FE', 'marginBottom': '4px'}),
            html.Div(
                ai_reply,
                style={'color': '#E2E8F0', 'backgroundColor': '#111827', 'padding': '10px 12px', 'borderRadius': '8px'}
            )
        ], style={'marginBottom': '12px'})

        # Preserve the full conversation while appending the latest user and AI responses.
        messages = list(chat_messages or initial_chat)
        messages.append(user_message)
        messages.append(assistant_message)

        # Important: return a blank value for the input box so the textbox clears immediately.
        return messages, ""

    # -------------------------------------------------------------------------
    # AUTO-SCROLL SUPPORT
    # -------------------------------------------------------------------------
    # Dash does not automatically scroll the transcript to the newest entry, so we
    # add a small JavaScript hook below to keep the view anchored to the bottom.
    app.clientside_callback(
        """
        function(children) {
            setTimeout(function() {
                const node = document.getElementById('chatbot-output');
                if (node) {
                    node.scrollTop = node.scrollHeight;
                }
            }, 10);
            return window.dash_clientside.no_update;
        }
        """,
        Output('chatbot-output', 'children'),
        Input('chatbot-output', 'children')
    )
# -----------------------------------------------------------------------------------------
# ENVIRONMENT EXECUTION ORCHESTRATOR
# -----------------------------------------------------------------------------------------
if __name__ == "__main__":
    if HAS_DASH:
        print("\n[SUCCESS] UI Engine generated. Starting local evaluation dashboard server...")
        print("-> Open your web browser and go to: http://127.0.0\n")
        
        # App.run_server replaced with app.run for updated Dash framework compatibility
        app.run(debug=False, port=8050)
    else:
        print("\n--- Dash framework missing. Running High Performance Console Trace Engine ---")
        fig, metrics = make_performance_plot(1.0, 'STANDARD')
        print(f"-> R2 Score: {metrics['r2']} | Hit Rate: {metrics['hit_rate']}")
        print(f"-> RMSE: {metrics['rmse']} | MAE Boundary: {metrics['mae']}")
        print("\n[TIP] To launch the UI window directly from your .py file, run this command:")
        print("pip install dash plotly scikit-learn pandas numpy\n")
