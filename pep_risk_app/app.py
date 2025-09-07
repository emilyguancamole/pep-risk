# app.py
# -----------------------------------------------------------------------------
# Author:             Converted from R Shiny app by Albert Kuo
#
# Plotly Dash app for PEP risk prediction using gradient boosting

##### THIS IS THE LAST VERSION WITH DOT PLOTS AND LIME GRAPH. NEXT WILL BE A BAR GRAPH FOR DISPLAY ONLY #####

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "PEP Risk Calculator"

# Load models and data
def load_models_and_data():
    """Load all required models and data files"""
    data_dir = "data"
    
    # Load models (trained with grid search)
    with open(os.path.join(data_dir, "gbm_model.pkl"), 'rb') as f:
        main_model = pickle.load(f)
    
    with open(os.path.join(data_dir, "gbm_model_trt.pkl"), 'rb') as f:
        therapy_models = pickle.load(f)
    
    # Load LIME explainer configuration
    with open(os.path.join(data_dir, "lime_explainer_config.pkl"), 'rb') as f:
        lime_config = pickle.load(f)
    
    # Load data from app data directory
    train_impute = pd.read_csv(os.path.join(data_dir, "train_impute.csv"))
    train_new = pd.read_csv(os.path.join(data_dir, "train_new.csv"))
    var_names = pd.read_csv(os.path.join(data_dir, "var_names.csv"))
    
    # Load from main data directory
    trt_groups = pd.read_csv(os.path.join(data_dir, "trt_randomized_groups.csv"))
    
    # Feature importance for reference
    feature_importance = pd.read_csv(os.path.join(data_dir, "feature_importance.csv"))
    
    return main_model, therapy_models, lime_config, train_impute, train_new, var_names, trt_groups, feature_importance


# Load all data at startup
main_model, therapy_models, lime_config, train_impute, train_new, var_names, trt_groups, feature_importance = load_models_and_data()

# Create feature columns list (matching R training) - exclude patient_id, pep, and therapy
feature_cols = [col for col in train_impute.columns if col not in ['patient_id', 'pep']]

print(f"Feature columns: {feature_cols}")
print(f"Number of features: {len(feature_cols)}")

# Treatment options
therapy_options = [
    "No treatment",
    "Aggressive hydration only", 
    "Indomethacin only",
    "Aggressive hydration and indomethacin",
    "PD stent only",
    "Indomethacin and PD stent"
]

# Define the layout
app.layout = html.Div([
    html.Link(
        href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap",
        rel="stylesheet"
    ),
    dcc.Tabs(id="tabs", value="estimator", children=[
        dcc.Tab(
            label="Estimator", value="estimator",
            style={'fontWeight': 'bold', 'padding':'6px', 'color': '#333'},
            selected_style={'background': "#3697ff", 'padding':'6px', 'color': 'white', 'fontWeight': 'bold'}
        ),
        dcc.Tab(
            label="About", value="about",
            style={'fontWeight': 'bold', 'padding':'6px', 'color': '#333'},
            selected_style={'background': "#3697ff", 'padding':'6px', 'color': 'white', 'fontWeight': 'bold'}
        )
    ]),
    html.Div(id="tab-content")
], style={'fontFamily': 'Roboto, Arial, sans-serif'})

def create_risk_factor_inputs():
    return [
        # Sex
        html.Div([
            html.Label("Sex", style={'font-weight': 'bold'}),
            dbc.RadioItems(
                id="gender_male_1",
                options=[
                    {"label": "Male", "value": 1},
                    {"label": "Female", "value": 0}
                ],
                value=1,
                inline=True
            )
        ], style={'margin-bottom': '15px'}),
        
        # Age
        html.Div([
            html.Label("Age", style={'font-weight': 'bold'}),
            dbc.Input(
                id="age_years",
                type="number",
                value=50,
                min=0,
                max=120,
                step=1
            )
        ], style={'margin-bottom': '15px'}),
        
        # BMI
        html.Div([
            html.Label("BMI", style={'font-weight': 'bold'}),
            dbc.Input(
                id="bmi",
                type="number",
                value=25,
                min=10,
                max=60,
                step=0.1
            )
        ], style={'margin-bottom': '15px'}),
        
        # Switch inputs for binary risk factors
        *[create_switch_input(var_id, label) for var_id, label in [
            ("sod", "Sphincter of oddi dysfunction"),
            ("history_of_pep", "History of PEP"),
            ("hx_of_recurrent_pancreatitis", "History of recurrent pancreatitis"),
            ("pancreatic_sphincterotomy", "Pancreatic sphincterotomy"),
            ("precut_sphincterotomy", "Precut sphincterotomy"),
            ("minor_papilla_sphincterotomy", "Minor papilla sphincterotomy"),
            ("failed_cannulation", "Failed cannulation"),
            ("difficult_cannulation", "Difficult cannulation"),
            ("pneumatic_dilation_of_intact_biliary_sphincter", "Pneumatic dilation of intact biliary sphincter"),
            ("pancreatic_duct_injection", "Pancreatic duct injection"),
            ("pancreatic_duct_injections_2", "Pancreatic duct injections > 2"),
            ("acinarization", "Acinarization"),
            ("trainee_involvement", "Trainee involvement"),
            ("cholecystectomy", "Cholecystectomy"),
            ("pancreo_biliary_malignancy", "Pancreo biliary malignancy"),
            ("guidewire_cannulation", "Guidewire cannulation"),
            ("guidewire_passage_into_pancreatic_duct", "Guidewire passage into pancreatic duct"),
            ("guidewire_passage_into_pancreatic_duct_2", "Guidewire passage into pancreatic duct > 2"),
            ("biliary_sphincterotomy", "Biliary sphincterotomy")
        ]],
        
        # Submit button
        html.Div([
            dbc.Button(
                "Calculate",
                id="update_button",
                color="primary",
                size="lg",
                className="me-2",
                style={'margin-top': '25px'}
            )
        ])
    ]

def create_switch_input(var_id, label):
    return html.Div([
        dbc.Switch(
            id=var_id,
            label=label,
            value=False,
            style={'font-weight': 'bold', 'font-size': '16px'}
        )
    ], style={'margin-bottom': '5px'})



def create_main_estimator_layout():
    """Create the main estimator page layout"""
    return dbc.Row([
        # Risk factors input column
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Risk factors")),
                dbc.CardBody(create_risk_factor_inputs())
            ])
        ], style={'width': '15%', 'padding': '20px', 'background-color': '#f8f9fa'}),

        # Main results column
        dbc.Col([
            html.H3("Post ERCP Pancreatitis Risk Calculator and Decision Making Tool", style={'margin-top': '20px'}),
            html.Br(),
            html.Div(id="pep_pred_summary", style={'font-size': '14pt', 'margin-bottom': '20px'}),
            html.P([
                html.Span("Quick Guide: ", style={'font-weight': 'bold'}),
                "The predicted risk of developing PEP for the patient is labelled in ",
                html.Span("yellow", style={'background-color': '#FFFF00'}),
                " and plotted below.",# The patient's 20 nearest neighbors from our dataset are also plotted with their predicted risk and actual PEP outcome, where ",
                # html.Span("red", style={'color': '#D55E00'}),
                # " indicates they developed PEP and ",
                # html.Span("blue", style={'color': '#56B4E9'}),
                " indicates they did not develop PEP. "
                "The contributions of each factor in increasing or decreasing the patient's risk are estimated with LIME weights below. "
                "To aid in decision making, we also provide the predicted risk under different treatment options."
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col([dcc.Graph(id="votes_plot")], width=5), # widths of the two plots
                dbc.Col([dcc.Graph(id="lime_plot")], width=7)
            ]),
            html.Div(style={'marginBottom': '1px', 'marginTop': '1px'}), # Small spacer between plots
            dcc.Graph(id="votes_trt_plot", style={'marginTop': '0px'}),
        ], width=9)
    ])


def create_about_layout():
    """Create the about page layout"""
    return dbc.Row([
        dbc.Col([
            html.H4("The App"),
            html.P("The Post ERCP Pancreatitis Risk Calculator and Decision Making Tool was designed to estimate a patient's risk of developing post ERCP pancreatitis and the effects of different treatments in reducing the risk. Its aim is to complement, rather than replace, the decision-making of physicians. The estimator was developed using data from 7,389 patients from 12 studies. The method achieved an AUC of 0.68 under 5-fold cross-validation."),
            
            html.H4("Citation"),
            html.P("Please cite our manuscript when using this tool."),
            
            html.H4("The Team"),
            html.P("Institutions contributing data include multiple research centers collaborating on post-ERCP pancreatitis prediction."),
            
            html.H4("Technical Details"),
            html.P("This application uses gradient boosting models with optimal hyperparameters found through grid search. The models were trained with therapy-specific adjustments and feature importance is calculated using LIME (Local Interpretable Model-agnostic Explanations).")
        ], style={'margin': '20px'})
    ])


# Callback for tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value')
)
def render_tab_content(active_tab):
    if active_tab == 'estimator':
        return create_main_estimator_layout()
    elif active_tab == 'about':
        return create_about_layout()

def create_input_dataframe(age_years, gender_male_1, bmi, sod, history_of_pep, 
                          hx_of_recurrent_pancreatitis, pancreatic_sphincterotomy,
                          precut_sphincterotomy, minor_papilla_sphincterotomy,
                          failed_cannulation, difficult_cannulation,
                          pneumatic_dilation_of_intact_biliary_sphincter,
                          pancreatic_duct_injection, pancreatic_duct_injections_2,
                          acinarization, trainee_involvement, cholecystectomy,
                          pancreo_biliary_malignancy, guidewire_cannulation,
                          guidewire_passage_into_pancreatic_duct,
                          guidewire_passage_into_pancreatic_duct_2,
                          biliary_sphincterotomy):
    """Create input dataframe for all therapy options"""
    
    # Base patient data (same for all therapies)
    base_data = {
        'age_years': age_years,
        'gender_male_1': gender_male_1,
        'bmi': bmi,
        'sod': sod,
        'history_of_pep': history_of_pep,
        'hx_of_recurrent_pancreatitis': hx_of_recurrent_pancreatitis,
        'pancreatic_sphincterotomy': pancreatic_sphincterotomy,
        'precut_sphincterotomy': precut_sphincterotomy,
        'minor_papilla_sphincterotomy': minor_papilla_sphincterotomy,
        'failed_cannulation': failed_cannulation,
        'difficult_cannulation': difficult_cannulation,
        'pneumatic_dilation_of_intact_biliary_sphincter': pneumatic_dilation_of_intact_biliary_sphincter,
        'pancreatic_duct_injection': pancreatic_duct_injection,
        'pancreatic_duct_injections_2': pancreatic_duct_injections_2,
        'acinarization': acinarization,
        'trainee_involvement': trainee_involvement,
        'cholecystectomy': cholecystectomy,
        'pancreo_biliary_malignancy': pancreo_biliary_malignancy,
        'guidewire_cannulation': guidewire_cannulation,
        'guidewire_passage_into_pancreatic_duct': guidewire_passage_into_pancreatic_duct,
        'guidewire_passage_into_pancreatic_duct_2': guidewire_passage_into_pancreatic_duct_2,
        'biliary_sphincterotomy': biliary_sphincterotomy,
    }
    
    # Create dataframe for each therapy
    therapy_data = []
    therapy_configs = {
        "No treatment": (0, 0, 0),
        "Aggressive hydration only": (1, 0, 0), 
        "Indomethacin only": (0, 1, 0),
        "Aggressive hydration and indomethacin": (1, 1, 0),
        "PD stent only": (0, 0, 1),
        "Indomethacin and PD stent": (0, 1, 1)
    }
    
    for therapy, (ah, indo, stent) in therapy_configs.items():
        row_data = base_data.copy()
        row_data.update({
            'aggressive_hydration': ah,
            'indomethacin_nsaid_prophylaxis': indo,
            'pancreatic_duct_stent_placement': stent,
            'therapy': therapy
        })
        therapy_data.append(row_data)
    
    return pd.DataFrame(therapy_data)

def normalize_patient_data(input_df):
    """Normalize patient data using training set statistics"""
    # Get the feature columns used in training (excluding target, id, and therapy columns)
    feature_columns = [col for col in train_impute.columns if col not in ['patient_id', 'pep']]
    
    # Create scaler fitted on training data
    scaler = StandardScaler()
    scaler.fit(train_impute[feature_columns])
    
    # Normalize the input data (only the feature columns that exist in training)
    input_normalized = input_df.copy()
    
    # Only normalize columns that exist in both input and training data
    columns_to_normalize = [col for col in feature_columns if col in input_df.columns]
    input_normalized[columns_to_normalize] = scaler.transform(input_df[columns_to_normalize])
    
    return input_normalized

def predict_with_therapy_adjustment(input_df_normalized):
    """Make predictions with therapy-specific adjustments (matching R logic)"""
    predictions = []
    
    for _, row in input_df_normalized.iterrows():
        therapy = row['therapy']
        
        if therapy == "No treatment":
            # Use main model for no treatment
            X_input = row[feature_cols].values.reshape(1, -1)
            pred = main_model.predict_proba(X_input)[0, 1]
            
        else:
            # For treatments, use therapy-specific adjustment
            # First get baseline prediction (no treatment version)
            baseline_row = row.copy()
            baseline_row['aggressive_hydration'] = 0
            baseline_row['indomethacin_nsaid_prophylaxis'] = 0
            baseline_row['pancreatic_duct_stent_placement'] = 0
            
            X_baseline = baseline_row[feature_cols].values.reshape(1, -1)
            baseline_pred = main_model.predict_proba(X_baseline)[0, 1]
            
            if therapy in therapy_models:
                # Get predictions from therapy-specific model
                p1 = therapy_models[therapy].predict_proba(X_baseline)[0, 1]  # no treatment
                X_treatment = row[feature_cols].values.reshape(1, -1)
                p2 = therapy_models[therapy].predict_proba(X_treatment)[0, 1]  # with treatment
                
                # Apply shrinkage and adjustment (matching R logic)
                shrinkage = 1.0 if p1 > 0.1 else p1 * 10
                adj_factor = (p2/p1 * shrinkage + 1 * (1 - shrinkage)) if p1 > 0 else 1.0
                pred = baseline_pred * adj_factor
                
                # Special case for "Aggressive hydration and indomethacin" - double adjustment
                if therapy == "Aggressive hydration and indomethacin":
                    # First adjust for aggressive hydration
                    ah_row = baseline_row.copy()
                    ah_row['aggressive_hydration'] = 1
                    X_ah = ah_row[feature_cols].values.reshape(1, -1)
                    
                    p3 = therapy_models["Aggressive hydration only"].predict_proba(X_baseline)[0, 1]
                    p4 = therapy_models["Aggressive hydration only"].predict_proba(X_ah)[0, 1]
                    
                    ah_shrinkage = 1.0 if p3 > 0.1 else p3 * 10
                    ah_adj_factor = (p4/p3 * ah_shrinkage + 1 * (1 - ah_shrinkage)) if p3 > 0 else 1.0
                    baseline_pred = baseline_pred * ah_adj_factor
                    
                    # Then apply the combined therapy adjustment
                    pred = baseline_pred * adj_factor
            else:
                # Fallback to main model if therapy model not available
                pred = baseline_pred
        
        predictions.append({
            'therapy': therapy,
            'prediction': pred
        })
    
    return pd.DataFrame(predictions)

def find_nearest_neighbors(input_df_normalized, n_neighbors=20):
    """Find nearest neighbors for each therapy group"""
    neighbors_data = []
    
    # Treatment subsets (matching R logic)
    treatment_subsets = {
        "No treatment": (0, 0, 0),
        "Aggressive hydration only": (1, 0, 0),
        "Indomethacin only": (0, 1, 0),
        "PD stent only": (0, 0, 1),
        "Aggressive hydration and indomethacin": (1, 1, 0),
        "Indomethacin and PD stent": (0, 1, 1)
    }
    
    # Get feature columns for KNN (excluding treatment columns)
    knn_features = [col for col in feature_cols if col not in 
                   ['aggressive_hydration', 'indomethacin_nsaid_prophylaxis', 'pancreatic_duct_stent_placement']]
    
    for therapy, (ah, indo, stent) in treatment_subsets.items():
        # Filter training data for this therapy subset
        subset_mask = (
            (train_impute['aggressive_hydration'] == ah) &
            (train_impute['indomethacin_nsaid_prophylaxis'] == indo) &
            (train_impute['pancreatic_duct_stent_placement'] == stent)
        )
        subset_train = train_impute[subset_mask].copy()
        
        # Use available data even if less than n_neighbors
        actual_neighbors = min(n_neighbors, len(subset_train))
        if len(subset_train) > 0 and actual_neighbors > 0:
            # Fit KNN model
            knn = NearestNeighbors(n_neighbors=actual_neighbors, algorithm='ball_tree')
            knn.fit(subset_train[knn_features])
            
            # Find neighbors for this therapy
            therapy_input = input_df_normalized[input_df_normalized['therapy'] == therapy]
            if len(therapy_input) > 0:
                distances, indices = knn.kneighbors(therapy_input[knn_features])
                
                # Get neighbor data
                neighbor_rows = subset_train.iloc[indices[0]]
                
                # Make predictions for neighbors
                neighbor_predictions = []
                for _, neighbor in neighbor_rows.iterrows():
                    if therapy == "No treatment":
                        X_neighbor = neighbor[feature_cols].values.reshape(1, -1)
                        pred = main_model.predict_proba(X_neighbor)[0, 1]
                    else:
                        # Apply therapy adjustment for neighbors (simplified)
                        baseline_neighbor = neighbor.copy()
                        baseline_neighbor['aggressive_hydration'] = 0
                        baseline_neighbor['indomethacin_nsaid_prophylaxis'] = 0
                        baseline_neighbor['pancreatic_duct_stent_placement'] = 0
                        
                        X_baseline = baseline_neighbor[feature_cols].values.reshape(1, -1)
                        baseline_pred = main_model.predict_proba(X_baseline)[0, 1]
                        
                        if therapy in therapy_models:
                            X_treatment = neighbor[feature_cols].values.reshape(1, -1)
                            p1 = therapy_models[therapy].predict_proba(X_baseline)[0, 1]
                            p2 = therapy_models[therapy].predict_proba(X_treatment)[0, 1]
                            
                            shrinkage = 1.0 if p1 > 0.1 else p1 * 10
                            adj_factor = (p2/p1 * shrinkage + 1 * (1 - shrinkage)) if p1 > 0 else 1.0
                            pred = baseline_pred * adj_factor
                        else:
                            pred = baseline_pred
                    
                    neighbor_predictions.append(pred)
                
                # Store neighbor data
                for i, (_, neighbor) in enumerate(neighbor_rows.iterrows()):
                    neighbors_data.append({
                        'therapy': therapy,
                        'prediction': neighbor_predictions[i],
                        'actual_pep': neighbor['pep'],
                        'patient_id': neighbor['patient_id']
                    })
    
    # Return DataFrame with proper structure even if empty
    if len(neighbors_data) == 0:
        return pd.DataFrame(columns=['therapy', 'prediction', 'actual_pep', 'patient_id'])
    print(f"DEBUG: Found {len(neighbors_data)} neighbors across therapies")
    return pd.DataFrame(neighbors_data)

def create_lime_explanation(input_df_normalized):
    """Create LIME explanation for the patient"""
    try:
        # Get no treatment input
        no_treatment_input = input_df_normalized[input_df_normalized['therapy'] == "No treatment"]
        if len(no_treatment_input) == 0:
            return None
            
        # Recreate LIME explainer using the same parameters as in training
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=lime_config['training_data'],
            feature_names=lime_config['feature_names'],
            class_names=lime_config['class_names'],
            mode=lime_config['mode'],
            discretize_continuous=lime_config['discretize_continuous'],
            discretizer=lime_config['discretizer'],
            random_state=lime_config.get('random_state', 1)  # Use saved random state
        )
        
        # Get explanation for features (matching R implementation)
        # R uses: ncol(test_sub) - 2, where test_sub excludes patient_id and therapy
        # So if we have 25 clinical features, R shows all of them
        X_input = no_treatment_input[feature_cols].values[0]
        explanation = explainer.explain_instance(
            X_input, 
            main_model.predict_proba,
            num_features=len(feature_cols)  # Use ALL 25 clinical features like R
        )
        
        # Extract feature weights and map to readable names
        feature_weights = []
        for feature, weight in explanation.as_list():
            # Parse LIME's discretized feature names to get original feature names
            # LIME may return features in forms like "feature_name > value" or "value < feature_name <= value"; ex: "age_years > 0.79", "-0.64 < difficult_cannulation <= 1.56"
            clean_feature = feature
            
            # Case 1: "feature_name > value" or "feature_name <= value"
            if ' > ' in feature or ' <= ' in feature or ' < ' in feature:
                # Extract the feature name (everything before the operator)
                parts = feature.replace(' > ', '|').replace(' <= ', '|').replace(' < ', '|').split('|')
                if len(parts) >= 2:
                    potential_feature = parts[0].strip()
                    # Check if this is actually a feature name (not a number)
                    if potential_feature in feature_cols:
                        clean_feature = potential_feature
                    elif len(parts) >= 3:
                        # Case 2: "value < feature_name <= value" 
                        potential_feature = parts[1].strip()
                        if potential_feature in feature_cols:
                            clean_feature = potential_feature
            
            # If we still couldn't parse it, try to find any feature name in the string
            if clean_feature == feature or clean_feature.replace('-', '').replace('.', '').isdigit():
                for feat_name in feature_cols:
                    if feat_name in feature:
                        clean_feature = feat_name
                        break
                        
            feature_weights.append({
                'feature': feature,  # Keep original for display
                'clean_feature': clean_feature,
                'weight': weight,
                'abs_weight': abs(weight)
            })
        
        df_weights = pd.DataFrame(feature_weights)
        
        # Map to variable names if available
        if len(var_names) > 0:
            var_name_map = dict(zip(var_names['variable'], var_names['var_label']))
            df_weights['feature_label'] = df_weights['clean_feature'].map(var_name_map).fillna(df_weights['clean_feature'])
        else:
            df_weights['feature_label'] = df_weights['clean_feature']
        
        # Sort by absolute weight descending (highest impact first, like R)
        return df_weights.sort_values('abs_weight', ascending=False)
        
    except Exception as e:
        print(f"LIME explanation failed: {e}")
        # Return a simple feature importance as fallback
        try:
            return feature_importance.head(10).rename(columns={'importance': 'weight', 'feature': 'feature_label'}).assign(abs_weight=lambda x: abs(x['weight']))
        except:
            return None

# Main callback for predictions and plots
@app.callback(
    [Output('pep_pred_summary', 'children'),
     Output('votes_plot', 'figure'),
     Output('lime_plot', 'figure'), 
     Output('votes_trt_plot', 'figure')],
    [Input('update_button', 'n_clicks')],
    [State('age_years', 'value'),
     State('gender_male_1', 'value'),
     State('bmi', 'value'),
     State('sod', 'value'),
     State('history_of_pep', 'value'),
     State('hx_of_recurrent_pancreatitis', 'value'),
     State('pancreatic_sphincterotomy', 'value'),
     State('precut_sphincterotomy', 'value'),
     State('minor_papilla_sphincterotomy', 'value'),
     State('failed_cannulation', 'value'),
     State('difficult_cannulation', 'value'),
     State('pneumatic_dilation_of_intact_biliary_sphincter', 'value'),
     State('pancreatic_duct_injection', 'value'),
     State('pancreatic_duct_injections_2', 'value'),
     State('acinarization', 'value'),
     State('trainee_involvement', 'value'),
     State('cholecystectomy', 'value'),
     State('pancreo_biliary_malignancy', 'value'),
     State('guidewire_cannulation', 'value'),
     State('guidewire_passage_into_pancreatic_duct', 'value'),
     State('guidewire_passage_into_pancreatic_duct_2', 'value'),
     State('biliary_sphincterotomy', 'value')]
)
def update_predictions(n_clicks, age_years, gender_male_1, bmi, sod, history_of_pep,
                      hx_of_recurrent_pancreatitis, pancreatic_sphincterotomy,
                      precut_sphincterotomy, minor_papilla_sphincterotomy,
                      failed_cannulation, difficult_cannulation,
                      pneumatic_dilation_of_intact_biliary_sphincter,
                      pancreatic_duct_injection, pancreatic_duct_injections_2,
                      acinarization, trainee_involvement, cholecystectomy,
                      pancreo_biliary_malignancy, guidewire_cannulation,
                      guidewire_passage_into_pancreatic_duct,
                      guidewire_passage_into_pancreatic_duct_2,
                      biliary_sphincterotomy):
    
    if n_clicks == 0:
        # Return default/empty plots
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Click Submit to generate predictions")
        return "Click Submit to see predictions", empty_fig, empty_fig, empty_fig
    
    try:
        print(f"DEBUG: Starting prediction process...")
        
        # Handle None values by setting defaults
        age_years = age_years or 65
        gender_male_1 = gender_male_1 if gender_male_1 is not None else 1
        bmi = bmi or 25
        sod = sod if sod is not None else 0
        history_of_pep = history_of_pep if history_of_pep is not None else 0
        
        # To match R app exactly (which has bugs), we need to replicate the bug:
        # In R: hx_of_recurrent_pancreatitis = as.integer(input$history_of_pep)
        # In R: pancreatic_sphincterotomy = as.integer(input$history_of_pep)
        hx_of_recurrent_pancreatitis = history_of_pep  # Bug replication
        pancreatic_sphincterotomy = history_of_pep     # Bug replication
        
        # Handle other None values
        precut_sphincterotomy = precut_sphincterotomy if precut_sphincterotomy is not None else 0
        minor_papilla_sphincterotomy = minor_papilla_sphincterotomy if minor_papilla_sphincterotomy is not None else 0
        failed_cannulation = failed_cannulation if failed_cannulation is not None else 0
        difficult_cannulation = difficult_cannulation if difficult_cannulation is not None else 0
        pneumatic_dilation_of_intact_biliary_sphincter = pneumatic_dilation_of_intact_biliary_sphincter if pneumatic_dilation_of_intact_biliary_sphincter is not None else 0
        pancreatic_duct_injection = pancreatic_duct_injection if pancreatic_duct_injection is not None else 0
        pancreatic_duct_injections_2 = pancreatic_duct_injections_2 if pancreatic_duct_injections_2 is not None else 0
        acinarization = acinarization if acinarization is not None else 0
        trainee_involvement = trainee_involvement if trainee_involvement is not None else 0
        cholecystectomy = cholecystectomy if cholecystectomy is not None else 0
        pancreo_biliary_malignancy = pancreo_biliary_malignancy if pancreo_biliary_malignancy is not None else 0
        guidewire_cannulation = guidewire_cannulation if guidewire_cannulation is not None else 0
        guidewire_passage_into_pancreatic_duct = guidewire_passage_into_pancreatic_duct if guidewire_passage_into_pancreatic_duct is not None else 0
        guidewire_passage_into_pancreatic_duct_2 = guidewire_passage_into_pancreatic_duct_2 if guidewire_passage_into_pancreatic_duct_2 is not None else 0
        biliary_sphincterotomy = biliary_sphincterotomy if biliary_sphincterotomy is not None else 0
        
        print(f"DEBUG: All inputs processed, age={age_years}, gender={gender_male_1}, bmi={bmi}")
        
        # Create input dataframe
        input_df = create_input_dataframe(
            age_years, gender_male_1, bmi, sod, history_of_pep,
            hx_of_recurrent_pancreatitis, pancreatic_sphincterotomy,
            precut_sphincterotomy, minor_papilla_sphincterotomy,
            failed_cannulation, difficult_cannulation,
            pneumatic_dilation_of_intact_biliary_sphincter,
            pancreatic_duct_injection, pancreatic_duct_injections_2,
            acinarization, trainee_involvement, cholecystectomy,
            pancreo_biliary_malignancy, guidewire_cannulation,
            guidewire_passage_into_pancreatic_duct,
            guidewire_passage_into_pancreatic_duct_2,
            biliary_sphincterotomy
        )
        print(f"DEBUG: Input dataframe created, shape: {input_df.shape}")
        print(f"DEBUG: Input columns: {input_df.columns.tolist()}")
        
        # Normalize data
        input_df_normalized = normalize_patient_data(input_df)
        # print(f"DEBUG: Data normalized successfully")
        
        # Make predictions
        predictions_df = predict_with_therapy_adjustment(input_df_normalized)
        # print(f"DEBUG: Predictions made successfully")
        
        # Get nearest neighbors
        neighbors_df = find_nearest_neighbors(input_df_normalized)
        print(f"DEBUG: Nearest neighbors found")
        
        # Create LIME explanation
        lime_df = create_lime_explanation(input_df_normalized)
        # print(f"DEBUG: LIME explanation created")
        
        # Create summary text
        no_treatment_pred = predictions_df[predictions_df['therapy'] == 'No treatment']['prediction'].iloc[0]
        summary_text = html.Div([
            html.B(f"Based on the input risk factors for the patient, the risk of developing PEP without prophylaxis is {no_treatment_pred*100:.1f}%.")
        ])
        
        # Create plots
        votes_fig = create_votes_plot(predictions_df, neighbors_df, "No treatment")
        lime_fig = create_lime_plot(lime_df)
        treatment_fig = create_treatment_plot(predictions_df, neighbors_df)
        
        print(f"DEBUG: All plots created successfully")
        return summary_text, votes_fig, lime_fig, treatment_fig
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        error_fig = go.Figure()
        error_fig.update_layout(title=f"Error: {str(e)}")
        return f"Error in prediction: {str(e)}", error_fig, error_fig, error_fig

def create_votes_plot(predictions_df, neighbors_df, therapy_filter):
    """Create the votes plot for specified therapy"""
    patient_pred = predictions_df[predictions_df['therapy'] == therapy_filter]
    
    # Check if neighbors_df has the therapy column and data
    if 'therapy' in neighbors_df.columns and len(neighbors_df) > 0:
        neighbors_subset = neighbors_df[neighbors_df['therapy'] == therapy_filter]
    else:
        neighbors_subset = pd.DataFrame(columns=['therapy', 'prediction', 'actual_pep', 'patient_id'])
    
    fig = go.Figure()
    
    if len(neighbors_subset) > 0:
        # Add neighbor points with jitter
        np.random.seed(1)  # For reproducible jitter
        x_jitter = np.random.normal(0, 0.1, len(neighbors_subset))
        
        # Separate by PEP outcome
        no_pep = neighbors_subset[neighbors_subset['actual_pep'] == 0]
        pep = neighbors_subset[neighbors_subset['actual_pep'] == 1]
        
        if len(no_pep) > 0:
            x_jitter_no_pep = np.random.normal(0, 0.1, len(no_pep))
            fig.add_trace(go.Scatter(
                x=[0] + x_jitter_no_pep.tolist(),
                y=[None] + no_pep['prediction'].tolist(),
                mode='markers',
                marker=dict(color='#D55E00', size=8, opacity=0.7),
                name='No PEP',
                showlegend=True
            ))
        
        if len(pep) > 0:
            x_jitter_pep = np.random.normal(0, 0.1, len(pep))
            fig.add_trace(go.Scatter(
                x=[0] + x_jitter_pep.tolist(),
                y=[None] + pep['prediction'].tolist(),
                mode='markers',
                marker=dict(color='#56B4E9', size=8, opacity=0.7),
                name='Developed PEP',
                showlegend=True
            ))
    
    # Add patient point
    if len(patient_pred) > 0:
        pred_value = patient_pred['prediction'].iloc[0]
        fig.add_trace(go.Scatter(
            x=[0],
            y=[pred_value],
            mode='markers+text',
            marker=dict(color='yellow', size=15, line=dict(color='black', width=2)),
            name='Patient',
            text=[f"Patient risk: {pred_value*100:.1f}%"],
            textposition="middle right",
            textfont=dict(size=12, color='black'),
            showlegend=False
        ))
    
    # Calculate y-axis range
    all_preds = []
    if len(neighbors_subset) > 0:
        all_preds.extend(neighbors_subset['prediction'].tolist())
    if len(patient_pred) > 0:
        all_preds.extend(patient_pred['prediction'].tolist())
    
    y_max = max(0.2, max(all_preds) if all_preds else 0.2)
    
    fig.update_layout(
        title="Predicted risk of PEP (No Treatment)",
        xaxis_title="",
        yaxis_title="Probability of developing PEP",
        yaxis=dict(tickformat='.0%', range=[0, y_max]),
        xaxis=dict(showticklabels=False, range=[-0.5, 0.5]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=40) # margins between plot and edges
    )
    
    return fig

def create_lime_plot(lime_df):
    """Create LIME feature importance plot (matching R implementation)"""
    if lime_df is None or len(lime_df) == 0:
        fig = go.Figure()
        fig.update_layout(title="LIME explanation not available")
        return fig
    
    # Use ALL features (matching R: shows all 25 features)
    # Data is already sorted by absolute weight descending
    all_features = lime_df.copy()
    
    # Use feature labels (mapped names) for cleaner display
    y_labels = all_features['feature_label'] if 'feature_label' in all_features.columns else all_features['clean_feature']
    
    # Reverse order so highest impact appears at TOP (like R)
    y_labels = y_labels[::-1]
    weights = all_features['weight'][::-1]
    
    # Create horizontal bar plot with colors matching R
    colors = ['#D55E00' if w > 0 else '#56B4E9' for w in weights]
    
    fig = go.Figure(go.Bar(
        y=y_labels,
        x=weights,
        orientation='h',
        marker_color=colors,
        text=[f"{w:.3f}" for w in weights],
        textposition='auto',
        textfont=dict(size=10)
    ))
    
    fig.update_layout(
        title="LIME weights",  # Match R title
        xaxis_title="Weight (Contribution to Risk)",
        yaxis_title="Risk Factors",
        height=550,
        yaxis=dict(automargin=True, tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=9)),
        showlegend=False,
        margin=dict(t=40, b=40, l=40, r=40) # margins between plot and edges
    )
    
    return fig

def create_treatment_plot(predictions_df, neighbors_df):
    """Create treatment comparison plot"""
    # Filter out "No treatment"
    treatment_preds = predictions_df[predictions_df['therapy'] != 'No treatment'].copy()
    
    # Check if neighbors_df has the therapy column and data
    if 'therapy' in neighbors_df.columns and len(neighbors_df) > 0:
        treatment_neighbors = neighbors_df[neighbors_df['therapy'] != 'No treatment'].copy()
    else:
        treatment_neighbors = pd.DataFrame(columns=['therapy', 'prediction', 'actual_pep', 'patient_id'])
    
    # Rename therapies for better display
    therapy_rename = {
        "Aggressive hydration only": "Aggressive hydration",
        "Indomethacin only": "Rectal NSAIDs",
        "Aggressive hydration and indomethacin": "Hydration + NSAIDs", 
        "PD stent only": "PD stent",
        "Indomethacin and PD stent": "NSAIDs + PD stent"
    }
    
    treatment_preds['therapy_display'] = treatment_preds['therapy'].map(therapy_rename)
    
    # Only rename if treatment_neighbors has data
    if len(treatment_neighbors) > 0:
        treatment_neighbors['therapy_display'] = treatment_neighbors['therapy'].map(therapy_rename)
    
    fig = go.Figure()
    
    # Add neighbor points for each treatment
    therapy_order = ["Aggressive hydration", "Rectal NSAIDs", "Hydration + NSAIDs", "PD stent", "NSAIDs + PD stent"]
    
    for i, therapy in enumerate(therapy_order):
        # Only process neighbors if we have neighbor data
        if len(treatment_neighbors) > 0:
            therapy_neighbors_subset = treatment_neighbors[treatment_neighbors['therapy_display'] == therapy]
        else:
            therapy_neighbors_subset = pd.DataFrame()
            
        if len(therapy_neighbors_subset) > 0:
            # Add jitter for better visualization
            np.random.seed(1)
            x_pos = i
            x_jitter = np.random.normal(x_pos, 0.1, len(therapy_neighbors_subset))
            
            # Separate by PEP outcome
            no_pep = therapy_neighbors_subset[therapy_neighbors_subset['actual_pep'] == 0]
            pep = therapy_neighbors_subset[therapy_neighbors_subset['actual_pep'] == 1]
            
            if len(no_pep) > 0:
                x_jitter_no_pep = np.random.normal(x_pos, 0.1, len(no_pep))
                fig.add_trace(go.Scatter(
                    x=x_jitter_no_pep,
                    y=no_pep['prediction'],
                    mode='markers',
                    marker=dict(color='#56B4E9', size=6, opacity=0.6),
                    name='No PEP' if therapy == therapy_order[0] else None,
                    showlegend=(therapy == therapy_order[0]),
                    legendgroup='no_pep'
                ))
            
            if len(pep) > 0:
                x_jitter_pep = np.random.normal(x_pos, 0.1, len(pep))
                fig.add_trace(go.Scatter(
                    x=x_jitter_pep,
                    y=pep['prediction'],
                    mode='markers',
                    marker=dict(color='#D55E00', size=6, opacity=0.6),
                    name='Developed PEP' if therapy == therapy_order[0] else None,
                    showlegend=(therapy == therapy_order[0]),
                    legendgroup='pep'
                ))
    
    # Add patient predictions
    patient_x = []
    patient_y = []
    patient_text = []
    
    for i, therapy in enumerate(therapy_order):
        therapy_pred = treatment_preds[treatment_preds['therapy_display'] == therapy]
        if len(therapy_pred) > 0:
            patient_x.append(i)
            pred_val = therapy_pred['prediction'].iloc[0]
            patient_y.append(pred_val)
            patient_text.append(f"{pred_val*100:.1f}%")
    
    if patient_x:
        fig.add_trace(go.Scatter(
            x=patient_x,
            y=patient_y,
            mode='markers+text',
            marker=dict(color='yellow', size=12, line=dict(color='black', width=2)),
            name='Patient predictions',
            text=patient_text,
            textposition="top center",
            textfont=dict(size=10, color='black'),
            showlegend=False
        ))
    
    fig.update_layout(
        title="Predicted risk of PEP with treatment",
        xaxis_title="Treatment Option",
        yaxis_title="Probability of developing PEP",
        yaxis=dict(tickformat='.0%'),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(therapy_order))),
            ticktext=therapy_order,
            tickangle=45
        ),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40, l=40, r=40) # margins between plot and edges
    )
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)