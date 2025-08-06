# Jupiler Pro League Prediction Pro

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.47+-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.7+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An intelligent football match prediction system for the Belgian Pro League (Jupiler Pro League) using machine learning. The project includes a complete pipeline from data loading to an interactive web interface for obtaining predictions.

## Try It Live!

**Ready to make predictions?**

**[Launch Application](https://your-app-name.streamlit.app)** *(coming soon)*

No installation required! Select your teams and get instant ML-powered predictions for Belgian Pro League matches with confidence scores and detailed statistics.

---

## Key Features

- **Automatic data loading** from football-data.co.uk
- **Advanced data processing** with feature engineering
- **Machine learning** with RandomForestClassifier and GridSearchCV
- **Interactive web interface** deployed on Streamlit Cloud
- **Team and match statistics visualization** 
- **Cloud model storage** in Dropbox
- **CI/CD pipeline** with automatic retraining
- **Live predictions** accessible via web browser

## Quick Start

**For Users**: Use the [live application](https://your-app-name.streamlit.app) above - no setup needed!

**For Developers**: Follow the [Installation and Setup](#🛠️-installation-and-setup) section to run locally or contribute.

## Project Structure

```
football-prediction-royal-standard-de-liege/
├── src/
│   ├── backend/                              # Backend logic
│   │   ├── downloaders/                      # Data downloaders
│   │   │   ├── base_downloder.py             # Base downloader class
│   │   │   ├── belgium_league_downloader.py  # Belgian Pro League data downloader
│   │   │   └── local_downloader.py           # Local downloader
│   │   ├── transformers/                     # Data transformers
│   │   │   ├── base_transformer.py           # Base transformer
│   │   │   ├── belgium_league_transformer.py # Belgian Pro League transformer
│   │   │   └── local_transformer.py          # Local transformer
│   │   ├── cleaner/                          # Data cleaning
│   │   │   ├── cleaner.py                    # Main cleaning class
│   │   │   └── cleaning_config.py            # Cleaning configuration
│   │   ├── model/                            # ML model
│   │   │   ├── ml_model.py                   # Model training
│   │   │   └── trained_model/                # Saved models
│   │   ├── registry/                         # Component registries
│   │   │   ├── downloader_registry.py        # Downloader registry
│   │   │   └── transformer_registry.py       # Transformer registry
│   │   ├── utils/                            # Backend utilities
│   │   │   └── combine_dfs.py                # DataFrame combining
│   │   ├── data_pipeline.py                  # Data processing pipeline
│   │   └── trainer_pipeline.py               # Model training pipeline
│   └── frontend/                             # Frontend application
│       ├── streamlit_app.py                  # Main Streamlit application
│       ├── components/                       # UI components
│       │   ├── feature_prepare.py            # Feature preparation for prediction ??
│       │   ├── input_form.py                 # ??
│       │   └── jupiler_teams_data.csv        # Team components
│       └── styles/                           # Interface styles
│           └── app_styles.py                 # CSS styles
├── data/                                     # Data
│   ├── raw/                                  # Raw data
│   │   └── historical_B1_data.csv            # Historical data
│   └── prepared/                             # Processed data
│       └── Belgium_league_2526.csv           # Prepared Belgian Pro League data
├── utils/                                    # Common utilities
│   ├── data_io.py                            # Data and model I/O
│   ├── logger_config.py                      # Logging configuration
│   └── dropbox/                              # Dropbox integration
│       ├── dropbox_io.py                     # Dropbox operations
│       └── token_manager.py                  # Token management
├── .github/workflows/                        # GitHub Actions (fix path!)
│   └── pipeline.yml                          # CI/CD pipeline
├── config.py                                 # Application configuration
├── main.py                                   # Main script (backend)
├── requirements.txt                          # Python dependencies
├── .env.example                              # Environment variables example
└── README.md                                 # Documentation
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/football-prediction-royal-standard-de-liege.git
cd football-prediction-royal-standard-de-liege
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

```bash
cp .env.example .env
```

Edit the `.env` file:

```ini
# Data save/load paths

CSV_SAVE_LOAD_PATH=./data/
MODEL_SAVE_LOAD_PATH=./src/backend/model/trained_model/

# Data source URL

BELGIUM_DATA_BASE_URL=https://www.football-data.co.uk/

# Dropbox configuration (optional)

DROPBOX_ACCESS_TOKEN=your_dropbox_access_token
DROPBOX_REFRESH_TOKEN=your_dropbox_refresh_token
DROPBOX_APP_KEY=your_dropbox_app_key
DROPBOX_APP_SECRET=your_dropbox_app_secret
DROPBOX_TOKEN_URL=https://api.dropbox.com/oauth2/token
DROPBOX_MODEL_PATH=/models/latest_model.joblib
```

## Usage

### For End Users: Live Web Application

**[Use Live Application](https://your-app-name.streamlit.app)** *(Recommended)*

The easiest way to get predictions - just visit the link above!

### For Developers: Local Development

**Run the full pipeline:**

```bash
python main.py
```

This script performs:
1. Loading fresh data
2. Processing and feature engineering
3. ML model training
4. Model upload to Dropbox

**Run the web application locally:**

```bash
streamlit run src/frontend/streamlit_app.py
```

Open your browser and go to `http://localhost:8501`

### Run individual components

**Data processing only:**
```python
from src.backend.data_pipeline import run_pipeline
run_pipeline(source_name="Belgium_league_2526", save=True)
```

**Model training only:**
```python
from src.backend.trainer_pipeline import run_training_pipeline
model = run_training_pipeline(source_name="Belgium_league_2526", save=True)
```

## ML Architecture

### Machine Learning Model
- **Algorithm**: RandomForestClassifier
- **Optimization**: GridSearchCV for hyperparameter tuning
- **Target variable**: Match result (1 - home win, 0 - draw, -1 - away win)

### Feature Engineering
The system creates the following features:
- **Team form**: results of the last 5/10 matches
- **Goal statistics**: goals scored/conceded
- **Home/away performance**: separate statistics for home and away games
- **Head-to-head**: history of personal meetings
- **Win percentage**: win rate for recent matches

### Data processing pipeline
```
Raw data → Cleaning → Feature Engineering → Model Training → Predictions
```

## Web Interface

**Live Application**: Deployed on Streamlit Cloud for instant access ([link at top](#🌐-try-it-live))

The application features:
- **Team selection** from dropdown lists
- **ML predictions** with confidence indication
- **Score prediction** for matches
- **Team statistics** with visualization
- **Radar charts** for team comparison

**Local Development**: You can also run the application locally for development purposes (see [Usage](#🚀-usage) section).

## CI/CD

**Backend Pipeline (GitHub Actions)**:
- Retrains the model every Sunday at 3:00 UTC
- Uploads the updated model to Dropbox
- Supports manual pipeline execution

**Frontend Deployment (Streamlit Cloud)**:
- Automatically deploys from the main branch
- Fetches the latest model from Dropbox
- Provides 24/7 availability for predictions

## Dependencies

Main libraries:
- **pandas 2.3.1** - data processing
- **scikit-learn 1.7.1** - machine learning
- **streamlit 1.47.1** - web interface
- **beautifulsoup4 4.13.4** - web scraping
- **requests 2.32.4** - HTTP requests
- **dropbox 12.0.2** - cloud storage
- **pydantic 2.11.7** - data validation

## Architectural Patterns

The project uses:
- **Registry Pattern** for registering downloaders and transformers
- **Strategy Pattern** for different data loading strategies
- **Template Method** in base classes
- **Dependency Injection** through Pydantic Settings
- **Pipeline Pattern** for data processing

## Contributors

The project is developed by "The Dream Team":
- Santo
- Herve
- Konstantin

## Future Development

Improvement plans:
- [ ] 


## License

MIT License - details in the LICENSE file.

---

**Wishing you successful predictions!** 
