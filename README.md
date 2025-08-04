**Project structure**

```bash
football-predictor/ 
├── src
│    ├── backend/ 
│    │   ├── downloader/
│    │   │    ├── base_downloader.py
│    │   │    ├── local_downloader.py    
│    │   │    └── downloader_registry.py
│    │   │
│    │   ├── transformers/
│    │   │    ├── base_transformer.py
│    │   │    ├── local_transformer.py   
│    │   │    └── transformer_registry.py
│    │   │
│    │   ├── cleaner/         
│    │   │    ├── cleaner.py        
│    │   │    └── cleaning_config.py  
│    │   │
│    │   ├── model/
│    │   │    ├── trained_model/
│    │   │    │      └── trained.joblib 
│    │   │    └── base_predictor.py 
│    │   │
│    │   ├── trainer_pipeline.py
│    │   └── data_pipeline.py
│    │
│    └── frontend/                       
│         ├── app_styles.py             
│         └── streamlit_app.py
│  
├── notebooks/  
│   └── analyse.ipynb           # notebook for analyse (example)  
│  
├── data/                       # data storage
│   ├── prepared/
│   │    └── data.csv
│   └── raw/                   
│  
├── utils/  
│   ├── data_io.py             # utils for save/load csv/ model 
│   └── logger_config.py       # config for logger  
├── .env                       # environment variables  
├── .env.example               
├── config.py                      
├── .gitignore  
├── requirements.txt         # dependencies  
└── README.md  
```

**Environment variables**

This project uses environment variables to configure paths and URLs.
To get started, copy the .env.example file to .env and customize it if needed:

```bash
cp .env.example .env
``
Edit .env to set your local paths and configuration. For example:

```ini
CSV_SAVE_LOAD_PATH=./data/
MODEL_SAVE_LOAD_PATH=./trained_model/
BELGIUM_DATA_BASE_URL=https://www.football-data.co.uk/
```