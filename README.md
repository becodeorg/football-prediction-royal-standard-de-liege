**Project structure**

```bash
football-predictor/  
├── backend/ 
│   ├── downloader/
│   │    ├── base_downloader.py
│   │    ├── local_downloader.py    
│   │    └── downloader_registry.py
│   │
│   ├── transformers/
│   │    ├── base_transformer.py
│   │    ├── local_transformer.py   # example
│   │    └── transformer_registry.py
│   │
│   ├── cleaner/         
│   │    ├── cleaner.py     # data cleaning   
│   │    └── cleaning_config.py  # maybe don't need   
│   │
│   └── data_pipeline.py
│
├── model/  
│   └── base_predictor.py           # model 
│  
├── notebooks/  
│   └── analyse.ipynb           # notebook for analyse (example)  
│  
├── data/                       # data storage
│   ├── dataset.csv             # raw data
│   └── local.csv               # prepared data
│  
├── ui/                         # visual   
│   ├── app_styles.py           # styles for streamlit app  
│   └── streamlit_app.py        #  streamlit app  
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