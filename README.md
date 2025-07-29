**Project structure**

```bash
football-predictor/  
├── backend/ 
│   ├── downloader/
│   │    ├── base_downloader.py
│   │    ├── some_downloader.py    # example
│   │    ├── some_2_downloader.py  # example
│   │    └── downloader_registry.py
│   │
│   ├── transformers/
│   │    ├── base_transformer.py
│   │    ├── some_transformer.py   # example
│   │    ├── some_2_transformer.py # example
│   │    └── transformer_registry.py
│   │
│   ├── cleaner/         
│   │    ├── data_cleaner.py     # data cleaning   
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
