**Project structure**

football-predictor/
├── backend/
│   ├── preprocessing/
│   │   ├── cleaner.py          # data cleaning
│   │   ├── cleaning_config.py
│   ├── pipeline.py             # main pipeline for transformation data
│   └── downloader.py           # load data from Kaggle
│
├── model/
│   └── predictor.py            # model (example)
│
├── notebooks/
│   └── analyse.ipynb           # notebook for analyse (example)
│
├── trained_model/              #
│
├── data/                       # data storage
│   └── dataset.csv
│
├── ui/                         # visual
│   ├── app_styles.py           # styles for streamlit app
│   └── streamlit_app.py        #  streamlit app
│
├── utils/
│   ├── logger_config.py       # config for logger
│   └── data_io.py          # load/ save csv, model
├── .env                     # environment variables
├── .env.example
├── config.py                # check for environment variables and load them in one go
├── .gitignore
├── requirements.txt         # dependencies
└── README.md


