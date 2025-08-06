import streamlit as st

"""
Color Names:
White: #ffffff, Black: #000000, Gray: #808080, Red: #ff0000, Blue: #0000ff, Green: #00ff00, Yellow: #ffff00, Pink: #ffc0cb,
Orange: #ffa500, Purple: #800080, LightBlue: #add8e6, Coral: #FF7F50, Lavender: #E6E6FA, Sky Blue: #87CEEB, Beige: #F5F5DC,
Light Green: #90EE90, Dark Green: #006400, Light Pink: #FFB6C1, Sea Green: #2E8B57, Semi-transparent Blue: hsla(210, 100%, 50%, 0.5),
Transparent Black: hsla(0, 0%, 0%, 0.2)
"""


class AppStyle:
    """
    A utility class for applying consistent visual styling to a Streamlit application.

    This class provides static methods to:
    - Set the app's background color.
    - Center a title on the page.
    - Add a footer with author and license info.
    - Display a styled prediction result block.
    """

    @staticmethod
    def apply_background_color(color: str = "#ffffff") -> None:
        """
        Sets the background color of the Streamlit app.
        :param color: A valid CSS color value (e.g., hex code or named color).
                         Default is white ('#ffffff').
        :return:
        """
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: {color};
            }}
            </style>
        """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def center_title(text: str) -> None:
        """
        Displays a centered title on the Streamlit page.
        :param text: The title text to display.
        :return: None
        """
        st.markdown(
            f"<h1 style='text-align: center;'>{text}</h1>", unsafe_allow_html=True
        )

    @staticmethod
    def add_footer(author: str = "Your Name", year: str = "2025") -> None:
        """
        Adds a custom footer at the bottom of the app with author information
        and a non-commercial license notice.
        :param author: Name of the author or creator.
        :param year: Year to display in the copyright.
        :return: None
        """
        st.markdown("---")
        footer = f"""
        <div style="text-align: center; font-size: 15px; color: #666;">
            Created by <strong>{author}</strong> © {year}<br>
            License: <a href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a><br>
            For personal and non-commercial use only.
        </div>
        """
        st.markdown(footer, unsafe_allow_html=True)

    @staticmethod
    def show_prediction_block(result: str) -> None:
        """
        Displays a styled prediction result block centered on the page.
        :param price: The predicted price to display.
        :return: None
        """
        st.markdown(
            f"""
                <div style='
                    background-color:#e0f7fa;
                    color:#006064;
                    padding:15px;
                    border-radius:10px;
                    text-align:center;
                    font-size:18px;
                    margin-top:20px;
                '>
                     <strong>{result} </strong>
                </div>
                """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def apply_custom_css() -> None:
        """
        Applies all custom CSS styles for the football prediction app.
        This includes team logos, statistics, headers, and layout styling.
        :return: None
        """
        st.markdown("""
        <style>
            /* Global Styles */
            .main-header {
                background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
                padding: 20px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }

            .main-header h1 {
                color: white;
                font-size: 2.5rem;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .stats-container {
                background: rgba(255, 255, 255, 0.05);
                padding: 20px;
                border-radius: 15px;
                margin: 15px 0;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .vs-divider {
                text-align: center;
                font-size: 3rem;
                font-weight: bold;
                color: #FF6B6B;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                margin: 20px 0;
            }

            .team-logo {
                width: 240px !important;
                height: 240px !important;
                object-fit: contain;
                border-radius: 10px;
                padding: 5px;
                margin-bottom: 15px;
                display: block;
                margin-left: auto;
                margin-right: auto;
            }

            /* Centered titles */
            .centered-title {
                text-align: center;
                margin: 5px 0;
            }

            .centered-subtitle {
                text-align: center;
                margin: 15px 0;
            }

            /* Center selectboxes and team headers */
            div[data-testid="column"] h4 {
                text-align: center;
            }

            /* Center text in selectboxes - More specific targeting */
            div[data-testid="stSelectbox"] > div > div {
                text-align: center !important;
            }

            /* Center the selected value display */
            div[data-testid="stSelectbox"] > div > div > div {
                text-align: center !important;
                justify-content: center !important;
            }

            /* Center the input element inside selectbox */
            div[data-testid="stSelectbox"] input {
                text-align: center !important;
            }

            /* Center dropdown options */
            div[data-testid="stSelectbox"] div[role="listbox"] {
                text-align: center !important;
            }

            div[data-testid="stSelectbox"] div[role="listbox"] div {
                text-align: center !important;
                justify-content: center !important;
            }

            /* Center dropdown menu items */
            div[data-testid="stSelectbox"] ul li {
                text-align: center !important;
                justify-content: center !important;
            }

            /* Additional targeting for selectbox content */
            .stSelectbox > div > div > div {
                text-align: center !important;
            }

            /* Center text in the selectbox widget */
            .stSelectbox [data-baseweb="select"] > div {
                text-align: center !important;
                justify-content: center !important;
            }

            /* Center the placeholder and selected text */
            .stSelectbox [data-baseweb="select"] > div > div {
                text-align: center !important;
            }

            /* Style for predict button container */
            .predict-container {
                margin: 20px 0;
                text-align: center;
            }

            /* Preview section styling */
            .preview-container {
                background: rgba(255, 255, 255, 0.03);
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            /* FIFA-style team statistics */
            .fifa-stats-container {
                text-align: center;
                margin-top: 15px;
            }

            .fifa-stats-row {
                display: flex;
                justify-content: center;
                gap: 0px;
            }

            .fifa-stat-item {
                flex: 0 0 auto;
                padding: 0px;
                text-align: center;
                margin: 0 10px;
            }

            .fifa-stat-label {
                font-size: 0.8rem;
                color: #ccc;
            }

            .fifa-stat-value {
                font-size: 2.4rem;
                font-weight: bold;
            }

            /* Team name styling */
            .team-name {
                text-align: center;
                font-weight: bold;
                font-size: 1.2rem;
                margin-bottom: 10px;
            }

            /* Recent matches styling */
            .recent-matches-title {
                text-align: center;
                font-size: 0.9rem;
                color: #ccc;
                margin: 10px 0 8px 0;
                font-weight: bold;
            }

            .recent-matches-container {
                display: flex;
                justify-content: center;
                gap: 1px;
                margin-bottom: 15px;
            }

            .recent-match-item {
                display: flex;
                flex-direction: column;
                align-items: center;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 6px;
                padding: 4px 3px;
                min-width: 40px;
                font-size: 0.7rem;
            }

            .match-result {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
                font-size: 0.65rem;
                margin-bottom: 2px;
            }

            .match-result.win {
                background-color: #4CAF50;
                color: white;
            }

            .match-result.loss {
                background-color: #f44336;
                color: white;
            }

            .match-result.draw {
                background-color: #FF9800;
                color: white;
            }

            .match-opponent {
                font-size: 0.6rem;
                color: #ccc;
                text-align: center;
                margin-bottom: 2px;
                line-height: 1;
            }

            .match-score {
                font-size: 0.6rem;
                color: #fff;
                font-weight: bold;
            }

            .recent-matches-error {
                text-align: center;
                font-size: 0.8rem;
                color: #ff6b6b;
                margin: 10px 0;
                font-style: italic;
            }

            /* Prediction result styling */
            .prediction-result {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 25px;
                border-radius: 20px;
                text-align: center;
                color: white;
                margin: 20px 0;
                box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            }

            .prediction-outcome {
                margin: 0;
                font-size: 2rem;
            }

            .prediction-score {
                margin: 15px 0;
                font-size: 1.8rem;
                color: #FFD700;
            }

            .prediction-confidence {
                margin-top: 20px;
                font-size: 1.1rem;
                font-weight: bold;
            }

            /* Default emoji styles for fallback */
            .default-team-emoji {
                text-align: center;
                font-size: 4rem;
                margin: 15px 0;
            }
        </style>
        """, unsafe_allow_html=True)