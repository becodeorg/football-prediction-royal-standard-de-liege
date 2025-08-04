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
