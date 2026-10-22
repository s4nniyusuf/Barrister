"""Settings for the Barrister API."""


class Settings:
    """Static configuration values used when creating the API application."""

    # Local Streamlit addresses allowed to call the API from a browser.
    cors_origins = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ]


settings = Settings()
