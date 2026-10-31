from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Songil BE"
    database_url: str = "postgresql+psycopg://songil:songil@localhost:5432/songil"


settings = Settings()
