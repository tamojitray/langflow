from lfx.services.settings.base import Settings
from lfx.services.settings.auth import AuthSettings
from lfx.services.settings.service import SettingsService

settings = Settings()
# Simulate setting the config dir as Langflow does
if not settings.config_dir:
    from platformdirs import user_data_dir
    data_dir = user_data_dir("langflow", "langflow")
    settings.config_dir = data_dir

print(f"CONFIG_DIR: {settings.config_dir}")
db_url = settings.database_url
print(f"DATABASE_URL: {db_url}")
