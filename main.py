"""Application entrypoint."""

# Import the settings module from the correct package
from config import settings

print(settings.APP_PORT)

if __name__ == "__main__":
    print("This is the main module.")
