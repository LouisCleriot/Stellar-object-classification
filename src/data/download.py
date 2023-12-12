import requests
from halo import Halo

def download_file_from_google_drive(file_id, destination):

    URL = f"https://drive.google.com/uc?id={file_id}&export=download"

    with requests.get(URL, stream=True) as response:
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

if __name__ == "__main__":
    file_id = "1gQ467ctq498CPSW537EiUHWh3f3pTmpc"
    
    destination = "data/raw/star_classification.csv"  # Change the file extension as needed
    spinner = Halo(text="Downloading dataset", spinner="dots")
    spinner.start()
    try:
        download_file_from_google_drive(file_id, destination)
        spinner.succeed("Dataset downloaded successfully to data/raw/star_classification.csv")
    except:
        spinner.fail("Failed to download dataset")
