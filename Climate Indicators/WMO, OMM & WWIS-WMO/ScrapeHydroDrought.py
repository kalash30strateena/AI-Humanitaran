import os
import time
import pandas as pd
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import glob

def clean_and_rename_csv(download_dir, city_name):
    # Find the "grafico series temporalesserie t.csv" file
    target_pattern = os.path.join(download_dir, "grafico series temporalesserie t.csv")
    city_csv = os.path.join(download_dir, f"{city_name}.csv")
    found = False

    # Rename the target file to citywise name
    if os.path.exists(target_pattern):
        os.rename(target_pattern, city_csv)
        found = True

def wait_and_click(driver, by, value, timeout=30):
    """Wait for and click an element with retries and overlay handling, with robust scrolling."""
    wait = WebDriverWait(driver, timeout)
    for _ in range(3):
        try:
            element = wait.until(EC.element_to_be_clickable((by, value)))
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element
            )
            time.sleep(1)
            driver.execute_script("arguments[0].click();", element)
            time.sleep(2)
            return True
        except Exception:
            try:
                ActionChains(driver).send_keys(Keys.ESCAPE).perform()
                time.sleep(1)
            except:
                pass
    return False

def wait_for_download(download_dir, timeout=30):
    """Wait for a file to be downloaded"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        files = os.listdir(download_dir)
        if any(f.endswith('.csv') for f in files):
            time.sleep(2)
            return True
        time.sleep(1)
    return False

def verify_csv_file(filepath):
    """Keep file only if ALL 'valor_indice' values are non-null."""
    try:
        df = pd.read_csv(filepath)
        return 'valor_indice' in df.columns and df['valor_indice'].notnull().all()  ########
    except:
        return False
    
def initialize_driver(download_dir):
    """Initialize and return a new WebDriver instance"""
    chrome_options = Options()
    chrome_options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--headless=new")   # <--- for hiding crome window
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": """
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            })
        """
    })
    return driver

def select_country(driver, wait):
    """Select Argentina from the country dropdown if not already selected"""
    try:
        selected = driver.find_elements(By.XPATH, "//mat-select[@id='mat-select-0']//span[contains(text(), 'Argentina')]")
        if selected and selected[0].is_displayed():
            print("Argentina already selected.")
            return True
        print("Selecting Argentina...")
        wait_and_click(driver, By.ID, "mat-select-0")
        wait_and_click(driver, By.XPATH, "//mat-option[contains(., 'Argentina')]")
        time.sleep(2)
        return True
    except Exception as e:
        print(f"Failed to select Argentina: {e}")
        return False
    
def process_default_city(driver, wait, download_dir):
    """Process all cities/stations and download their CSVs (renamed)."""
    try:
        # Set Chrome to allow multiple automatic downloads
        driver.execute_cdp_cmd(
            "Page.setDownloadBehavior",
            {
                "behavior": "allow",
                "downloadPath": download_dir
            }
        )

        # Open station dropdown and get all city/station options
        wait_and_click(driver, By.ID, "mat-select-2")
        options = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//mat-option")))
        total_cities = len(options)
        batch_size = 15
        
        print(f"Number of cities/stations in the dropdown: {len(options)}")

        for batch_start in range(1, total_cities + 1, batch_size):
            batch_end = min(batch_start + batch_size - 1, total_cities)
            print(f"\nProcessing cities {batch_start} to {batch_end}")

            for idx in range(batch_start, batch_end + 1):
                # Open dropdown each time (it closes after selection)
                wait_and_click(driver, By.ID, "mat-select-2")
                options = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//mat-option")))
                city_name = options[idx - 1].text.strip().replace("/", "_").replace("\\", "_")
                print(f"Selecting city/station: {city_name}")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", options[idx - 1])
                time.sleep(1)
                driver.execute_script("arguments[0].click();", options[idx - 1])
                time.sleep(2)

                print("Selecting indice:...")
                wait_and_click(driver, By.ID, "mat-select-4")    
                wait_and_click(driver, By.XPATH, "//mat-option[1]")
                time.sleep(1)

                # Set "Cuatro años" for Escala temporal (first/default option)
                print("Selecting Cantidad de dias a mostrar:...")
                wait_and_click(driver, By.ID, "mat-select-6")    
                #wait_and_click(driver, By.XPATH, "//mat-option[5]")
                
                # Wait for the radio group to be present
                wait.until(EC.presence_of_element_located((By.XPATH, "//mat-radio-group")))
                wait_and_click(driver, By.XPATH, "//span[@class='mat-radio-label-content' and contains(., '2 años')]")
                

                # Example:  radio button (adjust index as needed)
                #wait_and_click(driver, By.XPATH, "(//mat-radio-button//span[@class='mat-radio-label-content'])[5]")
                #time.sleep(1)


            # Click "Visualizar"
                print("Selecting visualizer...")
                wait_and_click(driver, By.XPATH, "//button[contains(., 'Visualizar')]")
                time.sleep(5)

                try:
                    no_data = driver.find_element(By.XPATH, "//*[contains(text(), 'No hay datos disponibles')]")
                    if no_data.is_displayed():
                        print(f"No data available for {city_name}. Skipping.")
                        continue
                except NoSuchElementException:
                    pass

                # Download CSV
                print(f"Downloading CSV for {city_name}...")
                wait_and_click(driver, By.XPATH, "//button[contains(., 'Descargar CSV')]")
                if not wait_for_download(download_dir):
                    print(f"Download timeout - no CSV file received for {city_name}")
                    continue

                # Clean up and rename only the correct CSV
                clean_and_rename_csv(download_dir, city_name)
                city_csv = os.path.join(download_dir, f"{city_name}.csv")

                if not verify_csv_file(city_csv):
                    print(f"Downloaded CSV file is invalid or empty for {city_name}..removing")
                    os.remove(city_csv)
                    continue

                print(f"Successfully downloaded CSV for {city_name}")
            
            # After each batch except the last, sleep and refresh
            if batch_end < total_cities:
                print("Sleeping before next batch...")
                time.sleep(10)  
                driver.refresh()
                time.sleep(10)  # Wait for page to reload    

        return True

    except Exception as e:
        print(f"Error processing default city/station: {e}")
        return False

def hydro_run_sissa_download():
    """Run the SISSA download process for all cities/stations, keeping only citywise CSVs."""
    # Setup paths
    current_dir = Path(__file__).resolve().parent
    download_dir = str(current_dir / "downloads_hydro")
    os.makedirs(download_dir, exist_ok=True)

    # DO NOT REMOVE any existing CSVs at the start!

    driver = None
    try:
        driver = initialize_driver(download_dir)
        wait = WebDriverWait(driver, 30)

        print("Opening SISSA website...")
        driver.get("https://dashboard.crc-sas.org/sissa/ssi/series-temporales")
        time.sleep(20)

        # Select Argentina
        if not select_country(driver, wait):
            raise Exception("Failed to select Argentina")

        # Process all cities/stations
        print("\nProcessing all cities/stations...")
        if process_default_city(driver, wait, download_dir):
            print("✅ Download complete.")
        else:
            print("⚠️ Failed to download data for the cities/stations.")

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        if driver:
            driver.save_screenshot(str(current_dir / "error_screenshot.png"))
    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    hydro_run_sissa_download()