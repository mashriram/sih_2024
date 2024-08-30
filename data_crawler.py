import requests
from bs4 import BeautifulSoup

# Initial URL
url = "https://fcainfoweb.nic.in/reports/report_menu_web.aspx"

# Form data to be sent in the POST request
payload = {
    "ctl00$MainContent$Ddl_Rpt_type": "Retail",
    "ctl00$MainContent$ddl_Language": "English",
    "ctl00$MainContent$Rbl_Rpt_type": "Price report",
    "ctl00$MainContent$Ddl_Rpt_Option0": "Daily Prices",
    "ctl00$MainContent$Txt_FrmDate": "01/08/2024",
    "ctl00$MainContent$btn_getdata1": "Get Data",
}

# Start a session to handle cookies and redirects
session = requests.Session()

# Make the initial POST request with the form data
response = session.post(url, data=payload)
print(response.content)

# Check if the response is a 302 redirect
if response.status_code == 302:
    # Get the new location from the response headers
    redirect_url = response.headers.get("Location")

    # Make a GET request to the redirect URL with the cookies
    final_response = session.get(redirect_url)

    # If the request was successful, parse the content with BeautifulSoup
    if final_response.status_code == 200:
        soup = BeautifulSoup(final_response.content, "html.parser")
        # Now you can work with the soup object to extract data
        print(soup.prettify())
    else:
        print(
            f"Failed to retrieve content from {redirect_url}, status code: {final_response.status_code}"
        )
else:
    print(f"Unexpected status code: {response.status_code}")
