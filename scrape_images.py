from selenium import webdriver
import time
import requests

print("Opening Browser")
# Open the web browser, log in, and go to the image.
browser = webdriver.Firefox()
browser.get('http://tattoodles.com/login.php')
usernameElem = browser.find_element_by_name('username')
usernameElem.send_keys('maxisawesome')
passwordElem = browser.find_element_by_name('password')
passwordElem.send_keys('T4ttoodlespass14')
passwordElem.submit()
time.sleep(3)
browser.get('http://tattoodles.com/tattooimages.php/print/6433.jpg')

# Transfer cookies to requests
headers = {
        "User-Agent":
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"}
s = requests.session()
s.headers.update(headers)

for cookie in browser.get_cookies():
    c = {cookie['name']: cookie['value']}
    s.cookies.update(c)

print("Beginning downloading")
# get the image while logged in
for i in range(15000):
    time.sleep(.25)
    if i+1 % 100 == 1:
        print("img {} of 15000".format(i+1))
    imgurl = 'http://www.tattoodles.com/tattooimages.php/print/{}.jpg'
    imgurl = 'http://www.tattoodles.com/tattooimages.php/print/{}.jpg'.format((i+1))
    req = s.get(imgurl, allow_redirects=True)

    fails = []
    successes = []
    if len(req.content) > 0: 
        with open('data/{}.jpeg'.format(i+1), 'wb') as f:
            f.write(req.content)
            successes.append(i)
    else:
        fails.append(i)

print("Successes: ", successes)
print("Fails: ", fails)
print("Total Images Downloaded: ", len(successes))
print("Total Fails Downloaded: ", len(fails))
