from bs4 import BeautifulSoup
import requests
import re

url = "http://www.gutenberg.org/files/140/140-h/140-h.htm#link2HCH0002"
book = requests.get(url)
chapters = {}
chapters_raw = re.compile('</h2>(.*?)<h2>', re.DOTALL).findall(book.text)
chapters_soup = [BeautifulSoup(text, 'html.parser') for text in chapters_raw]
for i in range(1, len(chapters_soup)):
    pars = chapters_soup[i].find_all('p')
    chapters[i] = pars

pars_counter = 0
cont=False
first_read = True

print("""
 /$$$$$$$$/$$                    /$$$$$                               /$$          
|__  $$__/ $$                   |__  $$                              | $$          
   | $$  | $$$$$$$   /$$$$$$       | $$ /$$   /$$ /$$$$$$$   /$$$$$$ | $$  /$$$$$$ 
   | $$  | $$__  $$ /$$__  $$      | $$| $$  | $$| $$__  $$ /$$__  $$| $$ /$$__  $$
   | $$  | $$  \ $$| $$$$$$$$ /$$  | $$| $$  | $$| $$  \ $$| $$  \ $$| $$| $$$$$$$$
   | $$  | $$  | $$| $$_____/| $$  | $$| $$  | $$| $$  | $$| $$  | $$| $$| $$_____/
   | $$  | $$  | $$|  $$$$$$$|  $$$$$$/|  $$$$$$/| $$  | $$|  $$$$$$$| $$|  $$$$$$$
   |__/  |__/  |__/ \_______/ \______/  \______/ |__/  |__/ \____  $$|__/ \_______/
                                                            /$$  \ $$              
                                                           |  $$$$$$/              
                                                            \______/               
""")
print("")
print("")
print("By Upton Sinclair")
print("Thanks to project Gutenberg")


print("Type q to quit")
print("Type b to look busy")
print("Type any other key to continue reading")
print("Enjoy!")

x = 'y'
while(True):

    if first_read == True:
        chapter = input("Start reading chapter: ")
        chapter = int(chapter)
        first_read = False

    else:
        x = input("continue: ")

    if chapter == 'q' or x == 'q':
        break
    try:
        print(chapters[chapter][pars_counter].prettify())
        pars_counter+=1
    except IndexError:
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        print(" ")
        cont_str = input("Chapter completed. Continue? y for yes: ")
        if cont_str == 'y' or cont_str == '':
            chapter += 1
            pars_counter = 0
        else:
            print("Goodbye!")
            break


