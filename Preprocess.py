# Preprocessing
# We decided to look at 50 pages from the 1st harry potter book
# Pages 102 to 152 (inclusive)

file = open("./Books/HP1.txt","r")
book = ""
for i in range(153):
    if i >= 102:
        book += file.readline()
    else:   
        file.readline()

words = book.lower()
new_book =""

# remove all punctuation from the text
for char in words:
    ascii_val = ord(char)
    if ((48 <= ascii_val <= 57) or (97 <= ascii_val <= 122)) or char == " ":
        new_book+= char


words = new_book.split(" ")