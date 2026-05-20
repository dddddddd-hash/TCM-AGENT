import fitz

doc = fitz.open("data/raw/first_examle.pdf")
text = doc[10].get_text()
print(repr(text[:500]))
