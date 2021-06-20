import sys
from nltk.tokenize import word_tokenize, sent_tokenize


def count_test_train():
	type_set = set()

	for file in ["gro-ner-train.csv", "gro-ner-test.csv"]:
		with open(file, encoding="utf-8") as f:
			for line in f:
				type_set.add(line.split(";")[0])
	print(len(type_set))


def count_original():
	type_set = set()
	token_count = 0
	with open("../data_gronings.txt", encoding="utf-8") as f:
		lines = ""
		word_list = []
		for i, line in enumerate(f.readlines()):
			lines += line + " "
		for sent in sent_tokenize(lines):
			for word in word_tokenize(sent, language="dutch"):
				type_set.add(word)
				token_count += 1
	print("Tokens: " + str(token_count))
	print("Types: " + str(len(type_set)))


def main():
	if sys.argv[1] == "train_test":
		count_test_train()
	elif sys.argv[1] == "original":
		count_original()
	else:
		sys.stderror.write("Please give one of (train_test | original) as a first command line argument\nE.g.: python3 Data.py train_test")


if __name__ == '__main__':
	main()