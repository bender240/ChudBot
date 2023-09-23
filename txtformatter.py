import regex as re

filename = r'adv.txt'

string = open(filename).read()

cleaned = re.sub('1', '', string)
cleaned = re.sub('2', '', cleaned)
cleaned = re.sub('3', '', cleaned)
cleaned = re.sub('4', '', cleaned)
cleaned = re.sub('5', '', cleaned)
cleaned = re.sub('6', '', cleaned)
cleaned = re.sub('7', '', cleaned)
cleaned = re.sub('8', '', cleaned)
cleaned = re.sub('9', '', cleaned)
cleaned = re.sub('0', '', cleaned)

open(advclean.txt, 'w').writelines(cleaned)
