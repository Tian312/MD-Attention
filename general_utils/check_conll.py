import codecs,re,os,sys

conll = codecs.open(sys.argv[1],"r").read().split("\n")
pre = "O"
line=0
for c in conll:
    line+=1
    if re.search("^\s*$",c):
        pre="O"
        continue
    tag= c.split("\t")[-1]
    if re.search("^I-",tag):
        t = tag.split("-")[1]
        valid1 = "B-"+t
#        print (tag,valid1)
        if pre != valid1 and pre != tag:
            print ("error in line", line, pre,tag)
    pre = tag


