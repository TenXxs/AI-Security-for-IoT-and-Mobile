import os
import re

MD_FILE_NAME = "README.md"
PDF_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "IoT"))

class Stack:
    def __init__(self, *args, **kwargs):
        self._data = []
    
    def pop(self):
        return self._data.pop()
    
    def push(self, elem):
        self._data.append(elem)
    
    def peek(self):
        return self._data[-1]
    
    def len(self):
        return len(self._data)

def getAllPDF(path = ""):
    allFiles = []
    if (path == ""):
        return allFiles
    for root, dirs, files in os.walk(path, True):
        tag = ">".join(os.path.relpath(root, path).split(os.sep))
        for each in files:
            if (re.search(r"(.pdf)$", each) != None):
                each = each.replace(" --", ":")
                ext = re.split(r"【([0-9]*) ([a-zA-Z&0-9 ]*)】(.*)(\.pdf)$", each)
                allFiles.append([tag, ext[1], ext[2], ext[3]])
    return allFiles

def classify(items):
    rtn = {}
    def simpleRecur(dir, cs, value):
        if (len(cs) == 1):
            if cs[0] not in dir:
                dir[cs[0]] = []
            dir[cs[0]].append(value)
        else:
            if cs[0] not in dir:
                dir[cs[0]] = {}
            simpleRecur(dir[cs[0]], cs[1::], value)

    for each in items:
        category, *rest = each
        cs = category.split(">")
        simpleRecur(rtn, cs, rest)

    return rtn

def generateMarkDown(info= {}):
    md = ""
    stack = Stack()

    for each in info.keys():
        stack.push([info[each], 0])
        md = f"""{md}\n{"#"*(stack.len() + 1)} {each}\n"""
        while(stack.len() > 0):
            top, index = stack.pop()
            if isinstance(top, type({})):
                if (index != len(top.keys())):
                    stack.push([top, index + 1])
                    newKey = list(top.keys())[index]
                    stack.push([top[newKey], 0])
                    md = f"""{md}\n{"#"*(stack.len() + 1)} {newKey}\n"""
            else:
                for row in top:
                    md = f"{md}\n- {row[0]}, **{row[1]}**, {row[2]}\n"
                
    return md

pdfs = getAllPDF(PDF_DIR)
preprint = classify(pdfs)
md = generateMarkDown(preprint)
with open(MD_FILE_NAME, "w+") as fout:
    fout.write(md)
