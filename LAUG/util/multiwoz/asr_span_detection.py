import json
import locale
 
NUMBER_CONSTANT = {0:"zero ", 1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six", 7:"seven",
                8:"eight", 9:"nine", 10:"ten", 11:"eleven", 12:"twelve", 13:"thirteen",
                14:"fourteen", 15:"fifteen", 16:"sixteen", 17:"seventeen", 18:"eighteen", 19:"nineteen" };
IN_HUNDRED_CONSTANT = {2:"twenty", 3:"thirty", 4:"forty", 5:"fifty", 6:"sixty", 7:"seventy", 8:"eighty", 9:"ninety"}
BASE_CONSTANT = {0:" ", 1:"hundred", 2:"thousand", 3:"million", 4:"billion"};
def translateNumberToEnglish(number):
    if str(number).isnumeric():
        if str(number)[0] == '0' and len(str(number)) > 1:
            return translateNumberToEnglish(int(number[1:]));
        if int(number) < 20:
            return NUMBER_CONSTANT[int(number)];
        elif int(number) < 100:
            if str(number)[1] == '0':
                return IN_HUNDRED_CONSTANT[int(str(number)[0])];
            else:
                return IN_HUNDRED_CONSTANT[int(str(number)[0])] + " " + NUMBER_CONSTANT[int(str(number)[1])];
        else:
            strNumber=str(number)
            numberArray = str(strNumber).split(",");
            stringResult = "";
            groupCount = len(numberArray) + 1;
            for groupNumber in numberArray:
                if groupCount > 1 and groupNumber[0:] != "000":
                    stringResult += str(getUnderThreeNumberString(str(groupNumber))) + " ";
                else:
                    break;
                groupCount -= 1;
                if groupCount > 1:
                    stringResult += BASE_CONSTANT[groupCount] + " ";
            endPoint = len(stringResult) - len(" hundred,");
            return stringResult;

def getUnderThreeNumberString(number):
    if str(number).isnumeric() and len(number) < 4:
        if len(number) < 3:
            return translateNumberToEnglish(int(number));
        elif len(number) == 3 and number[0:] == "000":
            return " ";
        elif len(number) == 3 and number[1:] == "00":
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1];
        else:    
            return NUMBER_CONSTANT[int(number[0])] + "  " + BASE_CONSTANT[1] + " and " + translateNumberToEnglish((number[1:]));
    
def translateTimeToEnglish(t):
    t=t.split(':')
    if t[1]!='00':
      return translateNumberToEnglish(t[0])+' '+translateNumberToEnglish(t[1])
    else:
      return translateNumberToEnglish(t[0])+' '+'o\'clock'

def span_typer(s):
    if s.isnumeric():
        return "number"
    if s.find(':')>=0:
        s=s.split(':')
        if len(s)==2:
            if s[0].isnumeric() and s[1].isnumeric():
                return "time"
    return "none"


def span_error_detect(original_text,new_text,span_list):
#input:original_text,new_text,one span_info [slot,slot,span,start,end]
#output: is_span_changed? ,is_span_found? , new span_info [slot,slot,new span,new start,new end]
    span=span_list[2].lower()
    span_type=span_typer(span)
    if span_type=="time":
        span2=translateTimeToEnglish(span)
    if span_type=="number":
        span2=translateNumberToEnglish(span)
    if span_type=="none":
        span2=span
    span_changed,span_found=0,0
    if new_text.find(span)>=0:
        span_changed,span_found=0,1
        span_start=new_text.count(' ',0,new_text.find(span))
        span_end=span_start+len(span.split())-1
        new_span_list=[span_list[0],span_list[1],span,span_start,span_end]
    elif new_text.find(span2)>=0:
        span_changed,span_found=1,1
        span=span2
        span_start=new_text.count(' ',0,new_text.find(span))
        span_end=span_start+len(span.split())-1
        new_span_list=[span_list[0],span_list[1],span,span_start,span_end]
    else:
        span=span2
        span_words=span.split()
        new_words=new_text.split()
        tag=[0]*len(new_words)
        for i in range(len(new_words)):
            if new_words[i] in span_words:
                tag[i]=1
        max_start,max_end=-1,-1
        max_match=-1
        for i in range(len(new_words)):
            if tag[i]==1:
                anchor=i
                start=i-span_words.index(new_words[anchor])
                end=min(start+len(span_words)-1,len(new_words)-1)
                match=0
                s,e=start,end
                while new_words[s][0]!=span_words[s-start][0] and s<anchor:
                    s+=1
                while new_words[e][0]!=span_words[e-start][0] and e>anchor:
                    e-=1
                for j in range(s,e+1):
                    if tag[j]==1:
                        match+=1
                    elif new_words[j][0]==span_words[j-start][0]:
                        match+=0.5
                if match>=max(len(span_words)/2,1,max_match):
                    max_match=match
                    max_start,max_end=s,e
        if max_match>=max(len(span_words)-1,1):
            span_changed,span_found=1,1
            new_span_list=[span_list[0],span_list[1],' '.join(new_words[max_start:max_end+1]),max_start,max_end]
    
    
    if span_found==0:
        new_span_list=[span_list[0],span_list[1],span_list[2],"not found","not found"]
    return span_changed,span_found,new_span_list
